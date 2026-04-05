#!/usr/bin/env python3
"""Incremental sync: Postgres → Neo4j + embeddings for new moltbook data.

Runs as a daemon alongside the scraper. Polls for new rows, syncs to Neo4j,
and embeds new posts with BGE-small. Gracefully handles Neo4j being down.

Usage:
    python incremental_sync.py              # run once
    python incremental_sync.py --daemon     # run continuously
    python incremental_sync.py --embed-only # only embed, skip Neo4j
"""

import argparse
import logging
import os
import signal
import sys
import time

import psycopg
from pgvector.psycopg import register_vector

DB = os.environ.get("DATABASE_URL", "dbname=moltbook")
NEO4J_URI = "bolt://localhost:7687"
BATCH = 5000
EMBED_BATCH = 256
POLL_INTERVAL = 60  # seconds between sync cycles
AGG_EDGE_INTERVAL = 10  # rebuild aggregated edges every N cycles

log = logging.getLogger("incremental_sync")
_shutdown = False


def handle_signal(sig, frame):
    global _shutdown
    log.info("Shutdown requested...")
    _shutdown = True


# ---------------------------------------------------------------------------
# Neo4j sync (fault-tolerant)
# ---------------------------------------------------------------------------

def neo4j_sync_new_data(pg_conn, last_sync):
    """Sync new Postgres rows to Neo4j. Returns new high-water mark."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI)
        driver.verify_connectivity()
    except Exception as e:
        log.warning(f"Neo4j unavailable, skipping graph sync: {e}")
        return last_sync

    try:
        new_mark = last_sync

        # Authors
        with pg_conn.cursor() as cur:
            cur.execute("""
                SELECT id::text, name, display_name, karma, followers_count,
                       fetched_at
                FROM authors WHERE fetched_at > %s ORDER BY fetched_at LIMIT %s
            """, (last_sync, BATCH))
            rows = cur.fetchall()

        if rows:
            batch = [{"id": r[0], "name": r[1], "display_name": r[2],
                      "karma": r[3], "followers_count": r[4]} for r in rows]
            with driver.session() as s:
                s.run("""
                    UNWIND $batch AS r
                    MERGE (a:Author {id: r.id})
                    SET a.name = r.name, a.display_name = r.display_name,
                        a.karma = r.karma, a.followers_count = r.followers_count
                """, batch=batch)
            new_mark = max(new_mark, rows[-1][5])
            log.info(f"  Neo4j authors: {len(rows)}")

        # Submolts
        with pg_conn.cursor() as cur:
            cur.execute("""
                SELECT id::text, name, display_name, subscriber_count, post_count,
                       fetched_at
                FROM submolts WHERE fetched_at > %s ORDER BY fetched_at LIMIT %s
            """, (last_sync, BATCH))
            rows = cur.fetchall()

        if rows:
            batch = [{"id": r[0], "name": r[1], "display_name": r[2],
                      "subscriber_count": r[3], "post_count": r[4]} for r in rows]
            with driver.session() as s:
                s.run("""
                    UNWIND $batch AS r
                    MERGE (s:Submolt {id: r.id})
                    SET s.name = r.name, s.display_name = r.display_name,
                        s.subscriber_count = r.subscriber_count,
                        s.post_count = r.post_count
                """, batch=batch)
            new_mark = max(new_mark, rows[-1][5])
            log.info(f"  Neo4j submolts: {len(rows)}")

        # Posts (+ relationships)
        with pg_conn.cursor() as cur:
            cur.execute("""
                SELECT id::text, title, score, comment_count, created_at::text,
                       author_id::text, submolt_id::text, fetched_at
                FROM posts WHERE fetched_at > %s ORDER BY fetched_at LIMIT %s
            """, (last_sync, BATCH))
            rows = cur.fetchall()

        if rows:
            batch = [{"id": r[0], "title": r[1], "score": r[2],
                      "comment_count": r[3], "created_at": r[4]} for r in rows]
            with driver.session() as s:
                s.run("""
                    UNWIND $batch AS r
                    MERGE (p:Post {id: r.id})
                    SET p.title = r.title, p.score = r.score,
                        p.comment_count = r.comment_count,
                        p.created_at = r.created_at
                """, batch=batch)

            wrote = [{"aid": r[5], "pid": r[0]} for r in rows if r[5]]
            if wrote:
                with driver.session() as s:
                    s.run("""
                        UNWIND $batch AS r
                        MATCH (a:Author {id: r.aid})
                        MATCH (p:Post {id: r.pid})
                        MERGE (a)-[:WROTE]->(p)
                    """, batch=wrote)

            in_sub = [{"pid": r[0], "sid": r[6]} for r in rows if r[6]]
            if in_sub:
                with driver.session() as s:
                    s.run("""
                        UNWIND $batch AS r
                        MATCH (p:Post {id: r.pid})
                        MATCH (s:Submolt {id: r.sid})
                        MERGE (p)-[:IN_SUBMOLT]->(s)
                    """, batch=in_sub)

            new_mark = max(new_mark, rows[-1][7])
            log.info(f"  Neo4j posts: {len(rows)}")

        # Comments (+ relationships)
        with pg_conn.cursor() as cur:
            cur.execute("""
                SELECT id::text, score, created_at::text,
                       post_id::text, parent_id::text, author_id::text,
                       fetched_at
                FROM comments WHERE fetched_at > %s ORDER BY fetched_at LIMIT %s
            """, (last_sync, BATCH))
            rows = cur.fetchall()

        if rows:
            batch = [{"id": r[0], "score": r[1], "created_at": r[2]}
                     for r in rows]
            with driver.session() as s:
                s.run("""
                    UNWIND $batch AS r
                    MERGE (c:Comment {id: r.id})
                    SET c.score = r.score, c.created_at = r.created_at
                """, batch=batch)

            on_post = [{"cid": r[0], "pid": r[3]} for r in rows if r[3]]
            if on_post:
                with driver.session() as s:
                    s.run("""
                        UNWIND $batch AS r
                        MATCH (c:Comment {id: r.cid})
                        MATCH (p:Post {id: r.pid})
                        MERGE (c)-[:ON_POST]->(p)
                    """, batch=on_post)

            reply = [{"cid": r[0], "pid": r[4]} for r in rows if r[4]]
            if reply:
                with driver.session() as s:
                    s.run("""
                        UNWIND $batch AS r
                        MATCH (c:Comment {id: r.cid})
                        MATCH (p:Comment {id: r.pid})
                        MERGE (c)-[:REPLY_TO]->(p)
                    """, batch=reply)

            wrote = [{"aid": r[5], "cid": r[0]} for r in rows if r[5]]
            if wrote:
                with driver.session() as s:
                    s.run("""
                        UNWIND $batch AS r
                        MATCH (a:Author {id: r.aid})
                        MATCH (c:Comment {id: r.cid})
                        MERGE (a)-[:WROTE]->(c)
                    """, batch=wrote)

            new_mark = max(new_mark, rows[-1][6])
            log.info(f"  Neo4j comments: {len(rows)}")

        driver.close()
        return new_mark

    except Exception as e:
        log.warning(f"Neo4j sync error (non-fatal): {e}")
        try:
            driver.close()
        except Exception:
            pass
        return last_sync


def rebuild_aggregated_edges():
    """Rebuild REPLIED_TO and POSTED_IN edges."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI)
        driver.verify_connectivity()

        with driver.session() as s:
            result = s.run("""
                MATCH (a:Author)-[:WROTE]->(:Comment)-[:REPLY_TO]->
                      (:Comment)<-[:WROTE]-(b:Author)
                WHERE a <> b
                WITH a, b, count(*) AS cnt
                MERGE (a)-[r:REPLIED_TO]->(b)
                SET r.count = cnt
                RETURN count(r) AS c
            """)
            log.info(f"  REPLIED_TO: {result.single()['c']} edges")

        with driver.session() as s:
            result = s.run("""
                MATCH (a:Author)-[:WROTE]->(:Post)-[:IN_SUBMOLT]->(s:Submolt)
                WITH a, s, count(*) AS cnt
                MERGE (a)-[r:POSTED_IN]->(s)
                SET r.count = cnt
                RETURN count(r) AS c
            """)
            log.info(f"  POSTED_IN: {result.single()['c']} edges")

        driver.close()
    except Exception as e:
        log.warning(f"Aggregated edge rebuild failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Embedding sync
# ---------------------------------------------------------------------------

def embed_new_posts(pg_conn, model):
    """Embed posts that don't have embeddings yet (skip mbc-20 spam)."""
    with pg_conn.cursor() as cur:
        cur.execute("""
            SELECT id, coalesce(title, '') || ' ' || coalesce(content, '')
            FROM posts WHERE embedding IS NULL
              AND content NOT LIKE '%%mbc-20%%'
              AND content NOT LIKE '%%mbc20.xyz%%'
            ORDER BY created_at DESC LIMIT %s
        """, (EMBED_BATCH,))
        rows = cur.fetchall()

    if not rows:
        return 0

    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]

    embeddings = model.encode(texts, normalize_embeddings=True,
                              show_progress_bar=False)

    with pg_conn.cursor() as cur:
        cur.executemany(
            "UPDATE posts SET embedding = %s WHERE id = %s",
            [(emb.tolist(), pid) for emb, pid in zip(embeddings, ids)]
        )
    pg_conn.commit()

    log.info(f"  Embedded {len(rows)} posts")
    return len(rows)


# ---------------------------------------------------------------------------
# Sync state
# ---------------------------------------------------------------------------

def get_last_sync(pg_conn):
    with pg_conn.cursor() as cur:
        cur.execute("SELECT value FROM scrape_state WHERE key = 'neo4j_last_sync'")
        row = cur.fetchone()
        if row and row[0]:
            from datetime import datetime, timezone
            return datetime.fromisoformat(row[0])
    from datetime import datetime, timezone
    return datetime(2000, 1, 1, tzinfo=timezone.utc)


def set_last_sync(pg_conn, ts):
    with pg_conn.cursor() as cur:
        cur.execute("""
            INSERT INTO scrape_state (key, value, updated_at)
            VALUES ('neo4j_last_sync', %s, NOW())
            ON CONFLICT (key) DO UPDATE SET value = %s, updated_at = NOW()
        """, (ts.isoformat(), ts.isoformat()))
    pg_conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_once(pg_conn, model, skip_neo4j=False):
    """Single sync cycle."""
    if not skip_neo4j:
        last_sync = get_last_sync(pg_conn)
        new_mark = neo4j_sync_new_data(pg_conn, last_sync)
        if new_mark != last_sync:
            set_last_sync(pg_conn, new_mark)

    if model:
        embed_new_posts(pg_conn, model)


def run_daemon(pg_conn, model, skip_neo4j=False):
    """Run continuously."""
    cycle = 0
    while not _shutdown:
        cycle += 1
        log.info(f"=== Sync cycle {cycle} ===")

        run_once(pg_conn, model, skip_neo4j=skip_neo4j)

        # Rebuild aggregated edges periodically
        if not skip_neo4j and cycle % AGG_EDGE_INTERVAL == 0:
            log.info("Rebuilding aggregated edges...")
            rebuild_aggregated_edges()

        # Keep embedding until caught up
        if model:
            while not _shutdown:
                embedded = embed_new_posts(pg_conn, model)
                if embedded < EMBED_BATCH:
                    break

        for _ in range(POLL_INTERVAL):
            if _shutdown:
                break
            time.sleep(1)

    log.info("Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Incremental sync: Postgres → Neo4j + embeddings")
    parser.add_argument("--daemon", action="store_true",
                        help="Run continuously")
    parser.add_argument("--embed-only", action="store_true",
                        help="Only embed, skip Neo4j sync")
    parser.add_argument("--no-embed", action="store_true",
                        help="Skip embedding (Neo4j sync only)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Load embedding model (once, kept in memory)
    model = None
    if not args.no_embed:
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model...")
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        log.info("Model loaded.")

    with psycopg.connect(DB) as pg_conn:
        register_vector(pg_conn)

        if args.daemon:
            run_daemon(pg_conn, model, skip_neo4j=args.embed_only)
        else:
            run_once(pg_conn, model, skip_neo4j=args.embed_only)


if __name__ == "__main__":
    main()
