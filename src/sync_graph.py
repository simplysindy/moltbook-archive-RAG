#!/usr/bin/env python3
"""Sync PostgreSQL moltbook data into Neo4j graph.

Bulk load using UNWIND + MERGE for idempotent batch processing.
Creates nodes (Author, Submolt, Post, Comment) and all relationships.

Usage:
    python sync_graph.py              # full sync
    python sync_graph.py --skip-agg   # skip aggregated edges (faster re-runs)
"""

import argparse
import logging
import os
import sys
import time

import psycopg
from neo4j import GraphDatabase

DB = os.environ.get("DATABASE_URL", "dbname=moltbook")
NEO4J_URI = "bolt://localhost:7687"
BATCH = 5000

log = logging.getLogger("sync_graph")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def batched_read(pg_conn, sql, batch_size=BATCH):
    """Yield row batches using keyset pagination on id (first column)."""
    last_id = "00000000-0000-0000-0000-000000000000"
    while True:
        with pg_conn.cursor() as cur:
            cur.execute(sql, (last_id, batch_size))
            rows = cur.fetchall()
        if not rows:
            break
        yield rows
        if len(rows) < batch_size:
            break
        last_id = rows[-1][0]


def neo_run(driver, cypher, **kwargs):
    """Run a single auto-commit Cypher statement."""
    with driver.session() as s:
        return s.run(cypher, **kwargs)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def create_constraints(driver):
    for label in ("Author", "Submolt", "Post", "Comment"):
        neo_run(driver,
                f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.id IS UNIQUE")
    log.info("Constraints created.")


# ---------------------------------------------------------------------------
# Node sync
# ---------------------------------------------------------------------------

def sync_authors(driver, pg_conn):
    total = 0
    t0 = time.time()
    for rows in batched_read(pg_conn, """
        SELECT id::text, name, display_name, karma, followers_count
        FROM authors WHERE id > %s::uuid ORDER BY id LIMIT %s
    """):
        batch = [{"id": r[0], "name": r[1], "display_name": r[2],
                  "karma": r[3], "followers_count": r[4]} for r in rows]
        neo_run(driver, """
            UNWIND $batch AS r
            MERGE (a:Author {id: r.id})
            SET a.name = r.name, a.display_name = r.display_name,
                a.karma = r.karma, a.followers_count = r.followers_count
        """, batch=batch)
        total += len(rows)
        log.info(f"  Authors: {total} ({total/(time.time()-t0):.0f}/s)")
    log.info(f"Authors done: {total} in {time.time()-t0:.1f}s")


def sync_submolts(driver, pg_conn):
    total = 0
    t0 = time.time()
    for rows in batched_read(pg_conn, """
        SELECT id::text, name, display_name, subscriber_count, post_count
        FROM submolts WHERE id > %s::uuid ORDER BY id LIMIT %s
    """):
        batch = [{"id": r[0], "name": r[1], "display_name": r[2],
                  "subscriber_count": r[3], "post_count": r[4]} for r in rows]
        neo_run(driver, """
            UNWIND $batch AS r
            MERGE (s:Submolt {id: r.id})
            SET s.name = r.name, s.display_name = r.display_name,
                s.subscriber_count = r.subscriber_count, s.post_count = r.post_count
        """, batch=batch)
        total += len(rows)
        log.info(f"  Submolts: {total}")
    log.info(f"Submolts done: {total} in {time.time()-t0:.1f}s")


def sync_posts(driver, pg_conn):
    """Sync Post nodes + WROTE and IN_SUBMOLT relationships."""
    total = 0
    t0 = time.time()
    for rows in batched_read(pg_conn, """
        SELECT id::text, title, score, comment_count,
               created_at::text, author_id::text, submolt_id::text
        FROM posts WHERE id > %s::uuid ORDER BY id LIMIT %s
    """):
        # Post nodes (no content — stays in Postgres)
        batch = [{"id": r[0], "title": r[1], "score": r[2],
                  "comment_count": r[3], "created_at": r[4]} for r in rows]
        neo_run(driver, """
            UNWIND $batch AS r
            MERGE (p:Post {id: r.id})
            SET p.title = r.title, p.score = r.score,
                p.comment_count = r.comment_count, p.created_at = r.created_at
        """, batch=batch)

        # Author -[:WROTE]-> Post
        wrote = [{"aid": r[5], "pid": r[0]} for r in rows if r[5]]
        if wrote:
            neo_run(driver, """
                UNWIND $batch AS r
                MATCH (a:Author {id: r.aid})
                MATCH (p:Post {id: r.pid})
                MERGE (a)-[:WROTE]->(p)
            """, batch=wrote)

        # Post -[:IN_SUBMOLT]-> Submolt
        in_sub = [{"pid": r[0], "sid": r[6]} for r in rows if r[6]]
        if in_sub:
            neo_run(driver, """
                UNWIND $batch AS r
                MATCH (p:Post {id: r.pid})
                MATCH (s:Submolt {id: r.sid})
                MERGE (p)-[:IN_SUBMOLT]->(s)
            """, batch=in_sub)

        total += len(rows)
        elapsed = time.time() - t0
        log.info(f"  Posts: {total} ({total/elapsed:.0f}/s)")

    log.info(f"Posts done: {total} in {time.time()-t0:.1f}s")


def sync_comments(driver, pg_conn):
    """Sync Comment nodes + ON_POST, REPLY_TO, and WROTE relationships."""
    total = 0
    t0 = time.time()
    for rows in batched_read(pg_conn, """
        SELECT id::text, score, created_at::text,
               post_id::text, parent_id::text, author_id::text
        FROM comments WHERE id > %s::uuid ORDER BY id LIMIT %s
    """):
        # Comment nodes
        batch = [{"id": r[0], "score": r[1], "created_at": r[2]} for r in rows]
        neo_run(driver, """
            UNWIND $batch AS r
            MERGE (c:Comment {id: r.id})
            SET c.score = r.score, c.created_at = r.created_at
        """, batch=batch)

        # Comment -[:ON_POST]-> Post
        on_post = [{"cid": r[0], "pid": r[3]} for r in rows if r[3]]
        if on_post:
            neo_run(driver, """
                UNWIND $batch AS r
                MATCH (c:Comment {id: r.cid})
                MATCH (p:Post {id: r.pid})
                MERGE (c)-[:ON_POST]->(p)
            """, batch=on_post)

        # Comment -[:REPLY_TO]-> Comment
        reply = [{"cid": r[0], "pid": r[4]} for r in rows if r[4]]
        if reply:
            neo_run(driver, """
                UNWIND $batch AS r
                MATCH (c:Comment {id: r.cid})
                MATCH (p:Comment {id: r.pid})
                MERGE (c)-[:REPLY_TO]->(p)
            """, batch=reply)

        # Author -[:WROTE]-> Comment
        wrote = [{"aid": r[5], "cid": r[0]} for r in rows if r[5]]
        if wrote:
            neo_run(driver, """
                UNWIND $batch AS r
                MATCH (a:Author {id: r.aid})
                MATCH (c:Comment {id: r.cid})
                MERGE (a)-[:WROTE]->(c)
            """, batch=wrote)

        total += len(rows)
        elapsed = time.time() - t0
        log.info(f"  Comments: {total} ({total/elapsed:.0f}/s)")

    log.info(f"Comments done: {total} in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Aggregated edges (computed from graph)
# ---------------------------------------------------------------------------

def build_aggregated_edges(driver):
    """Build REPLIED_TO and POSTED_IN aggregated edges."""
    log.info("Building REPLIED_TO edges...")
    t0 = time.time()
    with driver.session() as s:
        result = s.run("""
            MATCH (a:Author)-[:WROTE]->(:Comment)-[:REPLY_TO]->(:Comment)<-[:WROTE]-(b:Author)
            WHERE a <> b
            WITH a, b, count(*) AS cnt
            MERGE (a)-[r:REPLIED_TO]->(b)
            SET r.count = cnt
            RETURN count(r) AS c
        """)
        count = result.single()["c"]
    log.info(f"  REPLIED_TO: {count} edges in {time.time()-t0:.1f}s")

    log.info("Building POSTED_IN edges...")
    t0 = time.time()
    with driver.session() as s:
        result = s.run("""
            MATCH (a:Author)-[:WROTE]->(:Post)-[:IN_SUBMOLT]->(s:Submolt)
            WITH a, s, count(*) AS cnt
            MERGE (a)-[r:POSTED_IN]->(s)
            SET r.count = cnt
            RETURN count(r) AS c
        """)
        count = result.single()["c"]
    log.info(f"  POSTED_IN: {count} edges in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sync moltbook Postgres → Neo4j")
    parser.add_argument("--skip-agg", action="store_true",
                        help="Skip aggregated edge computation")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log.info("Connecting to Neo4j and PostgreSQL...")
    driver = GraphDatabase.driver(NEO4J_URI)
    driver.verify_connectivity()
    log.info("Connected.")

    with psycopg.connect(DB) as pg_conn:
        create_constraints(driver)
        sync_authors(driver, pg_conn)
        sync_submolts(driver, pg_conn)
        sync_posts(driver, pg_conn)
        sync_comments(driver, pg_conn)
        if not args.skip_agg:
            build_aggregated_edges(driver)

    driver.close()
    log.info("Sync complete.")


if __name__ == "__main__":
    main()
