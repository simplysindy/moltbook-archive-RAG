#!/usr/bin/env python3
"""Embed all moltbook posts using BGE-small-en-v1.5 and store in pgvector.

Processes in batches, resumable (skips already-embedded posts).
Expects: posts table with `embedding vector(384)` column.

Usage:
    python embed_posts.py                  # embed all un-embedded posts
    python embed_posts.py --batch 500      # custom batch size
    python embed_posts.py --limit 10000    # stop after N posts
"""

import argparse
import logging
import os
import sys
import time

import psycopg
from pgvector.psycopg import register_vector

DB = os.environ.get("DATABASE_URL", "dbname=moltbook")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIMS = 384
BATCH_SIZE = 256

# Skip spam posts (mbc-20 token mints/claims)
SPAM_FILTER = """
    AND content NOT LIKE '%%mbc-20%%'
    AND content NOT LIKE '%%mbc20.xyz%%'
"""

log = logging.getLogger("embed_posts")


def load_model():
    """Load BGE-small model (downloads on first run)."""
    from sentence_transformers import SentenceTransformer
    log.info(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    log.info("Model loaded.")
    return model


def get_pending_count(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM posts WHERE embedding IS NULL" + SPAM_FILTER)
        return cur.fetchone()[0]


def embed_batches(conn, model, batch_size=BATCH_SIZE, limit=None):
    """Embed posts in batches, updating Postgres as we go."""
    pending = get_pending_count(conn)
    log.info(f"Posts needing embedding: {pending}")
    if limit:
        pending = min(pending, limit)
        log.info(f"  (limited to {limit})")

    total = 0
    t0 = time.time()

    while total < pending:
        # Fetch batch of un-embedded posts
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, coalesce(title, '') || ' ' || coalesce(content, '')
                FROM posts
                WHERE embedding IS NULL
                """ + SPAM_FILTER + """
                ORDER BY id
                LIMIT %s
            """, (batch_size,))
            rows = cur.fetchall()

        if not rows:
            break

        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]

        # Embed
        embeddings = model.encode(texts, normalize_embeddings=True,
                                  show_progress_bar=False)

        # Write back
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE posts SET embedding = %s WHERE id = %s",
                [(emb.tolist(), pid) for emb, pid in zip(embeddings, ids)]
            )
        conn.commit()

        total += len(rows)
        elapsed = time.time() - t0
        rate = total / elapsed if elapsed > 0 else 0
        remaining = (pending - total) / rate if rate > 0 else 0
        log.info(f"  Embedded: {total}/{pending} ({rate:.0f}/s, ~{remaining/60:.0f}m remaining)")

    log.info(f"Done. {total} posts embedded in {(time.time()-t0)/60:.1f}m")
    return total


def create_index():
    """Create HNSW index on embedding column (requires autocommit)."""
    log.info("Creating HNSW index (this may take a while)...")
    t0 = time.time()
    with psycopg.connect(DB, autocommit=True) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_posts_embedding
                ON posts USING hnsw (embedding vector_cosine_ops)
            """)
    log.info(f"HNSW index created in {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Embed moltbook posts with BGE-small")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--limit", type=int, help="Max posts to embed")
    parser.add_argument("--index-only", action="store_true",
                        help="Only create the HNSW index, skip embedding")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    with psycopg.connect(DB) as conn:
        register_vector(conn)

        if args.index_only:
            create_index()
            return

        model = load_model()
        embedded = embed_batches(conn, model, batch_size=args.batch, limit=args.limit)

        if embedded > 0:
            create_index()


if __name__ == "__main__":
    main()
