#!/usr/bin/env python3
"""Seed the moltbook database from HuggingFace parquet files."""

import os
import pandas as pd
import psycopg

DB = os.environ.get("DATABASE_URL", "dbname=moltbook")
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def seed_submolts(conn):
    df = pd.read_parquet(os.path.join(DATA_DIR, "submolts.parquet"))
    print(f"Seeding {len(df)} submolts...")

    with conn.cursor() as cur:
        for _, row in df.iterrows():
            cur.execute("""
                INSERT INTO submolts (id, name, display_name, description,
                    subscriber_count, created_at, last_activity_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (
                row["id"], row["name"], row["display_name"],
                row.get("description"), row.get("subscriber_count", 0),
                row.get("created_at"), row.get("last_activity_at"),
            ))
    conn.commit()
    print("  Submolts done.")


def seed_posts(conn):
    df = pd.read_parquet(os.path.join(DATA_DIR, "posts.parquet"))
    print(f"Seeding {len(df)} posts...")

    # First pass: collect unique submolts from posts that aren't in submolts table yet
    with conn.cursor() as cur:
        inserted = 0
        for _, row in df.iterrows():
            post = row["post"]
            submolt = post.get("submolt", {}) or {}
            submolt_id = submolt.get("id")

            # Ensure submolt exists
            if submolt_id:
                cur.execute("""
                    INSERT INTO submolts (id, name, display_name)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """, (submolt_id, submolt.get("name", "unknown"),
                      submolt.get("display_name")))

            cur.execute("""
                INSERT INTO posts (id, title, content, url, submolt_id,
                    upvotes, downvotes, comment_count, topic_label,
                    toxic_level, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (
                row["id"], post.get("title"), post.get("content"),
                post.get("url"), submolt_id,
                post.get("upvotes", 0), post.get("downvotes", 0),
                post.get("comment_count", 0),
                row.get("topic_label"), row.get("toxic_level"),
                post.get("created_at"),
            ))
            inserted += 1
            if inserted % 5000 == 0:
                conn.commit()
                print(f"  {inserted}/{len(df)} posts...")

    conn.commit()
    print(f"  Posts done. {inserted} processed.")


def main():
    with psycopg.connect(DB) as conn:
        seed_submolts(conn)
        seed_posts(conn)

        # Verify
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM submolts")
            print(f"\nSubmolts in DB: {cur.fetchone()[0]}")
            cur.execute("SELECT count(*) FROM posts")
            print(f"Posts in DB: {cur.fetchone()[0]}")


if __name__ == "__main__":
    main()
