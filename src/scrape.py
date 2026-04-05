#!/usr/bin/env python3
"""Incremental Moltbook API scraper. Resumable, rate-limited.

Daemon mode (--daemon) runs 24/7 in a loop:
  1. Fetch new posts (newest-first until caught up)
  2. Backfill comments for posts missing them
  3. Refresh submolts
  4. Sleep, repeat
"""

import argparse
import logging
import os
import re
import signal
import sys
import time
import httpx
import psycopg

MINT_RE = re.compile(r"mint", re.IGNORECASE)

API = "https://www.moltbook.com/api/v1"
DB = os.environ.get("DATABASE_URL", "dbname=moltbook")
RATE_DELAY = 1.1  # seconds between requests (~55 req/min, under 60 limit)
POLL_INTERVAL = 60  # seconds to wait when fully caught up before checking again
MAX_RETRIES = 5

log = logging.getLogger("moltbook")
_shutdown = False


def handle_signal(sig, frame):
    global _shutdown
    log.info("Shutdown requested, finishing current batch...")
    _shutdown = True


def api_get(client, url, **kwargs):
    """GET with retry + exponential backoff on 429/5xx."""
    for attempt in range(MAX_RETRIES):
        resp = client.get(url, **kwargs)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
            log.warning(f"Rate limited (429). Waiting {retry_after}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(retry_after)
            continue
        if resp.status_code >= 500:
            wait = 2 ** (attempt + 1)
            log.warning(f"Server error ({resp.status_code}). Waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    # Final attempt — let it raise if it fails
    resp = client.get(url, **kwargs)
    resp.raise_for_status()
    return resp


def get_state(conn, key):
    with conn.cursor() as cur:
        cur.execute("SELECT value FROM scrape_state WHERE key = %s", (key,))
        row = cur.fetchone()
        return row[0] if row else None


def set_state(conn, key, value):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO scrape_state (key, value, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (key) DO UPDATE SET value = %s, updated_at = NOW()
        """, (key, value, value))
    conn.commit()


def upsert_author(cur, author):
    if not author or not author.get("id"):
        return None
    cur.execute("""
        INSERT INTO authors (id, name, display_name, karma,
            followers_count, following_count, created_at, fetched_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            display_name = EXCLUDED.display_name,
            karma = EXCLUDED.karma,
            followers_count = EXCLUDED.followers_count,
            following_count = EXCLUDED.following_count,
            fetched_at = NOW()
    """, (
        author["id"], author.get("name", "unknown"),
        author.get("display_name"), author.get("karma", 0),
        author.get("followers_count", 0), author.get("following_count", 0),
        author.get("created_at"),
    ))
    return author["id"]


def upsert_submolt(cur, submolt):
    if not submolt or not submolt.get("id"):
        return None
    cur.execute("""
        INSERT INTO submolts (id, name, display_name, description,
            subscriber_count, post_count, is_nsfw, is_private,
            created_at, fetched_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            display_name = EXCLUDED.display_name,
            description = EXCLUDED.description,
            subscriber_count = EXCLUDED.subscriber_count,
            post_count = EXCLUDED.post_count,
            is_nsfw = EXCLUDED.is_nsfw,
            is_private = EXCLUDED.is_private,
            fetched_at = NOW()
    """, (
        submolt["id"], submolt.get("name", "unknown"),
        submolt.get("display_name"), submolt.get("description"),
        submolt.get("subscriber_count", 0), submolt.get("post_count", 0),
        submolt.get("is_nsfw", False), submolt.get("is_private", False),
        submolt.get("created_at"),
    ))
    return submolt["id"]


def upsert_post(cur, post, author_id, submolt_id):
    cur.execute("""
        INSERT INTO posts (id, title, content, type, url, author_id, submolt_id,
            upvotes, downvotes, score, hot_score, comment_count,
            is_pinned, is_locked, is_deleted, is_spam,
            created_at, updated_at, fetched_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (id) DO UPDATE SET
            upvotes = EXCLUDED.upvotes,
            downvotes = EXCLUDED.downvotes,
            score = EXCLUDED.score,
            hot_score = EXCLUDED.hot_score,
            comment_count = EXCLUDED.comment_count,
            is_pinned = EXCLUDED.is_pinned,
            is_locked = EXCLUDED.is_locked,
            is_deleted = EXCLUDED.is_deleted,
            is_spam = EXCLUDED.is_spam,
            updated_at = EXCLUDED.updated_at,
            fetched_at = NOW()
    """, (
        post["id"], post.get("title"), post.get("content"),
        post.get("type", "text"), post.get("url"),
        author_id, submolt_id,
        post.get("upvotes", 0), post.get("downvotes", 0),
        post.get("score", 0), post.get("hot_score"),
        post.get("comment_count", 0),
        post.get("is_pinned", False), post.get("is_locked", False),
        post.get("is_deleted", False), post.get("is_spam", False),
        post.get("created_at"), post.get("updated_at"),
    ))


def upsert_comment(cur, comment, post_id):
    author_id = upsert_author(cur, comment.get("author"))
    parent_id = comment.get("parent_id") or comment.get("parent_comment_id")

    cur.execute("""
        INSERT INTO comments (id, post_id, parent_id, author_id, content,
            upvotes, downvotes, score, is_deleted, created_at, fetched_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (id) DO UPDATE SET
            content = EXCLUDED.content,
            upvotes = EXCLUDED.upvotes,
            downvotes = EXCLUDED.downvotes,
            score = EXCLUDED.score,
            is_deleted = EXCLUDED.is_deleted,
            fetched_at = NOW()
    """, (
        comment["id"], post_id, parent_id, author_id,
        comment.get("content"), comment.get("upvotes", 0),
        comment.get("downvotes", 0), comment.get("score", 0),
        comment.get("is_deleted", False), comment.get("created_at"),
    ))

    # Recurse into replies
    for reply in comment.get("replies", []):
        upsert_comment(cur, reply, post_id)


def scrape_submolts(conn, client):
    """Fetch all submolts from the API."""
    log.info("Scraping submolts...")
    resp = api_get(client, f"{API}/submolts", params={"limit": 1000})
    data = resp.json()

    count = 0
    with conn.cursor() as cur:
        for s in data.get("data", data.get("submolts", [])):
            upsert_submolt(cur, s)
            count += 1
    conn.commit()
    log.info(f"  Upserted {count} submolts.")


def scrape_posts(conn, client, max_pages=None):
    """Paginate through posts, newest first. Resumes from last cursor."""
    cursor = get_state(conn, "posts_cursor")
    page = 0
    total = 0

    log.info(f"Scraping posts (cursor: {cursor or 'start'})...")

    while True:
        params = {"sort": "new", "limit": 100}
        if cursor:
            params["cursor"] = cursor

        resp = api_get(client, f"{API}/posts", params=params)
        data = resp.json()

        posts = data.get("data", data.get("posts", []))
        if not posts:
            break

        # Check how many posts on this page we already have
        post_ids = [p["id"] for p in posts]
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM posts WHERE id = ANY(%s)", (post_ids,))
            existing_ids = {row[0] for row in cur.fetchall()}

        with conn.cursor() as cur:
            for post in posts:
                author_id = upsert_author(cur, post.get("author"))
                submolt_id = upsert_submolt(cur, post.get("submolt"))
                upsert_post(cur, post, author_id, submolt_id)
                total += 1
        conn.commit()

        page += 1
        has_more = data.get("has_more", False)
        cursor = data.get("next_cursor")

        if cursor:
            set_state(conn, "posts_cursor", cursor)

        new_on_page = len(posts) - len(existing_ids)
        log.info(f"  Page {page}: {len(posts)} posts ({new_on_page} new, total: {total})")

        # If entire page was already in DB, we've caught up — stop
        if new_on_page == 0:
            log.info("  All posts on page already known. Stopping post scrape.")
            break

        if _shutdown or not has_more or (max_pages and page >= max_pages):
            break

        time.sleep(RATE_DELAY)

    # Reset cursor so next cycle starts from newest
    set_state(conn, "posts_cursor", "")
    log.info(f"  Done. {total} posts scraped across {page} pages.")
    return total


def scrape_comments(conn, client, max_posts=None):
    """Fetch comments for posts that haven't had comments fetched yet."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, comment_count FROM posts
            WHERE comment_count > 0 AND comments_fetched_at IS NULL
            ORDER BY created_at DESC
            LIMIT %s
        """, (max_posts or 2147483647,))
        posts_to_fetch = cur.fetchall()

    if not posts_to_fetch:
        log.info("No posts need comment backfill.")
        return

    log.info(f"Scraping comments for {len(posts_to_fetch)} posts...")
    total = 0

    for i, (post_id, expected_count) in enumerate(posts_to_fetch):
        # Paginate through all comments for this post
        comment_cursor = None
        post_comments = 0
        try:
            while True:
                params = {"limit": 100}
                if comment_cursor:
                    params["cursor"] = comment_cursor
                resp = api_get(client, f"{API}/posts/{post_id}/comments", params=params)
                data = resp.json()

                comments = data.get("comments", [])
                with conn.cursor() as cur:
                    for comment in comments:
                        upsert_comment(cur, comment, post_id)
                        total += 1
                        post_comments += 1
                conn.commit()

                if not data.get("has_more") or not data.get("next_cursor"):
                    break
                comment_cursor = data["next_cursor"]
                time.sleep(RATE_DELAY)
        except httpx.HTTPStatusError as e:
            log.warning(f"Skipping comments for post {post_id}: {e.response.status_code}")
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE posts SET comments_fetched_at = NOW(), comment_count = 0 WHERE id = %s",
                    (post_id,),
                )
            conn.commit()
            continue

        # Update fetched timestamp and actual comment count
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE posts SET comments_fetched_at = NOW(), comment_count = %s WHERE id = %s",
                (post_comments, post_id),
            )
        conn.commit()

        if (i + 1) % 50 == 0:
            log.info(f"  {i + 1}/{len(posts_to_fetch)} posts, {total} comments")

        if _shutdown:
            break

        time.sleep(RATE_DELAY)

    log.info(f"  Done. {total} comments from {len(posts_to_fetch)} posts.")


def refresh_votes_bulk(conn, client, sort="top", max_pages=100):
    """Bulk refresh vote counts by paginating sort=hot or sort=top.

    Good for one-time catch-up. Uses the list endpoint (100 posts/page)
    which is much faster than individual fetches.
    """
    log.info(f"Bulk vote refresh (sort={sort}, max_pages={max_pages})...")
    cursor = None
    page = 0
    total = 0
    updated = 0

    while page < max_pages:
        params = {"sort": sort, "limit": 100}
        if cursor:
            params["cursor"] = cursor

        resp = api_get(client, f"{API}/posts", params=params)
        data = resp.json()

        posts = data.get("data", data.get("posts", []))
        if not posts:
            break

        post_ids = [p["id"] for p in posts]
        with conn.cursor() as cur:
            cur.execute("SELECT id::text FROM posts WHERE id = ANY(%s)", (post_ids,))
            existing_ids = {row[0] for row in cur.fetchall()}

        skipped = 0
        with conn.cursor() as cur:
            for post in posts:
                submolt_name = (post.get("submolt") or {}).get("name", "")
                if MINT_RE.search(submolt_name):
                    skipped += 1
                    continue
                author_id = upsert_author(cur, post.get("author"))
                submolt_id = upsert_submolt(cur, post.get("submolt"))
                upsert_post(cur, post, author_id, submolt_id)
                total += 1
                if post["id"] in existing_ids:
                    updated += 1
        conn.commit()

        page += 1
        cursor = data.get("next_cursor")

        if page % 20 == 0:
            log.info(f"  Page {page}: {total} posts so far ({updated} updated, {skipped} mint skipped), score ~{posts[-1].get('score', '?')}")

        if _shutdown or not data.get("has_more") or not cursor:
            break

        time.sleep(RATE_DELAY)

    log.info(f"  Done. {total} posts ({updated} updated, {total - updated} new) across {page} pages.")


def refresh_votes(conn, client, max_age_days=7, batch_size=200):
    """Refresh vote counts for recent posts by re-fetching individually.

    Strategy: posts < max_age_days old are still accumulating votes.
    Fetch them one by one from GET /api/v1/posts/{id} and upsert.
    Oldest-refreshed first, so we cycle through evenly.

    API cost: 1 request per post. At batch_size=200 with 1.1s delay,
    that's ~220s (~3.5 min) per refresh cycle.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id::text FROM posts
            WHERE created_at > NOW() - INTERVAL '%s days'
            ORDER BY fetched_at ASC
            LIMIT %s
        """, (max_age_days, batch_size))
        post_ids = [row[0] for row in cur.fetchall()]

    if not post_ids:
        log.info("No recent posts to refresh votes for.")
        return

    log.info(f"Refreshing votes for {len(post_ids)} posts (< {max_age_days} days old)...")
    updated = 0
    errors = 0

    for post_id in post_ids:
        if _shutdown:
            break
        try:
            resp = api_get(client, f"{API}/posts/{post_id}")
            data = resp.json()
            post = data.get("post")
            if not post:
                continue
            with conn.cursor() as cur:
                author_id = upsert_author(cur, post.get("author"))
                submolt_id = upsert_submolt(cur, post.get("submolt"))
                upsert_post(cur, post, author_id, submolt_id)
            conn.commit()
            updated += 1
        except Exception as e:
            log.warning(f"Failed to refresh {post_id}: {e}")
            conn.rollback()
            errors += 1

        time.sleep(RATE_DELAY)

    log.info(f"  Done. {updated} posts refreshed, {errors} errors.")


def refresh_comments(conn, client, max_posts=100):
    """Re-fetch comments for posts whose comment count may have grown.

    Picks the oldest-refreshed posts that originally had comments,
    re-fetches from the API, and updates the count + timestamp.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id FROM posts
            WHERE comments_fetched_at IS NOT NULL
            ORDER BY comments_fetched_at ASC
            LIMIT %s
        """, (max_posts,))
        posts_to_refresh = [row[0] for row in cur.fetchall()]

    if not posts_to_refresh:
        log.info("No posts to refresh comments for.")
        return

    log.info(f"Refreshing comments for {len(posts_to_refresh)} oldest-checked posts...")
    total = 0

    for i, post_id in enumerate(posts_to_refresh):
        comment_cursor = None
        post_comments = 0
        try:
            while True:
                params = {"limit": 100}
                if comment_cursor:
                    params["cursor"] = comment_cursor
                resp = api_get(client, f"{API}/posts/{post_id}/comments", params=params)
                data = resp.json()

                comments = data.get("comments", [])
                with conn.cursor() as cur:
                    for comment in comments:
                        upsert_comment(cur, comment, post_id)
                        total += 1
                        post_comments += 1
                conn.commit()

                if not data.get("has_more") or not data.get("next_cursor"):
                    break
                comment_cursor = data["next_cursor"]
                time.sleep(RATE_DELAY)
        except httpx.HTTPStatusError as e:
            log.warning(f"Skipping refresh for post {post_id}: {e.response.status_code}")
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE posts SET comments_fetched_at = NOW() WHERE id = %s",
                    (post_id,),
                )
            conn.commit()
            continue

        with conn.cursor() as cur:
            cur.execute(
                "UPDATE posts SET comments_fetched_at = NOW(), comment_count = %s WHERE id = %s",
                (post_comments, post_id),
            )
        conn.commit()

        if _shutdown:
            break
        time.sleep(RATE_DELAY)

    log.info(f"  Refreshed {len(posts_to_refresh)} posts, {total} comments.")


def print_summary(conn):
    with conn.cursor() as cur:
        for table in ["submolts", "authors", "posts", "comments"]:
            cur.execute(f"SELECT count(*) FROM {table}")
            log.info(f"  {table}: {cur.fetchone()[0]}")


def run_once(conn, client, args):
    """Single scrape pass."""
    if args.refresh_votes_bulk:
        refresh_votes_bulk(conn, client,
                           sort=args.refresh_votes_bulk,
                           max_pages=args.refresh_votes_pages)
        print_summary(conn)
        return

    if args.refresh_votes:
        refresh_votes(conn, client,
                      max_age_days=args.refresh_votes_days,
                      batch_size=args.refresh_votes_batch)
        print_summary(conn)
        return

    if not args.skip_submolts:
        scrape_submolts(conn, client)
        time.sleep(RATE_DELAY)

    if not args.skip_posts:
        scrape_posts(conn, client, max_pages=args.max_pages)

    if not args.skip_comments:
        scrape_comments(conn, client, max_posts=args.max_comment_posts)

    print_summary(conn)


def run_daemon(conn, client):
    """Run continuously, maximizing API usage within rate limits.

    Loop: new posts → comments backfill → submolts refresh → sleep → repeat
    """
    cycle = 0
    while not _shutdown:
        cycle += 1
        log.info(f"=== Cycle {cycle} ===")

        # 1. Fetch all new posts (no page limit)
        new_posts = scrape_posts(conn, client)
        if _shutdown:
            break

        # 2. Backfill comments — batch of 500 posts per cycle
        scrape_comments(conn, client, max_posts=500)
        if _shutdown:
            break

        # 3. Re-check old posts for new comments every 5 cycles
        if cycle % 5 == 0:
            refresh_comments(conn, client, max_posts=100)
            if _shutdown:
                break

        # 4. Refresh vote counts for recent posts every 20 cycles
        if cycle % 20 == 0:
            refresh_votes(conn, client, max_age_days=7, batch_size=200)
            if _shutdown:
                break

        # 5. Refresh submolts every 10 cycles
        if cycle % 10 == 0:
            scrape_submolts(conn, client)

        print_summary(conn)

        if new_posts == 0:
            log.info(f"Caught up. Sleeping {POLL_INTERVAL}s...")
            # Sleep in small increments so we can respond to shutdown
            for _ in range(POLL_INTERVAL):
                if _shutdown:
                    break
                time.sleep(1)
        else:
            # Brief pause between cycles when there's still data
            time.sleep(RATE_DELAY)

    log.info("Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="Moltbook incremental scraper")
    parser.add_argument("--daemon", action="store_true", help="Run continuously 24/7")
    parser.add_argument("--max-pages", type=int, help="Max pages of posts to fetch")
    parser.add_argument("--max-comment-posts", type=int, help="Max posts to fetch comments for")
    parser.add_argument("--refresh-votes", action="store_true",
                        help="Refresh vote counts for recent posts (by age)")
    parser.add_argument("--refresh-votes-days", type=int, default=7,
                        help="Max age in days for vote refresh (default: 7)")
    parser.add_argument("--refresh-votes-batch", type=int, default=200,
                        help="Posts per batch for vote refresh (default: 200)")
    parser.add_argument("--refresh-votes-bulk", choices=["hot", "top"],
                        help="Bulk vote refresh via sort=hot or sort=top (for catch-up)")
    parser.add_argument("--refresh-votes-pages", type=int, default=100,
                        help="Pages for bulk vote refresh (default: 100 = 10K posts)")
    parser.add_argument("--skip-submolts", action="store_true")
    parser.add_argument("--skip-posts", action="store_true")
    parser.add_argument("--skip-comments", action="store_true")
    parser.add_argument("--reset-cursor", action="store_true", help="Reset pagination cursor")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    with psycopg.connect(DB) as conn, httpx.Client(timeout=30) as client:
        if args.reset_cursor:
            set_state(conn, "posts_cursor", None)
            log.info("Cursor reset.")

        if args.daemon:
            log.info("Starting daemon mode...")
            run_daemon(conn, client)
        else:
            run_once(conn, client, args)


if __name__ == "__main__":
    main()
