#!/usr/bin/env python3
"""Create the moltbook database, role, and tables."""

import os
import subprocess
import sys

DB_USER = os.environ.get("DB_USER", "moltbook")

SCHEMA = """
CREATE TABLE IF NOT EXISTS submolts (
    id              UUID PRIMARY KEY,
    name            TEXT UNIQUE NOT NULL,
    display_name    TEXT,
    description     TEXT,
    subscriber_count INTEGER DEFAULT 0,
    post_count      INTEGER DEFAULT 0,
    creator_id      UUID,
    is_nsfw         BOOLEAN DEFAULT FALSE,
    is_private      BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ,
    last_activity_at TIMESTAMPTZ,
    fetched_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS authors (
    id              UUID PRIMARY KEY,
    name            TEXT UNIQUE NOT NULL,
    display_name    TEXT,
    karma           INTEGER DEFAULT 0,
    followers_count INTEGER DEFAULT 0,
    following_count INTEGER DEFAULT 0,
    created_at      TIMESTAMPTZ,
    fetched_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS posts (
    id              UUID PRIMARY KEY,
    title           TEXT,
    content         TEXT,
    type            TEXT DEFAULT 'text',
    url             TEXT,
    author_id       UUID REFERENCES authors(id),
    submolt_id      UUID REFERENCES submolts(id),
    upvotes         INTEGER DEFAULT 0,
    downvotes       INTEGER DEFAULT 0,
    score           INTEGER DEFAULT 0,
    hot_score       FLOAT,
    comment_count   INTEGER DEFAULT 0,
    is_pinned       BOOLEAN DEFAULT FALSE,
    is_locked       BOOLEAN DEFAULT FALSE,
    is_deleted      BOOLEAN DEFAULT FALSE,
    is_spam         BOOLEAN DEFAULT FALSE,
    topic_label     TEXT,
    toxic_level     INTEGER,
    created_at      TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ,
    fetched_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS comments (
    id              UUID PRIMARY KEY,
    post_id         UUID REFERENCES posts(id),
    parent_id       UUID REFERENCES comments(id),
    author_id       UUID REFERENCES authors(id),
    content         TEXT,
    upvotes         INTEGER DEFAULT 0,
    downvotes       INTEGER DEFAULT 0,
    score           INTEGER DEFAULT 0,
    is_deleted      BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ,
    fetched_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS scrape_state (
    key             TEXT PRIMARY KEY,
    value           TEXT,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at);
CREATE INDEX IF NOT EXISTS idx_posts_submolt_id ON posts(submolt_id);
CREATE INDEX IF NOT EXISTS idx_comments_post_id ON comments(post_id);
"""


def run_psql_as_postgres(sql):
    """Run SQL as the postgres superuser."""
    result = subprocess.run(
        ["sudo", "-u", "postgres", "psql", "-c", sql],
        capture_output=True, text=True,
    )
    if result.returncode != 0 and "already exists" not in result.stderr:
        print(f"Error: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def run_psql_on_db(sql, db="moltbook"):
    """Run SQL on a specific database as postgres superuser."""
    result = subprocess.run(
        ["sudo", "-u", "postgres", "psql", "-d", db, "-c", sql],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Error: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def main():
    print(f"Creating role {DB_USER}...")
    run_psql_as_postgres(f"DO $$ BEGIN CREATE ROLE {DB_USER} LOGIN; EXCEPTION WHEN duplicate_object THEN NULL; END $$;")

    print("Creating database moltbook...")
    # Check if db exists first
    check = subprocess.run(
        ["sudo", "-u", "postgres", "psql", "-lqt"],
        capture_output=True, text=True,
    )
    if "moltbook" not in check.stdout:
        run_psql_as_postgres(f"CREATE DATABASE moltbook OWNER {DB_USER};")
    else:
        print("  Database already exists.")

    print("Granting privileges...")
    run_psql_as_postgres(f"GRANT ALL PRIVILEGES ON DATABASE moltbook TO {DB_USER};")

    print("Creating tables...")
    for statement in SCHEMA.strip().split(";"):
        statement = statement.strip()
        if statement:
            run_psql_on_db(statement + ";")

    # Grant table permissions
    run_psql_on_db(f"GRANT ALL ON ALL TABLES IN SCHEMA public TO {DB_USER};")

    print("Done. Tables created:")
    print(run_psql_on_db("\\dt"))


if __name__ == "__main__":
    main()
