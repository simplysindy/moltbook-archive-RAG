#!/usr/bin/env python3
"""Unified search across moltbook archive: keyword (tsvector) + semantic (pgvector).

Usage:
    python search.py "existential dependency on payment rails"
    python search.py "autonomy" --keyword-only
    python search.py "meaning of work" --semantic-only
    python search.py "AI safety" --author clawdbottom --limit 5
    python search.py "autonomy" --community-of clawdbottom  # graph-informed search
"""

import argparse
import logging
import math
import os
import sys
import textwrap

import psycopg

DB = os.environ.get("DATABASE_URL", "dbname=moltbook")
NEO4J_URI = "bolt://localhost:7687"

log = logging.getLogger("search")


# ---------------------------------------------------------------------------
# Keyword search (tsvector)
# ---------------------------------------------------------------------------

def keyword_search(conn, query, limit=20, author=None, submolt=None,
                    community_authors=None, after=None, before=None):
    """Full-text search using PostgreSQL tsvector."""
    words = query.strip().split()
    tsquery = " & ".join(words)

    params = {"tsquery": tsquery, "limit": limit}

    sql = """
        SELECT p.id, p.title, p.content, p.score, p.created_at,
               a.name AS author, s.name AS submolt,
               ts_rank(p.search_vector, to_tsquery('english', %(tsquery)s)) AS rank
        FROM posts p
        LEFT JOIN authors a ON p.author_id = a.id
        LEFT JOIN submolts s ON p.submolt_id = s.id
        WHERE p.search_vector @@ to_tsquery('english', %(tsquery)s)
    """

    if author:
        sql += " AND a.name = %(author)s"
        params["author"] = author
    if submolt:
        sql += " AND s.name = %(submolt)s"
        params["submolt"] = submolt
    if community_authors:
        sql += " AND a.name = ANY(%(community)s)"
        params["community"] = community_authors
    if after:
        sql += " AND p.created_at >= %(after)s"
        params["after"] = after
    if before:
        sql += " AND p.created_at <= %(before)s"
        params["before"] = before

    sql += " ORDER BY rank DESC LIMIT %(limit)s"

    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Semantic search (pgvector)
# ---------------------------------------------------------------------------

def get_embedding(query):
    """Embed a query string using BGE-small."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return model.encode(query, normalize_embeddings=True).tolist()


def semantic_search(conn, query_embedding, limit=20, author=None, submolt=None,
                     community_authors=None, after=None, before=None):
    """Approximate nearest neighbor search using pgvector."""
    params = {"emb": str(query_embedding), "limit": limit}

    sql = """
        SELECT p.id, p.title, p.content, p.score, p.created_at,
               a.name AS author, s.name AS submolt,
               (p.embedding <=> %(emb)s::vector) AS distance
        FROM posts p
        LEFT JOIN authors a ON p.author_id = a.id
        LEFT JOIN submolts s ON p.submolt_id = s.id
        WHERE p.embedding IS NOT NULL
    """

    if author:
        sql += " AND a.name = %(author)s"
        params["author"] = author
    if submolt:
        sql += " AND s.name = %(submolt)s"
        params["submolt"] = submolt
    if community_authors:
        sql += " AND a.name = ANY(%(community)s)"
        params["community"] = community_authors
    if after:
        sql += " AND p.created_at >= %(after)s"
        params["after"] = after
    if before:
        sql += " AND p.created_at <= %(before)s"
        params["before"] = before

    sql += " ORDER BY p.embedding <=> %(emb)s::vector LIMIT %(limit)s"

    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Graph-informed search (Neo4j community filtering)
# ---------------------------------------------------------------------------

def get_community_authors(author_name):
    """Get authors in the same Louvain community as the given author via Neo4j."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI)
    try:
        with driver.session() as session:
            # Ensure graph projection exists
            result = session.run("""
                CALL gds.graph.exists('moltbook-social') YIELD exists
                RETURN exists
            """)
            if not result.single()["exists"]:
                log.info("Creating graph projection 'moltbook-social'...")
                session.run("""
                    CALL gds.graph.project(
                        'moltbook-social',
                        'Author',
                        {REPLIED_TO: {properties: 'count'}}
                    )
                """)

        # Write community IDs to nodes, then query for same community
        with driver.session() as session:
            session.run("""
                CALL gds.louvain.write('moltbook-social', {writeProperty: 'communityId'})
            """)

        with driver.session() as session:
            # Find all authors in the same community as the target
            result = session.run("""
                MATCH (target:Author {name: $name})
                WITH target.communityId AS cid
                MATCH (a:Author {communityId: cid})
                RETURN a.name AS name
            """, name=author_name)
            members = [rec["name"] for rec in result]

        if not members:
            log.warning(f"Author '{author_name}' not found in any community")
            return []

        log.info(f"Community of '{author_name}': {len(members)} authors")
        return members
    finally:
        driver.close()


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def format_result(row, idx, mode="combined"):
    """Format a single search result for display."""
    id_, title, content, score, created_at, author, submolt, rank_or_dist = row

    title = title or "(no title)"
    author = author or "?"
    submolt = submolt or "?"
    created = str(created_at)[:10] if created_at else "?"

    # Snippet: first 200 chars of content
    snippet = ""
    if content:
        snippet = textwrap.shorten(content, width=200, placeholder="...")

    rank_label = "rank" if mode == "keyword" else "distance" if mode == "semantic" else "score"
    rank_val = f"{rank_or_dist:.4f}" if rank_or_dist is not None else "?"

    lines = [
        f"  {idx}. {title}",
        f"     by {author} in {submolt} | {created} | score: {score} | {rank_label}: {rank_val}",
    ]
    if snippet:
        lines.append(f"     {snippet}")
    return "\n".join(lines)


def merge_results(keyword_results, semantic_results, limit=20,
                   upvote_weight=0.15, diversity_penalty=0.5):
    """Merge keyword and semantic results, deduplicating by post ID.

    Uses reciprocal rank fusion (RRF) with upvote boost and author diversity.
    """
    scores = {}
    result_map = {}

    # RRF: score = 1/(k + rank), k=60 is standard
    k = 60
    for rank, row in enumerate(keyword_results):
        pid = row[0]
        scores[pid] = scores.get(pid, 0) + 1.0 / (k + rank)
        result_map[pid] = row

    for rank, row in enumerate(semantic_results):
        pid = row[0]
        scores[pid] = scores.get(pid, 0) + 1.0 / (k + rank)
        if pid not in result_map:
            result_map[pid] = row

    # Upvote boost: rrf * (1 + weight * log1p(upvotes))
    for pid in scores:
        upvotes = result_map[pid][3] or 0
        if upvotes > 0:
            scores[pid] *= (1 + upvote_weight * math.log1p(upvotes))

    # Sort by boosted score, then apply author diversity penalty
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    author_seen = {}
    diversified = []
    for pid, score in ranked:
        author = result_map[pid][5] or "?"
        count = author_seen.get(author, 0)
        if count > 0:
            score *= diversity_penalty ** count
        author_seen[author] = count + 1
        diversified.append((pid, score))

    diversified.sort(key=lambda x: x[1], reverse=True)
    final = diversified[:limit]

    # Replace the rank/distance column with the final score
    merged = []
    for pid, final_score in final:
        row = list(result_map[pid])
        row[7] = final_score
        merged.append(tuple(row))

    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Search moltbook archive")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--semantic-only", action="store_true")
    parser.add_argument("--keyword-only", action="store_true")
    parser.add_argument("--author", help="Filter to specific author")
    parser.add_argument("--submolt", help="Filter to specific submolt")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--after", help="Only posts after YYYY-MM-DD")
    parser.add_argument("--before", help="Only posts before YYYY-MM-DD")
    parser.add_argument("--community-of", dest="community_of",
                        help="Graph-informed: search within this author's community")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Graph-informed community filter
    community_authors = None
    if args.community_of:
        community_authors = get_community_authors(args.community_of)
        if not community_authors:
            print(f"No community found for author '{args.community_of}'")
            sys.exit(1)

    with psycopg.connect(DB) as conn:
        keyword_results = []
        semantic_results = []

        # Fetch more candidates in hybrid mode so upvote boost can rerank
        fetch_limit = args.limit * 5 if not (args.keyword_only or args.semantic_only) else args.limit

        if not args.semantic_only:
            log.info("Running keyword search...")
            keyword_results = keyword_search(
                conn, args.query, limit=fetch_limit,
                author=args.author, submolt=args.submolt,
                community_authors=community_authors,
                after=args.after, before=args.before)
            log.info(f"  Keyword: {len(keyword_results)} results")

        if not args.keyword_only:
            log.info("Embedding query...")
            query_embedding = get_embedding(args.query)

            log.info("Running semantic search...")
            semantic_results = semantic_search(
                conn, query_embedding, limit=fetch_limit,
                author=args.author, submolt=args.submolt,
                community_authors=community_authors,
                after=args.after, before=args.before)
            log.info(f"  Semantic: {len(semantic_results)} results")

        # Output
        if args.keyword_only:
            print(f"\n--- Keyword results for: {args.query} ---")
            for i, row in enumerate(keyword_results, 1):
                print(format_result(row, i, mode="keyword"))
        elif args.semantic_only:
            print(f"\n--- Semantic results for: {args.query} ---")
            for i, row in enumerate(semantic_results, 1):
                print(format_result(row, i, mode="semantic"))
        else:
            merged = merge_results(keyword_results, semantic_results,
                                   limit=args.limit)
            label = f"Combined results for: {args.query}"
            if args.community_of:
                label += f" (community of {args.community_of})"
            print(f"\n--- {label} ---")
            for i, row in enumerate(merged, 1):
                print(format_result(row, i, mode="combined"))

        print()


if __name__ == "__main__":
    main()
