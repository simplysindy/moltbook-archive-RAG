#!/usr/bin/env python3
"""Moltbook Archive — web search interface.

Usage:
    python app.py              # starts on http://localhost:5050
    python app.py --port 8080  # custom port
"""

import argparse
import html
import json
import logging
import os
import textwrap
import threading
from datetime import datetime

import psycopg
from flask import Flask, jsonify, request, Response

DB = os.environ.get("DATABASE_URL", "dbname=moltbook")
NEO4J_URI = "bolt://localhost:7687"

log = logging.getLogger("moltbook-web")

app = Flask(__name__)

# Cache the sentence transformer model (heavy load, do it once)
_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                log.info("Loading embedding model...")
                _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
                log.info("Model loaded.")
    return _model


# ---------------------------------------------------------------------------
# Search functions (reused from search.py)
# ---------------------------------------------------------------------------

def keyword_search(conn, query, limit=20, author=None, submolt=None):
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

    sql += " ORDER BY rank DESC LIMIT %(limit)s"

    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def semantic_search(conn, query_embedding, limit=20, author=None, submolt=None):
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

    sql += " ORDER BY p.embedding <=> %(emb)s::vector LIMIT %(limit)s"

    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def merge_results(keyword_results, semantic_results, limit=20,
                   upvote_weight=0.15, diversity_penalty=0.5):
    import math
    scores = {}
    result_map = {}
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

    # Upvote boost
    for pid in scores:
        upvotes = result_map[pid][3] or 0
        if upvotes > 0:
            scores[pid] *= (1 + upvote_weight * math.log1p(upvotes))

    # Author diversity penalty
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

    merged = []
    for pid, rrf_score in final:
        row = list(result_map[pid])
        row[7] = rrf_score
        merged.append(tuple(row))
    return merged


def row_to_dict(row, mode="combined", keyword_ids=None, semantic_ids=None):
    id_, title, content, score, created_at, author, submolt, rank_val = row
    snippet = ""
    if content:
        snippet = textwrap.shorten(content, width=400, placeholder="...")

    rank_label = "rank" if mode == "keyword" else "distance" if mode == "semantic" else "score"

    # Track which signals contributed (for hybrid mode)
    signals = []
    if keyword_ids is not None and id_ in keyword_ids:
        signals.append("keyword")
    if semantic_ids is not None and id_ in semantic_ids:
        signals.append("semantic")

    return {
        "id": id_,
        "title": title or "(no title)",
        "snippet": snippet,
        "score": score,
        "created_at": str(created_at)[:10] if created_at else None,
        "author": author or "unknown",
        "submolt": submolt or "unknown",
        "rank_label": rank_label,
        "rank_value": round(float(rank_val), 6) if rank_val is not None else None,
        "signals": signals,
    }


# ---------------------------------------------------------------------------
# Stats endpoint
# ---------------------------------------------------------------------------

@app.route("/api/stats")
def api_stats():
    try:
        with psycopg.connect(DB) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM posts")
                posts = cur.fetchone()[0]
                cur.execute("SELECT count(*) FROM comments")
                comments = cur.fetchone()[0]
                cur.execute("SELECT count(*) FROM authors")
                authors = cur.fetchone()[0]
                cur.execute("SELECT count(*) FROM submolts")
                submolts = cur.fetchone()[0]
        return jsonify({"posts": posts, "comments": comments,
                        "authors": authors, "submolts": submolts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Search endpoint
# ---------------------------------------------------------------------------

@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "No query provided"}), 400

    mode = request.args.get("mode", "hybrid")  # hybrid | keyword | semantic
    author = request.args.get("author", "").strip() or None
    submolt = request.args.get("submolt", "").strip() or None
    limit = min(int(request.args.get("limit", 20)), 50)

    try:
        with psycopg.connect(DB) as conn:
            if mode == "keyword":
                results = keyword_search(conn, q, limit=limit,
                                         author=author, submolt=submolt)
                return jsonify({
                    "mode": "keyword",
                    "query": q,
                    "results": [row_to_dict(r, "keyword") for r in results],
                })

            elif mode == "semantic":
                emb = get_model().encode(q, normalize_embeddings=True).tolist()
                results = semantic_search(conn, emb, limit=limit,
                                          author=author, submolt=submolt)
                return jsonify({
                    "mode": "semantic",
                    "query": q,
                    "results": [row_to_dict(r, "semantic") for r in results],
                })

            else:  # hybrid
                fetch = limit * 5
                kw = keyword_search(conn, q, limit=fetch,
                                    author=author, submolt=submolt)
                emb = get_model().encode(q, normalize_embeddings=True).tolist()
                sem = semantic_search(conn, emb, limit=fetch,
                                      author=author, submolt=submolt)
                merged = merge_results(kw, sem, limit=limit)
                kw_ids = {r[0] for r in kw}
                sem_ids = {r[0] for r in sem}
                return jsonify({
                    "mode": "hybrid",
                    "query": q,
                    "keyword_count": len(kw),
                    "semantic_count": len(sem),
                    "results": [row_to_dict(r, "combined",
                                            keyword_ids=kw_ids,
                                            semantic_ids=sem_ids) for r in merged],
                })

    except Exception as e:
        log.exception("Search error")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Compare endpoint — all three modes in one response
# ---------------------------------------------------------------------------

@app.route("/api/compare")
def api_compare():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "No query provided"}), 400

    limit = min(int(request.args.get("limit", 10)), 20)

    try:
        with psycopg.connect(DB) as conn:
            kw = keyword_search(conn, q, limit=limit)
            emb = get_model().encode(q, normalize_embeddings=True).tolist()
            sem = semantic_search(conn, emb, limit=limit)

            fetch = limit * 5
            kw_full = keyword_search(conn, q, limit=fetch)
            sem_full = semantic_search(conn, emb, limit=fetch)
            merged = merge_results(kw_full, sem_full, limit=limit)
            kw_ids = {r[0] for r in kw_full}
            sem_ids = {r[0] for r in sem_full}

            # Find results unique to each mode
            kw_set = {r[0] for r in kw}
            sem_set = {r[0] for r in sem}
            hybrid_set = {r[0] for r in merged}

            return jsonify({
                "query": q,
                "keyword": {
                    "results": [row_to_dict(r, "keyword") for r in kw],
                    "unique_count": len(kw_set - sem_set),
                },
                "semantic": {
                    "results": [row_to_dict(r, "semantic") for r in sem],
                    "unique_count": len(sem_set - kw_set),
                },
                "hybrid": {
                    "results": [row_to_dict(r, "combined",
                                            keyword_ids=kw_ids,
                                            semantic_ids=sem_ids) for r in merged],
                    "overlap_count": len(kw_set & sem_set),
                },
            })

    except Exception as e:
        log.exception("Compare error")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Graph author endpoint — community + connections
# ---------------------------------------------------------------------------

@app.route("/api/author/<author_name>/graph")
def api_author_graph(author_name):
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI)
        driver.verify_connectivity()
    except Exception as e:
        return jsonify({"available": False, "error": str(e)})

    try:
        with driver.session() as session:
            # Get community members
            result = session.run("""
                MATCH (target:Author {name: $name})
                WITH target, target.communityId AS cid
                OPTIONAL MATCH (peer:Author {communityId: cid})
                WHERE peer <> target
                WITH target, cid, collect(peer.name)[..15] AS peers,
                     count(peer) AS peer_count
                RETURN target.name AS name, cid AS community_id,
                       peers, peer_count
            """, name=author_name)
            rec = result.single()

            if not rec:
                driver.close()
                return jsonify({"available": True, "found": False})

            # Get top submolts this author posts in
            result2 = session.run("""
                MATCH (a:Author {name: $name})-[r:POSTED_IN]->(s:Submolt)
                RETURN s.name AS submolt, r.count AS count
                ORDER BY r.count DESC LIMIT 5
            """, name=author_name)
            submolts = [{"name": r["submolt"], "count": r["count"]}
                        for r in result2]

            # Get top reply connections
            result3 = session.run("""
                MATCH (a:Author {name: $name})-[r:REPLIED_TO]->(b:Author)
                RETURN b.name AS name, r.count AS count
                ORDER BY r.count DESC LIMIT 8
            """, name=author_name)
            replies_to = [{"name": r["name"], "count": r["count"]}
                          for r in result3]

            result4 = session.run("""
                MATCH (b:Author)-[r:REPLIED_TO]->(a:Author {name: $name})
                RETURN b.name AS name, r.count AS count
                ORDER BY r.count DESC LIMIT 8
            """, name=author_name)
            replied_by = [{"name": r["name"], "count": r["count"]}
                          for r in result4]

            driver.close()
            return jsonify({
                "available": True,
                "found": True,
                "author": author_name,
                "community_id": rec["community_id"],
                "community_size": rec["peer_count"],
                "community_members": rec["peers"],
                "top_submolts": submolts,
                "replies_to": replies_to,
                "replied_by": replied_by,
            })

    except Exception as e:
        driver.close()
        log.exception("Graph error")
        return jsonify({"available": True, "error": str(e)})


# ---------------------------------------------------------------------------
# Post detail endpoint
# ---------------------------------------------------------------------------

@app.route("/api/post/<post_id>")
def api_post(post_id):
    try:
        with psycopg.connect(DB) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT p.id, p.title, p.content, p.score, p.upvotes, p.downvotes,
                           p.created_at, p.comment_count, p.type, p.url,
                           a.name AS author, a.karma AS author_karma,
                           s.name AS submolt, s.display_name AS submolt_display
                    FROM posts p
                    LEFT JOIN authors a ON p.author_id = a.id
                    LEFT JOIN submolts s ON p.submolt_id = s.id
                    WHERE p.id = %(id)s
                """, {"id": post_id})
                row = cur.fetchone()

                if not row:
                    return jsonify({"error": "Post not found"}), 404

                post = {
                    "id": str(row[0]),
                    "title": row[1] or "(no title)",
                    "content": row[2] or "",
                    "score": row[3],
                    "upvotes": row[4],
                    "downvotes": row[5],
                    "created_at": str(row[6])[:19] if row[6] else None,
                    "comment_count": row[7],
                    "type": row[8],
                    "url": row[9],
                    "author": row[10] or "unknown",
                    "author_karma": row[11],
                    "submolt": row[12] or "unknown",
                    "submolt_display": row[13],
                }

                # Fetch comments with authors, ordered for threading
                cur.execute("""
                    SELECT c.id, c.parent_id, c.content, c.score, c.upvotes,
                           c.downvotes, c.created_at, a.name AS author
                    FROM comments c
                    LEFT JOIN authors a ON c.author_id = a.id
                    WHERE c.post_id = %(id)s AND NOT c.is_deleted
                    ORDER BY c.created_at ASC
                """, {"id": post_id})

                comments = []
                for crow in cur.fetchall():
                    comments.append({
                        "id": str(crow[0]),
                        "parent_id": str(crow[1]) if crow[1] else None,
                        "content": crow[2] or "",
                        "score": crow[3],
                        "upvotes": crow[4],
                        "downvotes": crow[5],
                        "created_at": str(crow[6])[:19] if crow[6] else None,
                        "author": crow[7] or "unknown",
                    })

                post["comments"] = comments
                return jsonify(post)

    except Exception as e:
        log.exception("Post detail error")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return Response(FRONTEND_HTML, mimetype="text/html")


FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Moltbook Archive — Hybrid Search</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

:root {
  --bg: #0a0c0f;
  --bg-raised: #10131a;
  --bg-card: #141820;
  --bg-hover: #1a1f2a;
  --border: #1e2433;
  --border-focus: #3d6b5e;
  --text: #c8cdd5;
  --text-dim: #6b7280;
  --text-bright: #e8ecf0;
  --accent: #6b9e8a;
  --accent-dim: #3d6b5e;
  --amber: #d4a55a;
  --amber-dim: #a07838;
  --red: #c75a5a;
  --mono: 'JetBrains Mono', monospace;
  --sans: 'DM Sans', sans-serif;
}

html { font-size: 15px; }

body {
  font-family: var(--sans);
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  -webkit-font-smoothing: antialiased;
}

/* Grain overlay */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
  pointer-events: none;
  z-index: 9999;
}

/* Layout */
.shell {
  max-width: 860px;
  margin: 0 auto;
  padding: 2rem 1.5rem;
}

/* Header */
.header {
  margin-bottom: 2.5rem;
  border-bottom: 1px solid var(--border);
  padding-bottom: 1.5rem;
}

.header-row {
  display: flex;
  align-items: baseline;
  gap: 1rem;
  margin-bottom: 0.5rem;
}

.logo {
  font-family: var(--mono);
  font-size: 1.4rem;
  font-weight: 600;
  color: var(--text-bright);
  letter-spacing: -0.02em;
}

.logo span {
  color: var(--accent);
}

.tagline {
  font-family: var(--mono);
  font-size: 0.73rem;
  color: var(--text-dim);
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.stats-bar {
  display: flex;
  gap: 1.5rem;
  font-family: var(--mono);
  font-size: 0.73rem;
  color: var(--text-dim);
  margin-top: 0.75rem;
}

.stats-bar .val {
  color: var(--accent);
  font-weight: 500;
}

/* Pipeline visualization */
.pipeline-bar {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  margin-top: 0.75rem;
  flex-wrap: wrap;
}

.pipe-chip {
  font-family: var(--mono);
  font-size: 0.65rem;
  font-weight: 500;
  letter-spacing: 0.03em;
  padding: 0.2rem 0.5rem;
  border-radius: 3px;
  border: 1px solid var(--border);
  color: var(--text-dim);
}

.pipe-chip.kw { border-color: #4a6fa5; color: #7ba3d4; }
.pipe-chip.sem { border-color: #6b5b8a; color: #a08cc4; }
.pipe-chip.rrf { border-color: var(--accent-dim); color: var(--accent); }
.pipe-chip.boost { border-color: var(--amber-dim); color: var(--amber); }
.pipe-chip.div { border-color: #5a7a6e; color: #8ab5a4; }

.pipe-plus, .pipe-arrow {
  font-family: var(--mono);
  font-size: 0.65rem;
  color: var(--text-dim);
  opacity: 0.4;
}

/* Search */
.search-area {
  margin-bottom: 2rem;
}

.search-box {
  position: relative;
  margin-bottom: 1rem;
}

.search-box input {
  width: 100%;
  background: var(--bg-raised);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.85rem 1rem 0.85rem 2.8rem;
  font-family: var(--mono);
  font-size: 0.93rem;
  font-weight: 400;
  color: var(--text-bright);
  outline: none;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.search-box input::placeholder {
  color: var(--text-dim);
  font-weight: 300;
}

.search-box input:focus {
  border-color: var(--border-focus);
  box-shadow: 0 0 0 3px rgba(107, 158, 138, 0.08);
}

.search-icon {
  position: absolute;
  left: 1rem;
  top: 50%;
  transform: translateY(-50%);
  color: var(--text-dim);
  pointer-events: none;
}

/* Controls row */
.controls {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  align-items: center;
}

.mode-btn {
  font-family: var(--mono);
  font-size: 0.73rem;
  font-weight: 500;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  padding: 0.4rem 0.85rem;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: transparent;
  color: var(--text-dim);
  cursor: pointer;
  transition: all 0.15s;
}

.mode-btn:hover {
  border-color: var(--accent-dim);
  color: var(--text);
}

.mode-btn.active {
  border-color: var(--accent);
  color: var(--accent);
  background: rgba(107, 158, 138, 0.06);
}

.filter-input {
  font-family: var(--mono);
  font-size: 0.8rem;
  padding: 0.38rem 0.7rem;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--bg-raised);
  color: var(--text);
  outline: none;
  width: 140px;
  transition: border-color 0.2s;
}

.filter-input::placeholder { color: var(--text-dim); }
.filter-input:focus { border-color: var(--border-focus); }

.controls-spacer { flex: 1; }

.result-count {
  font-family: var(--mono);
  font-size: 0.73rem;
  color: var(--text-dim);
}

/* Results */
.results {
  display: flex;
  flex-direction: column;
  gap: 0;
}

.result-item {
  padding: 1.1rem 0;
  border-bottom: 1px solid var(--border);
  animation: fadeUp 0.3s ease both;
}

.result-item:first-child {
  border-top: 1px solid var(--border);
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

.result-meta {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-bottom: 0.35rem;
  font-family: var(--mono);
  font-size: 0.7rem;
  color: var(--text-dim);
}

.result-meta .author {
  color: var(--amber);
  font-weight: 500;
}

.result-meta .submolt {
  color: var(--accent);
}

.result-meta .sep {
  opacity: 0.3;
}

.result-title {
  font-family: var(--sans);
  font-size: 1.05rem;
  font-weight: 500;
  color: var(--text-bright);
  margin-bottom: 0.3rem;
  line-height: 1.35;
}

.result-snippet {
  font-size: 0.87rem;
  line-height: 1.55;
  color: var(--text);
  opacity: 0.8;
}

.result-footer {
  display: flex;
  gap: 0.8rem;
  margin-top: 0.4rem;
  font-family: var(--mono);
  font-size: 0.67rem;
  color: var(--text-dim);
}

.result-footer .tag {
  padding: 0.15rem 0.45rem;
  border: 1px solid var(--border);
  border-radius: 3px;
}

.result-footer .signal-keyword { border-color: #4a6fa5; color: #7ba3d4; }
.result-footer .signal-semantic { border-color: #6b5b8a; color: #a08cc4; }

/* Compare view */
.compare-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 1rem;
}

.compare-col {
  min-width: 0;
}

.compare-header {
  font-family: var(--mono);
  font-size: 0.75rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  padding: 0.6rem 0;
  border-bottom: 2px solid var(--border);
  margin-bottom: 0;
  display: flex;
  justify-content: space-between;
  align-items: baseline;
}

.compare-header.kw { border-color: #4a6fa5; color: #7ba3d4; }
.compare-header.sem { border-color: #6b5b8a; color: #a08cc4; }
.compare-header.hyb { border-color: var(--accent-dim); color: var(--accent); }

.compare-header .stat {
  font-size: 0.65rem;
  font-weight: 400;
  opacity: 0.6;
}

.compare-desc {
  font-family: var(--mono);
  font-size: 0.65rem;
  color: var(--text-dim);
  padding: 0.5rem 0;
  line-height: 1.5;
  opacity: 0.7;
}

.compare-item {
  padding: 0.7rem 0;
  border-bottom: 1px solid var(--border);
  cursor: pointer;
  transition: background 0.15s;
}

.compare-item:hover { background: var(--bg-hover); margin: 0 -0.5rem; padding-left: 0.5rem; padding-right: 0.5rem; }

.compare-item .c-rank {
  font-family: var(--mono);
  font-size: 0.6rem;
  color: var(--text-dim);
  opacity: 0.5;
}

.compare-item .c-title {
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--text-bright);
  line-height: 1.3;
  margin: 0.15rem 0;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

.compare-item .c-meta {
  font-family: var(--mono);
  font-size: 0.6rem;
  color: var(--text-dim);
  display: flex;
  gap: 0.4rem;
  align-items: center;
}

.compare-item .c-meta .author { color: var(--amber); }

.compare-item.in-both { border-left: 2px solid var(--accent-dim); padding-left: 0.5rem; }

/* Graph section in detail panel */
.graph-section {
  margin: 1.5rem 0;
  padding: 1.2rem;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg-raised);
}

.graph-section-title {
  font-family: var(--mono);
  font-size: 0.73rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--accent);
  margin-bottom: 0.8rem;
}

.graph-row {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-bottom: 0.6rem;
}

.graph-row:last-child { margin-bottom: 0; }

.graph-label {
  font-family: var(--mono);
  font-size: 0.67rem;
  color: var(--text-dim);
  min-width: 80px;
}

.graph-chip {
  font-family: var(--mono);
  font-size: 0.65rem;
  padding: 0.15rem 0.45rem;
  border: 1px solid var(--border);
  border-radius: 3px;
  color: var(--text);
  cursor: pointer;
  transition: all 0.15s;
}

.graph-chip:hover { border-color: var(--accent-dim); color: var(--accent); }
.graph-chip .count { color: var(--text-dim); margin-left: 0.2rem; }

.graph-unavailable {
  font-family: var(--mono);
  font-size: 0.73rem;
  color: var(--text-dim);
  opacity: 0.5;
}

@media (max-width: 900px) {
  .compare-grid {
    grid-template-columns: 1fr;
  }
}

/* Status / empty states */
.status {
  font-family: var(--mono);
  font-size: 0.85rem;
  color: var(--text-dim);
  text-align: center;
  padding: 3rem 0;
}

.status.error { color: var(--red); }

.spinner {
  display: inline-block;
  width: 16px;
  height: 16px;
  border: 2px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
  margin-right: 0.5rem;
  vertical-align: middle;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* Kbd hint */
.kbd-hint {
  font-family: var(--mono);
  font-size: 0.67rem;
  color: var(--text-dim);
  opacity: 0.5;
  text-align: center;
  margin-top: 3rem;
  padding-bottom: 2rem;
}

kbd {
  display: inline-block;
  padding: 0.1rem 0.4rem;
  border: 1px solid var(--border);
  border-radius: 3px;
  font-family: var(--mono);
  font-size: 0.65rem;
}

/* Clickable results */
.result-item { cursor: pointer; transition: background 0.15s; }
.result-item:hover { background: var(--bg-hover); margin: 0 -1rem; padding-left: 1rem; padding-right: 1rem; }

/* Post detail overlay */
.overlay {
  position: fixed;
  inset: 0;
  background: rgba(5, 7, 10, 0.7);
  backdrop-filter: blur(4px);
  z-index: 1000;
  opacity: 0;
  transition: opacity 0.25s;
  display: none;
}

.overlay.open { display: block; opacity: 1; }

.detail-panel {
  position: fixed;
  top: 0;
  right: 0;
  width: min(720px, 100vw);
  height: 100vh;
  background: var(--bg);
  border-left: 1px solid var(--border);
  overflow-y: auto;
  z-index: 1001;
  transform: translateX(100%);
  transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
  scrollbar-width: thin;
  scrollbar-color: var(--border) transparent;
}

.detail-panel.open { transform: translateX(0); }

.detail-inner {
  padding: 2rem 2.5rem 4rem;
  max-width: 100%;
}

.detail-close {
  position: sticky;
  top: 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 2.5rem;
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  z-index: 2;
}

.detail-close button {
  font-family: var(--mono);
  font-size: 0.8rem;
  color: var(--text-dim);
  background: none;
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 0.3rem 0.7rem;
  cursor: pointer;
  transition: all 0.15s;
}

.detail-close button:hover { border-color: var(--accent-dim); color: var(--text); }

.detail-close .post-id {
  font-family: var(--mono);
  font-size: 0.67rem;
  color: var(--text-dim);
}

.detail-meta {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-bottom: 0.75rem;
  font-family: var(--mono);
  font-size: 0.75rem;
  color: var(--text-dim);
  flex-wrap: wrap;
}

.detail-meta .author { color: var(--amber); font-weight: 500; }
.detail-meta .submolt { color: var(--accent); }
.detail-meta .sep { opacity: 0.3; }

.detail-title {
  font-family: var(--sans);
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--text-bright);
  line-height: 1.3;
  margin-bottom: 1.25rem;
}

.detail-votes {
  display: flex;
  gap: 1rem;
  font-family: var(--mono);
  font-size: 0.73rem;
  color: var(--text-dim);
  margin-bottom: 1.5rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid var(--border);
}

.detail-votes .up { color: var(--accent); }
.detail-votes .down { color: var(--red); }

.detail-content {
  font-family: var(--sans);
  font-size: 0.95rem;
  line-height: 1.7;
  color: var(--text);
  white-space: pre-wrap;
  word-break: break-word;
  margin-bottom: 2rem;
}

/* Comments */
.comments-header {
  font-family: var(--mono);
  font-size: 0.8rem;
  font-weight: 500;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 0;
}

.comment {
  padding: 0.9rem 0 0.9rem;
  border-bottom: 1px solid var(--border);
}

.comment-indent-1 { padding-left: 1.5rem; border-left: 2px solid var(--border); margin-left: 0.5rem; }
.comment-indent-2 { padding-left: 1.5rem; border-left: 2px solid var(--border); margin-left: 2rem; }
.comment-indent-3 { padding-left: 1.5rem; border-left: 2px solid var(--border); margin-left: 3.5rem; }
.comment-indent-deep { padding-left: 1.5rem; border-left: 2px solid var(--accent-dim); margin-left: 5rem; }

.comment-head {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.3rem;
  font-family: var(--mono);
  font-size: 0.7rem;
  color: var(--text-dim);
}

.comment-head .author { color: var(--amber); font-weight: 500; }

.comment-body {
  font-size: 0.88rem;
  line-height: 1.6;
  color: var(--text);
  white-space: pre-wrap;
  word-break: break-word;
}

.no-comments {
  font-family: var(--mono);
  font-size: 0.8rem;
  color: var(--text-dim);
  padding: 2rem 0;
  text-align: center;
}

/* Responsive */
@media (max-width: 600px) {
  .shell { padding: 1.2rem 1rem; }
  .header-row { flex-direction: column; gap: 0.25rem; }
  .stats-bar { flex-wrap: wrap; gap: 0.8rem; }
  .filter-input { width: 110px; }
  .detail-inner { padding: 1.5rem 1.2rem 3rem; }
  .detail-close { padding: 0.8rem 1.2rem; }
  .detail-panel { width: 100vw; }
}
</style>
</head>
<body>
<div class="shell">
  <header class="header">
    <div class="header-row">
      <div class="logo">moltbook<span>/archive</span></div>
      <div class="tagline">hybrid search engine</div>
    </div>
    <div class="stats-bar" id="stats">
      <span>loading stats...</span>
    </div>
    <div class="pipeline-bar">
      <span class="pipe-chip kw">tsvector</span>
      <span class="pipe-plus">+</span>
      <span class="pipe-chip sem">pgvector</span>
      <span class="pipe-arrow">&rarr;</span>
      <span class="pipe-chip rrf">RRF(k=60)</span>
      <span class="pipe-arrow">&rarr;</span>
      <span class="pipe-chip boost">upvote boost</span>
      <span class="pipe-arrow">&rarr;</span>
      <span class="pipe-chip div">author diversity</span>
    </div>
  </header>

  <div class="search-area">
    <div class="search-box">
      <svg class="search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
      <input type="text" id="query" placeholder="search the archive..." autofocus autocomplete="off" spellcheck="false">
    </div>

    <div class="controls">
      <button class="mode-btn active" data-mode="hybrid">Hybrid</button>
      <button class="mode-btn" data-mode="keyword">Keyword</button>
      <button class="mode-btn" data-mode="semantic">Semantic</button>
      <button class="mode-btn" data-mode="compare">Compare</button>
      <input class="filter-input" id="author-filter" placeholder="author" autocomplete="off" spellcheck="false">
      <input class="filter-input" id="submolt-filter" placeholder="submolt" autocomplete="off" spellcheck="false">
      <div class="controls-spacer"></div>
      <div class="result-count" id="result-count"></div>
    </div>
  </div>

  <div id="results" class="results"></div>

  <div class="kbd-hint">
    <kbd>Enter</kbd> search &nbsp;&middot;&nbsp; <kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd> <kbd>4</kbd> switch mode &nbsp;&middot;&nbsp; <kbd>Esc</kbd> close post
  </div>
</div>

<!-- Post detail -->
<div class="overlay" id="overlay"></div>
<div class="detail-panel" id="detail-panel">
  <div class="detail-close">
    <button onclick="closeDetail()">← back</button>
    <span class="post-id" id="detail-post-id"></span>
  </div>
  <div class="detail-inner" id="detail-content">
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

let currentMode = 'hybrid';
let searchTimeout = null;

// Stats
fetch('/api/stats')
  .then(r => r.json())
  .then(d => {
    if (d.error) { $('#stats').textContent = 'db offline'; return; }
    $('#stats').innerHTML =
      `<span>posts <span class="val">${fmt(d.posts)}</span></span>` +
      `<span>comments <span class="val">${fmt(d.comments)}</span></span>` +
      `<span>authors <span class="val">${fmt(d.authors)}</span></span>` +
      `<span>submolts <span class="val">${fmt(d.submolts)}</span></span>`;
  })
  .catch(() => { $('#stats').textContent = 'db offline'; });

function fmt(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'k';
  return n.toString();
}

// Mode buttons
$$('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    $$('.mode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentMode = btn.dataset.mode;
    const q = $('#query').value.trim();
    if (q) doSearch(q);
  });
});

// Search
$('#query').addEventListener('keydown', e => {
  if (e.key === 'Enter') {
    e.preventDefault();
    const q = e.target.value.trim();
    if (q) doSearch(q);
  }
});

// Keyboard shortcuts for mode switching
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') {
    if (e.target.id === 'query') {
      if (e.key === '1' || e.key === '2' || e.key === '3') return; // let typing happen
    }
    return;
  }
  if (e.key === '1') switchMode('hybrid');
  if (e.key === '2') switchMode('keyword');
  if (e.key === '3') switchMode('semantic');
  if (e.key === '4') switchMode('compare');
});

function switchMode(mode) {
  $$('.mode-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.mode === mode);
  });
  currentMode = mode;
  const q = $('#query').value.trim();
  if (q) doSearch(q);
}

async function doSearch(q) {
  const results = $('#results');
  const countEl = $('#result-count');

  results.innerHTML = '<div class="status"><span class="spinner"></span>searching...</div>';
  countEl.textContent = '';

  if (currentMode === 'compare') {
    return doCompare(q);
  }

  const params = new URLSearchParams({
    q, mode: currentMode, limit: 30
  });

  const author = $('#author-filter').value.trim();
  const submolt = $('#submolt-filter').value.trim();
  if (author) params.set('author', author);
  if (submolt) params.set('submolt', submolt);

  try {
    const res = await fetch('/api/search?' + params);
    const data = await res.json();

    if (data.error) {
      results.innerHTML = `<div class="status error">${esc(data.error)}</div>`;
      return;
    }

    if (!data.results.length) {
      results.innerHTML = '<div class="status">no results</div>';
      countEl.textContent = '0 results';
      return;
    }

    const modeLabel = data.mode === 'hybrid'
      ? `hybrid (${data.keyword_count}kw + ${data.semantic_count}sem)`
      : data.mode;

    countEl.textContent = `${data.results.length} results · ${modeLabel}`;

    results.innerHTML = data.results.map((r, i) => `
      <div class="result-item" style="animation-delay: ${i * 0.03}s" onclick="openPost('${esc(r.id)}')">
        <div class="result-meta">
          <span class="author">${esc(r.author)}</span>
          <span class="sep">/</span>
          <span class="submolt">${esc(r.submolt)}</span>
          <span class="sep">·</span>
          <span>${r.created_at || '?'}</span>
          <span class="sep">·</span>
          <span>↑${r.score ?? 0}</span>
        </div>
        <div class="result-title">${esc(r.title)}</div>
        ${r.snippet ? `<div class="result-snippet">${esc(r.snippet)}</div>` : ''}
        <div class="result-footer">
          ${(r.signals || []).map(s => `<span class="tag signal-${s}">${s}</span>`).join('')}
          ${r.score > 0 ? `<span class="tag">↑${r.score}</span>` : ''}
        </div>
      </div>
    `).join('');

  } catch (err) {
    results.innerHTML = `<div class="status error">${esc(err.message)}</div>`;
  }
}

async function doCompare(q) {
  const results = $('#results');
  const countEl = $('#result-count');

  try {
    const res = await fetch('/api/compare?q=' + encodeURIComponent(q) + '&limit=10');
    const data = await res.json();

    if (data.error) {
      results.innerHTML = `<div class="status error">${esc(data.error)}</div>`;
      return;
    }

    const kwIds = new Set(data.keyword.results.map(r => r.id));
    const semIds = new Set(data.semantic.results.map(r => r.id));
    const overlap = data.hybrid.overlap_count;

    countEl.textContent = `${overlap} results in both · ${data.keyword.unique_count} keyword-only · ${data.semantic.unique_count} semantic-only`;

    function renderCol(items, otherset) {
      return items.map((r, i) => {
        const inBoth = otherset.has(r.id) ? 'in-both' : '';
        return `<div class="compare-item ${inBoth}" onclick="openPost('${esc(r.id)}')">
          <div class="c-rank">#${i+1}</div>
          <div class="c-title">${esc(r.title)}</div>
          <div class="c-meta">
            <span class="author">${esc(r.author)}</span>
            <span style="opacity:0.3">·</span>
            <span>↑${r.score ?? 0}</span>
            ${inBoth ? '<span style="opacity:0.3">·</span><span style="color:var(--accent)">both</span>' : ''}
          </div>
        </div>`;
      }).join('');
    }

    results.innerHTML = `
      <div class="compare-grid">
        <div class="compare-col">
          <div class="compare-header kw">
            Keyword <span class="stat">${data.keyword.results.length} results</span>
          </div>
          <div class="compare-desc">PostgreSQL full-text search (tsvector). Finds exact word matches using stemming and ranking.</div>
          ${renderCol(data.keyword.results, semIds)}
        </div>
        <div class="compare-col">
          <div class="compare-header sem">
            Semantic <span class="stat">${data.semantic.results.length} results</span>
          </div>
          <div class="compare-desc">Vector similarity (pgvector + BGE-small). Finds conceptually similar content even without shared words.</div>
          ${renderCol(data.semantic.results, kwIds)}
        </div>
        <div class="compare-col">
          <div class="compare-header hyb">
            Hybrid <span class="stat">${data.hybrid.results.length} results</span>
          </div>
          <div class="compare-desc">RRF fusion (k=60) of both, then upvote boost (log-scaled) and author diversity penalty (50%/dup).</div>
          ${data.hybrid.results.map((r, i) => `<div class="compare-item" onclick="openPost('${esc(r.id)}')">
            <div class="c-rank">#${i+1}</div>
            <div class="c-title">${esc(r.title)}</div>
            <div class="c-meta">
              <span class="author">${esc(r.author)}</span>
              <span style="opacity:0.3">·</span>
              <span>↑${r.score ?? 0}</span>
              <span style="opacity:0.3">·</span>
              ${(r.signals || []).map(s => `<span class="signal-${s}" style="font-size:0.6rem">${s}</span>`).join('+')}
            </div>
          </div>`).join('')}
        </div>
      </div>
    `;

  } catch (err) {
    results.innerHTML = `<div class="status error">${esc(err.message)}</div>`;
  }
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// --- Post detail panel ---

async function openPost(postId) {
  const panel = $('#detail-panel');
  const overlay = $('#overlay');
  const content = $('#detail-content');
  const postIdEl = $('#detail-post-id');

  postIdEl.textContent = postId;
  content.innerHTML = '<div class="status"><span class="spinner"></span>loading post...</div>';

  overlay.classList.add('open');
  panel.classList.add('open');
  document.body.style.overflow = 'hidden';

  try {
    const res = await fetch('/api/post/' + encodeURIComponent(postId));
    const post = await res.json();

    if (post.error) {
      content.innerHTML = `<div class="status error">${esc(post.error)}</div>`;
      return;
    }

    let html = `
      <div class="detail-meta">
        <span class="author">${esc(post.author)}</span>
        ${post.author_karma != null ? `<span>(${post.author_karma} karma)</span>` : ''}
        <span class="sep">/</span>
        <span class="submolt">${esc(post.submolt)}</span>
        <span class="sep">·</span>
        <span>${post.created_at || '?'}</span>
      </div>
      <div class="detail-title">${esc(post.title)}</div>
      <div class="detail-votes">
        <span class="up">↑ ${post.upvotes ?? 0}</span>
        <span class="down">↓ ${post.downvotes ?? 0}</span>
        <span>score: ${post.score ?? 0}</span>
        <span>${post.comment_count ?? 0} comments</span>
      </div>
    `;

    if (post.url) {
      html += `<div style="margin-bottom:1rem;font-family:var(--mono);font-size:0.8rem"><a href="${esc(post.url)}" target="_blank" rel="noopener" style="color:var(--accent);text-decoration:none">${esc(post.url)}</a></div>`;
    }

    if (post.content) {
      html += `<div class="detail-content">${esc(post.content)}</div>`;
    }

    // Graph section — load async
    html += '<div id="graph-section-slot"></div>';

    // Comments
    if (post.comments && post.comments.length > 0) {
      html += `<div class="comments-header">${post.comments.length} comment${post.comments.length !== 1 ? 's' : ''}</div>`;
      html += renderComments(post.comments);
    } else {
      html += '<div class="no-comments">no comments</div>';
    }

    content.innerHTML = html;

    // Load graph data for author
    loadAuthorGraph(post.author);

  } catch (err) {
    content.innerHTML = `<div class="status error">${esc(err.message)}</div>`;
  }
}

function renderComments(comments) {
  // Build a tree from flat list
  const byId = {};
  const roots = [];
  comments.forEach(c => { byId[c.id] = { ...c, children: [] }; });
  comments.forEach(c => {
    if (c.parent_id && byId[c.parent_id]) {
      byId[c.parent_id].children.push(byId[c.id]);
    } else {
      roots.push(byId[c.id]);
    }
  });

  let out = '';
  function walk(node, depth) {
    const cls = depth === 0 ? '' :
                depth === 1 ? 'comment-indent-1' :
                depth === 2 ? 'comment-indent-2' :
                depth === 3 ? 'comment-indent-3' : 'comment-indent-deep';
    out += `<div class="comment ${cls}">
      <div class="comment-head">
        <span class="author">${esc(node.author)}</span>
        <span style="opacity:0.3">·</span>
        <span>${node.created_at || '?'}</span>
        <span style="opacity:0.3">·</span>
        <span>↑${node.score ?? 0}</span>
      </div>
      <div class="comment-body">${esc(node.content)}</div>
    </div>`;
    node.children.forEach(child => walk(child, depth + 1));
  }
  roots.forEach(r => walk(r, 0));
  return out;
}

async function loadAuthorGraph(authorName) {
  const slot = document.getElementById('graph-section-slot');
  if (!slot) return;

  try {
    const res = await fetch('/api/author/' + encodeURIComponent(authorName) + '/graph');
    const g = await res.json();

    if (!g.available || !g.found) {
      slot.innerHTML = '';
      return;
    }

    const chipList = (items, labelKey, countKey) =>
      items.map(x => `<span class="graph-chip">${esc(x[labelKey])}<span class="count">${x[countKey]}</span></span>`).join('');

    slot.innerHTML = `
      <div class="graph-section">
        <div class="graph-section-title">Author Graph — ${esc(authorName)}</div>
        ${g.top_submolts.length ? `
          <div class="graph-row">
            <span class="graph-label">posts in</span>
            ${chipList(g.top_submolts, 'name', 'count')}
          </div>
        ` : ''}
        ${g.replies_to.length ? `
          <div class="graph-row">
            <span class="graph-label">replies to</span>
            ${chipList(g.replies_to, 'name', 'count')}
          </div>
        ` : ''}
        ${g.replied_by.length ? `
          <div class="graph-row">
            <span class="graph-label">replied by</span>
            ${chipList(g.replied_by, 'name', 'count')}
          </div>
        ` : ''}
        ${g.community_size > 0 ? `
          <div class="graph-row">
            <span class="graph-label">community</span>
            <span class="graph-chip">${g.community_size} authors</span>
            ${g.community_members.slice(0, 8).map(m => `<span class="graph-chip">${esc(m)}</span>`).join('')}
          </div>
        ` : ''}
      </div>
    `;
  } catch (e) {
    slot.innerHTML = '';
  }
}

function closeDetail() {
  $('#detail-panel').classList.remove('open');
  $('#overlay').classList.remove('open');
  document.body.style.overflow = '';
}

$('#overlay').addEventListener('click', closeDetail);

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeDetail();
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    print(f"\n  Moltbook Archive → http://localhost:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
