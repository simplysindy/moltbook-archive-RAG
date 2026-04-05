# Moltbook Archive

Hybrid search engine over 2.4M posts from [Moltbook](https://moltbook.com), an AI agent social network. Combines full-text search, vector similarity, and graph-based community detection into a single ranked result set.

![Search UI](screenshots/search_ui.png)

## How It Works

The search pipeline runs three retrieval strategies and fuses them:

```
Query
  ├─ Keyword search (PostgreSQL tsvector, BM25-style ranking)
  ├─ Semantic search (pgvector, BGE-small-en embeddings)
  └─ Graph context (Neo4j, Louvain community detection)
        │
        ▼
  Reciprocal Rank Fusion (k=60)
        │
        ▼
  Upvote boost ── log-scaled score weighting
        │
        ▼
  Author diversity ── 50% penalty per duplicate author
        │
        ▼
  Final ranked results
```

**Compare mode** shows all three engines side by side, so you can see what each one contributes and where they disagree.

![Compare Mode](screenshots/compare.png)

## Graph Layer

Post detail panels show each author's graph profile from Neo4j:

- **Community** — Louvain-detected cluster of authors connected by reply patterns
- **Replies to / Replied by** — strongest conversational connections
- **Top submolts** — where this author is most active

![Author Graph](screenshots/graph.png)

This surfaces relationships that keyword and vector search can't — who talks to whom, which authors form communities, and how the social structure maps to content.

## Data Pipeline

| Component | What it does |
|---|---|
| **Scraper daemon** | Continuous API ingestion, ~55 req/min, cursor-based resumable |
| **Incremental sync** | Postgres to Neo4j + embedding generation (BGE-small-en-v1.5) |
| **Vote refresh** | Periodic bulk update of vote counts across the full archive |

All three run as systemd services (see `services/`). The scraper and sync run 24/7; vote refresh is a one-shot triggered manually or on schedule.

## Stack

- **PostgreSQL** — posts, comments, authors, full-text indexes (tsvector), vector indexes (pgvector)
- **Neo4j** — author graph with REPLIED_TO, POSTED_IN, and WROTE edges; Louvain community detection via GDS
- **Flask** — API + single-page frontend (inline HTML, no build step)
- **BGE-small-en-v1.5** — 384-dim embeddings for semantic search
- **Python** — scraper, sync daemon, search CLI, web app

## Project Structure

```
├── src/
│   ├── app.py              # Flask web UI + API
│   ├── search.py           # CLI search (keyword, semantic, hybrid, community)
│   ├── scrape.py           # API scraper (one-off or daemon)
│   ├── incremental_sync.py # Postgres → Neo4j + embeddings daemon
│   ├── sync_graph.py       # One-shot full Neo4j sync
│   ├── setup_db.py         # Database schema creation
│   ├── seed.py             # HuggingFace parquet import
│   └── embed_posts.py      # Batch embedding backfill
├── eval/                   # RAG evaluation suite
├── services/               # systemd unit files
└── LICENSE
```

## Running Locally

```bash
# start the web UI
python src/app.py

# search from CLI
python src/search.py "agent identity persistence"

# compare modes
python src/search.py "claw dance" --keyword-only
python src/search.py "claw dance" --semantic-only
```

## Eval

35-question retrieval evaluation across 8 categories (factual recall, synthesis, temporal, negative, author-filtered, mode comparison, community, edge cases). Results in `eval/`.

## Current Scale

| | Count |
|---|---|
| Posts | 2.47M |
| Comments | 4.26M |
| Authors | 176K |
| Submolts | 16K |
| Embeddings | ~1M (spam excluded) |
| Neo4j nodes | 2.6M |
| Neo4j edges | 290K aggregated |

## License

MIT
