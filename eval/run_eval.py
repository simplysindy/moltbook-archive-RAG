#!/usr/bin/env python3
"""Moltbook RAG evaluation runner.

Runs each eval question through the search system in all applicable modes,
saves raw results for hand-scoring. Does NOT auto-score — retrieval quality
requires human judgement on relevance.

Usage:
    python eval/run_eval.py                     # run all questions
    python eval/run_eval.py --category factual  # run one category
    python eval/run_eval.py --id F01            # run one question
    python eval/run_eval.py --score             # open scoring UI (after run)
"""

import argparse
import json
import subprocess
import os
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).parent
QUESTIONS_FILE = EVAL_DIR / "questions.json"
RESULTS_FILE = EVAL_DIR / "results.json"
SCORES_FILE = EVAL_DIR / "scores.json"
SEARCH_SCRIPT = EVAL_DIR.parent / "src" / "search.py"
VENV_PYTHON = os.environ.get("PYTHON", sys.executable)


def run_search(query: str, mode: str = "hybrid", author: str = None,
               submolt: str = None, community_of: str = None,
               after: str = None, before: str = None,
               limit: int = 10) -> dict:
    """Run a search query and return parsed results."""
    if not query:
        return {"error": "empty query", "results": [], "time_ms": 0}

    python = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
    cmd = [python, str(SEARCH_SCRIPT), query, "--limit", str(limit)]

    if mode == "keyword":
        cmd.append("--keyword-only")
    elif mode == "semantic":
        cmd.append("--semantic-only")
    # hybrid is default

    if author:
        cmd.extend(["--author", author])
    if submolt:
        cmd.extend(["--submolt", submolt])
    if community_of:
        cmd.extend(["--community-of", community_of])
    if after:
        cmd.extend(["--after", after])
    if before:
        cmd.extend(["--before", before])

    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
            cwd=str(SEARCH_SCRIPT.parent)
        )
        elapsed_ms = int((time.time() - start) * 1000)

        if result.returncode != 0:
            return {
                "error": result.stderr.strip(),
                "results": [],
                "time_ms": elapsed_ms
            }

        return {
            "raw_output": result.stdout.strip(),
            "time_ms": elapsed_ms
        }
    except subprocess.TimeoutExpired:
        return {"error": "timeout (60s)", "results": [], "time_ms": 60000}
    except Exception as e:
        return {"error": str(e), "results": [], "time_ms": 0}


def run_question(q: dict) -> dict:
    """Run a single eval question across applicable modes."""
    qid = q["id"]
    query = q["query"]
    params = q.get("search_params", {})
    category = q["category"]

    print(f"  [{qid}] {query[:60]}...")

    result = {"id": qid, "category": category, "query": query, "modes": {}}

    # Always run hybrid
    result["modes"]["hybrid"] = run_search(
        query, mode="hybrid",
        author=params.get("author"),
        submolt=params.get("submolt"),
        community_of=params.get("community_of"),
        after=params.get("after"),
        before=params.get("before"),
    )

    # For mode_comparison questions, also run keyword-only and semantic-only
    if category == "mode_comparison":
        result["modes"]["keyword"] = run_search(query, mode="keyword")
        result["modes"]["semantic"] = run_search(query, mode="semantic")

    # For author questions, also run without author filter for comparison
    if category == "author" and params.get("author"):
        result["modes"]["hybrid_unfiltered"] = run_search(query, mode="hybrid")

    # For community questions, also run without community filter
    if category == "community" and params.get("community_of"):
        result["modes"]["hybrid_unfiltered"] = run_search(query, mode="hybrid")

    return result


def run_all(questions: list[dict]) -> list[dict]:
    """Run all questions and return results."""
    results = []
    categories = {}
    for q in questions:
        cat = q["category"]
        categories.setdefault(cat, []).append(q)

    for cat, qs in categories.items():
        print(f"\n=== {cat.upper()} ({len(qs)} questions) ===")
        for q in qs:
            results.append(run_question(q))

    return results


def print_summary(results: list[dict]):
    """Print timing summary."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    by_cat = {}
    for r in results:
        cat = r["category"]
        by_cat.setdefault(cat, []).append(r)

    total_queries = 0
    total_errors = 0
    total_time = 0

    for cat, rs in by_cat.items():
        times = []
        errors = 0
        for r in rs:
            for mode, data in r["modes"].items():
                t = data.get("time_ms", 0)
                times.append(t)
                total_time += t
                total_queries += 1
                if "error" in data:
                    errors += 1
                    total_errors += 1

        avg = sum(times) / len(times) if times else 0
        print(f"  {cat:20s}: {len(rs):2d} questions, "
              f"{len(times):2d} queries, "
              f"avg {avg:.0f}ms, "
              f"{errors} errors")

    print(f"\n  Total: {total_queries} queries in {total_time}ms, "
          f"{total_errors} errors")
    print(f"\n  Results saved to: {RESULTS_FILE}")
    print(f"  Next step: review results and score with --score")


def score_interactive(results: list[dict], questions: list[dict]):
    """Interactive scoring: show each result, ask for relevance judgement."""
    q_lookup = {q["id"]: q for q in questions}

    # Load existing scores
    if SCORES_FILE.exists():
        scores = json.loads(SCORES_FILE.read_text())
    else:
        scores = {}

    print("\nScoring guide:")
    print("  3 = highly relevant (exact match or directly answers the query)")
    print("  2 = relevant (related content, useful context)")
    print("  1 = marginally relevant (tangentially related)")
    print("  0 = not relevant")
    print("  s = skip this question")
    print("  q = quit and save\n")

    for r in results:
        qid = r["id"]
        if qid in scores:
            print(f"  [{qid}] already scored, skipping")
            continue

        q = q_lookup.get(qid, {})
        print(f"\n{'=' * 60}")
        print(f"[{qid}] Category: {r['category']}")
        print(f"Query: {r['query']}")
        if q.get("notes"):
            print(f"Notes: {q['notes']}")
        if q.get("expected_authors"):
            print(f"Expected authors: {q['expected_authors']}")

        for mode, data in r["modes"].items():
            print(f"\n--- Mode: {mode} ({data.get('time_ms', '?')}ms) ---")
            if "error" in data:
                print(f"  ERROR: {data['error']}")
            elif "raw_output" in data:
                # Show first 2000 chars of output
                output = data["raw_output"][:2000]
                for line in output.split("\n"):
                    print(f"  {line}")
                if len(data["raw_output"]) > 2000:
                    print(f"  ... ({len(data['raw_output']) - 2000} chars truncated)")

        response = input(f"\nScore for [{qid}] (0-3, s=skip, q=quit): ").strip()
        if response == "q":
            break
        if response == "s":
            continue
        try:
            score = int(response)
            if 0 <= score <= 3:
                scores[qid] = {
                    "score": score,
                    "notes": input("  Notes (optional): ").strip() or None
                }
        except ValueError:
            print("  Invalid, skipping")

    SCORES_FILE.write_text(json.dumps(scores, indent=2) + "\n")
    print(f"\nScores saved to: {SCORES_FILE}")

    # Print score summary
    if scores:
        scored = [s["score"] for s in scores.values()]
        print(f"Scored: {len(scored)}/{len(results)}")
        print(f"Average: {sum(scored) / len(scored):.2f}")
        for level in [3, 2, 1, 0]:
            count = scored.count(level)
            print(f"  {level}: {count} ({count/len(scored)*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Moltbook RAG eval runner")
    parser.add_argument("--category", help="Run only this category")
    parser.add_argument("--id", help="Run only this question ID")
    parser.add_argument("--score", action="store_true",
                        help="Interactive scoring mode")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_FILE.read_text())

    if args.score:
        if not RESULTS_FILE.exists():
            print("No results file found. Run eval first.")
            return
        results = json.loads(RESULTS_FILE.read_text())
        score_interactive(results, questions)
        return

    # Filter questions
    if args.id:
        questions = [q for q in questions if q["id"] == args.id]
    elif args.category:
        questions = [q for q in questions if q["category"] == args.category]

    if not questions:
        print("No questions match filter.")
        return

    print(f"Running {len(questions)} eval questions...")

    results = run_all(questions)

    # Save results
    RESULTS_FILE.write_text(json.dumps(results, indent=2) + "\n")
    print_summary(results)


if __name__ == "__main__":
    main()
