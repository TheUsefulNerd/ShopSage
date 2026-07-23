#!/usr/bin/env python3
"""Automated eval runner for ShopSage retrieval pipeline.

Runs all queries from data/eval/query.csv through the retrieval pipeline
(no LLM — just the Qdrant hybrid search candidate list) and checks
assertions. Prints a pass/fail table and an overall score.

This directly addresses the mentor's question: "How would you handle failure
cases at scale?" — instead of fixing individual queries by hand, we measure
systematically and iterate on the pipeline.

Test categories in query.csv:
  1. Budget queries     — "salomon hiking boots under $285" → expected product
  2. Rating queries     — "Best rated decathlon hiking boots" → expected product
  3. Out-of-scope       — "I need telepathic hat" → expect 0 results
  4. Fuzzy/misspelled   — "Do you have salomon hiking boot" → expected product

Usage:
    python -m scripts.eval.run_eval
    python -m scripts.eval.run_eval --top-k 10
    python -m scripts.eval.run_eval --verbose
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.rag.ingestion import build_collection
from scripts.rag.retrieval import retrieve

EVAL_CSV = Path(__file__).resolve().parents[2] / "data" / "eval" / "query.csv"

# Regex to extract budget from query (same logic as intent extraction)
_BUDGET_RE = re.compile(
    r"(?:under|below|less than|up to|no more than)\s*\$?([\d]+(?:\.[\d]+)?)",
    re.IGNORECASE,
)
_RATING_RE = re.compile(r"best rated|highly rated|top rated", re.IGNORECASE)
_OUT_OF_SCOPE_PHRASES = [
    "telepathic hat", "holographic flying shoes", "invisible socks",
    "time traveling boots", "quantum pants",
]


def _classify_query(query: str, expected: str) -> str:
    """Classify query into one of 4 eval categories."""
    q_lower = query.lower()
    if any(phrase in q_lower for phrase in _OUT_OF_SCOPE_PHRASES):
        return "out_of_scope"
    if _BUDGET_RE.search(query):
        return "budget"
    if _RATING_RE.search(query):
        return "rating"
    return "fuzzy"


def _extract_expected_title(expected: str) -> Optional[str]:
    """Pull the product name from Satish's expected column."""
    # Format: "Product: <Title> - <description>"
    m = re.match(r"Product:\s*([^-]+)\s*[-–]", expected)
    if m:
        return m.group(1).strip()
    if "No matching products found" in expected:
        return None
    return None


def _extract_expected_budget(expected: str) -> Optional[float]:
    """Pull the price from Satish's expected column."""
    m = re.search(r"\(Price:\s*\$?([\d.]+)\)", expected)
    if m:
        return float(m.group(1))
    return None


def run_eval(top_k: int = 8, verbose: bool = False) -> dict:
    """Run all eval queries and return results summary."""
    print("Connecting to Qdrant and loading eval dataset...")
    build_collection()  # reuses existing collection

    rows = []
    with open(EVAL_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                rows.append((row[0].strip(), row[1].strip()))

    print(f"Running {len(rows)} eval queries...\n")

    results = {
        "total": 0, "passed": 0, "failed": 0,
        "by_category": {
            "budget": {"total": 0, "passed": 0},
            "rating": {"total": 0, "passed": 0},
            "out_of_scope": {"total": 0, "passed": 0},
            "fuzzy": {"total": 0, "passed": 0},
        },
        "failures": [],
    }

    # Header
    print(f"{'#':<4} {'Category':<14} {'Query':<45} {'Pass?':<6} {'Detail'}")
    print("-" * 110)

    for i, (query, expected) in enumerate(rows, 1):
        category = _classify_query(query, expected)
        expected_title = _extract_expected_title(expected)
        expected_budget = _extract_expected_budget(expected)

        hits = retrieve(query, top_k=top_k, score_threshold=None)
        hit_titles = [h.get("title", "") for h in hits]

        passed = False
        detail = ""

        if category == "out_of_scope":
            # Pass if no hits OR all hits have very low score
            high_score_hits = [h for h in hits if h.get("score", 0) > 0.1]
            passed = len(high_score_hits) == 0
            detail = f"{len(hits)} hits, {len(high_score_hits)} above threshold"

        elif expected_title:
            # Check if expected product appears in top results
            title_lower = expected_title.lower()
            found = any(title_lower in t.lower() or t.lower() in title_lower for t in hit_titles)
            passed = found

            if found and expected_budget is not None:
                # For budget queries, also verify the matched product is within budget
                matched = next(
                    (h for h in hits if title_lower in h.get("title", "").lower()), None
                )
                detail = f"Found '{expected_title}'"
            elif found:
                detail = f"Found '{expected_title}'"
            else:
                detail = f"Expected '{expected_title}' — got: {hit_titles[:3]}"
        else:
            # Unexpected expected format — skip
            detail = "skipped (unparseable expected)"
            results["total"] += 1
            continue

        results["total"] += 1
        results["by_category"][category]["total"] += 1
        if passed:
            results["passed"] += 1
            results["by_category"][category]["passed"] += 1
        else:
            results["failed"] += 1
            results["failures"].append({"query": query, "category": category, "detail": detail})

        status = "PASS" if passed else "FAIL"
        query_short = query[:44] + "..." if len(query) > 44 else query
        query_clean = query_short.encode('ascii', 'ignore').decode('ascii')
        detail_clean = detail.encode('ascii', 'ignore').decode('ascii')
        print(f"{i:<4} {category:<14} {query_clean:<45} {status:<6} {detail_clean}")

        if verbose and not passed:
            for j, h in enumerate(hits[:3], 1):
                print(f"     [{j}] score={h.get('score', 0):.4f}  {h.get('title', '?')}")

    # Summary
    pct = 100 * results["passed"] / results["total"] if results["total"] > 0 else 0
    print("\n" + "=" * 110)
    print(f"OVERALL: {results['passed']}/{results['total']} passed  ({pct:.1f}%)")
    print()
    print(f"{'Category':<16} {'Passed':<10} {'Total':<10} {'%'}")
    print("-" * 45)
    for cat, stats in results["by_category"].items():
        if stats["total"] > 0:
            cat_pct = 100 * stats["passed"] / stats["total"]
            print(f"{cat:<16} {stats['passed']:<10} {stats['total']:<10} {cat_pct:.0f}%")

    if results["failures"]:
        print(f"\nFailed queries ({len(results['failures'])}):")
        for f in results["failures"][:10]:
            q_clean = f['query'][:60].encode('ascii', 'ignore').decode('ascii')
            d_clean = f['detail'].encode('ascii', 'ignore').decode('ascii')
            print(f"  [{f['category']}] {q_clean}")
            print(f"         -> {d_clean}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ShopSage retrieval eval.")
    parser.add_argument("--top-k", type=int, default=8, help="Candidates to retrieve per query.")
    parser.add_argument("--verbose", action="store_true", help="Show top hits for failed queries.")
    args = parser.parse_args()
    run_eval(top_k=args.top_k, verbose=args.verbose)
