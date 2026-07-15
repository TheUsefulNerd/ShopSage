#!/usr/bin/env python3
"""Retrieval + shortlist logic for ShopSage Week 1 RAG pipeline.

Two public functions:
  - retrieve(query, top_k, max_distance) → list of candidate dicts
  - get_shortlist(query, top_k, customer_age) → (answer_str, filtered_candidates)

And one helper:
  - extract_budget(query) → (min_budget, max_budget)

Design notes:
  - Budget filtering is done deterministically in Python (not by the LLM), so
    the guardrail ("never recommend over-budget items") can't be silently dropped
    by a bad generation.
  - Age-restriction filtering uses categories.is_age_sensitive only.
    products.age_restricted was confirmed as a duplicate on current data
    (nunique()==1 per category) and is intentionally ignored here.
  - MAX_RELEVANT_DISTANCE is a cosine-distance cutoff for what counts as a real
    candidate. 0.6 is a starting guess for all-MiniLM-L6-v2 on short product
    text. Tune it against real queries: run retrieve(..., max_distance=None) on
    a range of queries, print the distances, and set the cutoff just above where
    good matches end and bad ones begin.
"""

from __future__ import annotations

import os
import re
from typing import Optional

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "shopsage_products"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Cosine-distance cutoff: 0 = identical, 2 = opposite direction.
# Hits beyond this threshold are dropped before the LLM sees them.
# TUNE THIS — don't trust the default blindly.
# See the notebook's "=== debug: no cutoff ===" cell output for your dataset's
# actual distance distribution, then pick a value that cleanly separates
# relevant from irrelevant hits.
MAX_RELEVANT_DISTANCE: float = 0.6

SYSTEM_PROMPT = (
    "You are ShopSage, a budget-aware shopping assistant. You are given a user query "
    "and a pre-filtered list of candidate products (already within budget and age-appropriate). "
    "Recommend the 2-3 best matches from the candidates only — never invent products or attributes "
    "not present in the candidate list. For each pick, give a one-sentence reason tied to the query. "
    "If no candidates fit, say so plainly and do not force a recommendation."
)

# Budget regex patterns — handles decimals and both upper and lower bounds.
# Known gap: "between $X and $Y" is not handled yet.
_UPPER_RE = re.compile(
    r"(?:under|below|less than|up to|no more than)\s*\$?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_LOWER_RE = re.compile(
    r"(?:over|above|more than|at least)\s*\$?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Module-level singletons (lazy-initialized on first use)
# ---------------------------------------------------------------------------

_embedder: Optional[SentenceTransformer] = None
_groq_client: Optional[Groq] = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    collection: chromadb.Collection,
    top_k: int = 5,
    max_distance: Optional[float] = MAX_RELEVANT_DISTANCE,
) -> list[dict]:
    """Embed query and return the top_k most similar products from Chroma.

    Args:
        query: Natural language user query.
        collection: The Chroma collection to search.
        top_k: Number of results to fetch from Chroma before distance filtering.
        max_distance: Cosine-distance cutoff. Hits beyond this are dropped.
                      Pass None to disable filtering (useful for debugging
                      and calibrating the cutoff value).

    Returns:
        List of candidate dicts, each with all metadata fields plus:
        - "chunk_text": the embedded document text
        - "distance": cosine distance (lower = more similar)
    """
    embedder = _get_embedder()
    query_embedding = embedder.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if max_distance is not None and dist > max_distance:
            continue
        hits.append({**meta, "chunk_text": doc, "distance": dist})
    return hits


# ---------------------------------------------------------------------------
# Budget parsing
# ---------------------------------------------------------------------------

def extract_budget(query: str) -> tuple[Optional[float], Optional[float]]:
    """Parse budget constraints from a natural-language query.

    Returns:
        (min_budget, max_budget) — either can be None if not stated.

    Handles:
        - "under $80", "below $80", "less than $80", "up to $80",
          "no more than $80" → max_budget
        - "over $50", "above $50", "more than $50",
          "at least $50" → min_budget
        - Decimals: "$79.99"

    Known gap: "between $X and $Y" is not handled.
    """
    upper = _UPPER_RE.search(query)
    lower = _LOWER_RE.search(query)
    max_budget = float(upper.group(1)) if upper else None
    min_budget = float(lower.group(1)) if lower else None
    return min_budget, max_budget


# ---------------------------------------------------------------------------
# Shortlist generation
# ---------------------------------------------------------------------------

def get_shortlist(
    query: str,
    collection: chromadb.Collection,
    top_k: int = 8,
    customer_age: Optional[int] = None,
) -> tuple[str, list[dict]]:
    """Full RAG pipeline: retrieve → filter → LLM rank + narrate.

    Steps:
      1. Parse budget from query (deterministic regex).
      2. Retrieve top_k candidates from Chroma with distance cutoff.
      3. Deterministically filter by budget and age restriction.
      4. Pass up to 3 survivors to Groq to write the human-readable answer.

    The LLM only ever sees already-filtered candidates — it doesn't decide
    which products are in budget or age-appropriate.

    Args:
        query: User's natural-language shopping request.
        collection: Chroma collection to search.
        top_k: How many Chroma hits to retrieve before filtering.
        customer_age: Customer's age, if known. If None, age-sensitive items
                      are NOT filtered out (unknown age = pass-through for now;
                      update this default once Week 2 wires up per-user context).

    Returns:
        (answer, filtered_candidates)
        - answer: Human-readable string from Groq (or a "nothing found" message).
        - filtered_candidates: The full filtered list (useful for debug/logging).
    """
    min_budget, max_budget = extract_budget(query)
    candidates = retrieve(query, collection, top_k=top_k)

    filtered = []
    for c in candidates:
        # Budget guardrail (deterministic — not delegated to LLM)
        if max_budget is not None and c["base_price"] > max_budget:
            continue
        if min_budget is not None and c["base_price"] < min_budget:
            continue
        # Age guardrail — uses is_age_sensitive (from categories table) only.
        # products.age_restricted is intentionally ignored here; it was
        # confirmed to be a duplicate field on the current dataset.
        if c["is_age_sensitive"] and customer_age is not None and customer_age < 18:
            continue
        filtered.append(c)

    if not filtered:
        return (
            "I couldn't find anything matching that budget/criteria in the catalog. "
            "Want me to widen the search?",
            [],
        )

    top_picks = filtered[:3]  # best-ranked survivors (Chroma already sorted by distance)
    candidate_block = "\n".join(
        f"- {c['title']} | ${c['base_price']} | {c['brand_name']} | "
        f"{c['category_name']} | {c['color_family']}"
        for c in top_picks
    )

    groq = _get_groq_client()
    completion = groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"User query: {query}\n\nCandidates:\n{candidate_block}",
            },
        ],
        temperature=0.3,
    )
    return completion.choices[0].message.content, filtered


# ---------------------------------------------------------------------------
# Debug helper — use this to calibrate MAX_RELEVANT_DISTANCE
# ---------------------------------------------------------------------------

def debug_distances(
    queries: list[str],
    collection: chromadb.Collection,
    top_k: int = 10,
) -> None:
    """Print raw distances (no cutoff) for a list of queries.

    Use this to calibrate MAX_RELEVANT_DISTANCE:
      1. Pick a set of queries — some that should find relevant results
         and some that shouldn't (e.g., categories not in the catalog).
      2. Look at where the relevant vs. irrelevant hits diverge.
      3. Set MAX_RELEVANT_DISTANCE just above that gap.

    Example:
        from scripts.rag.retrieval import debug_distances
        debug_distances(
            ["waterproof jacket", "house party supplies"],
            collection,
        )
    """
    for q in queries:
        print(f"\n=== {q} ===")
        hits = retrieve(q, collection, top_k=top_k, max_distance=None)
        for h in hits:
            print(
                f"  {h['distance']:.3f}  ${h['base_price']:<8}  "
                f"{h['title']}  ({h['category_name']})"
            )
