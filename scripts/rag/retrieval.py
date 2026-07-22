#!/usr/bin/env python3
"""Retrieval logic for ShopSage — Chroma search and candidate filtering.

No Groq/LLM code lives here. That's in app.py now, so this file has zero
dependency on the app's entry point.

Two public functions:
  - retrieve(query, top_k, max_distance) → list of candidate dicts
  - build_candidate_block(query, top_k, customer_age) → ALL age-filtered
    candidate dicts, up to top_k in length (empty list if none survive
    filtering). Does NOT slice to top 3 — see its own docstring for why.

Design notes:
  - Budget/color/size filtering is NOT done here — app.py's chat_fn applies
    those deterministically, AFTER live-enriching candidates via
    get_product_details (tools.py), since price/stock must come from a live
    lookup, not this module's Chroma-only data. This file only knows about
    age (static, safe to keep in frozen Chroma metadata) — see below.
  - Age-restriction filtering uses categories.is_age_sensitive only.
    products.age_restricted was confirmed as a duplicate on current data
    (nunique()==1 per category) and is intentionally ignored.
  - MAX_RELEVANT_DISTANCE is a cosine-distance cutoff for what counts as a real
    candidate. 0.6 is a starting guess for all-MiniLM-L6-v2 on short product
    text. Tune it against real queries: run retrieve(..., max_distance=None) on
    a range of queries, print the distances, and set the cutoff just above where
    good matches end and bad ones begin.
"""

from __future__ import annotations

import os
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

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

# Cosine-distance cutoff: 0 = identical, 2 = opposite direction.
# Hits beyond this threshold are dropped before the LLM sees them.
# TUNE THIS — don't trust the default blindly.
# See the notebook's "=== debug: no cutoff ===" cell output for your dataset's
# actual distance distribution, then pick a value that cleanly separates
# relevant from irrelevant hits.
MAX_RELEVANT_DISTANCE: float = 2


# ---------------------------------------------------------------------------
# Module-level singletons (lazy-initialized on first use)
# ---------------------------------------------------------------------------

_embedder: Optional[SentenceTransformer] = None
_hf_token: Optional[str] = None


def _get_hf_token() -> str:
    global _hf_token
    if _hf_token is None:
        api_key = os.environ.get("HF_TOKEN")
        if not api_key:
            raise EnvironmentError(
                "HF_TOKEN is not set. Add it to your .env file."
            )
        _hf_token = api_key
    return _hf_token


def _get_embedder() -> HuggingFaceEmbeddings:
    global _embedder
    if _embedder is None:
        print("[_get_embedder] CREATING embedder for the first time this process...")
        hf_token = _get_hf_token()
        _embedder = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu", "token": hf_token},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("[_get_embedder] Embedder created and cached.")
    else:
        print("[_get_embedder] Reusing already-cached embedder.")
    return _embedder


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
    query_embedding = embedder.embed_documents([query])
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
            # NOTE: base_price no longer printed — it's not in Chroma
            # metadata anymore (removed to avoid stale frozen prices).
            print(
                f"  {h['distance']:.3f}  {h['title']}  ({h['category_name']})"
            )


# ---------------------------------------------------------------------------
# Candidate finder (Groq call removed — see app.py's narrate_shortlist() for
# the LLM-narrated version, which builds its own prompt text from this list)
# ---------------------------------------------------------------------------

def build_candidate_block(
    query: str,
    collection: chromadb.Collection,
    top_k: int = 8,
    customer_age: Optional[int] = None,
) -> list[dict]:
    """Retrieve → filter by age → return ALL survivors (up to top_k). No LLM
    call here, and NO top-3 slicing here either (see NOTE below).

    Renamed from get_shortlist(): "shortlist" implied an LLM-narrated answer,
    which no longer happens in this function. It also no longer builds any
    text block itself — it just returns the filtered candidate list; the
    caller (app.py's chat_fn) decides what to do with an empty list and how
    to format candidates into prompt text.

    NOTE: this used to slice to the top 3 internally, but that slice moved
    to the caller (chat_fn) — a live stock-check (get_product_details,
    tools.py) needs to run and filter out out-of-stock candidates BEFORE the
    final top-3 selection, not after. Slicing to 3 here would leave nothing
    to backfill with if some of those 3 turn out to be out of stock.

    Steps:
      1. Retrieve top_k candidates from Chroma with distance cutoff.
      2. Deterministically filter by age restriction.
      3. Return ALL survivors (caller applies stock-filter + top-3 slice).

    NOTE: budget filtering does NOT happen here — this function has no
    access to live price data (Chroma-only). app.py's chat_fn applies
    budget filtering deterministically, after enriching candidates via
    get_product_details (tools.py), using real base_price values.

    Args:
        query: User's natural-language shopping request.
        collection: Chroma collection to search.
        top_k: How many Chroma hits to retrieve before filtering.
        customer_age: Customer's age, if known. If None, age-sensitive items
                      are NOT filtered out (unknown age = pass-through for now;
                      update this default once Week 2 wires up per-user context).

    Returns:
        list[dict] — ALL age-filtered candidates, best-ranked first (Chroma
        already sorts by distance), up to top_k in length. Empty list if
        nothing survives filtering — the caller is responsible for handling
        that case.
    """
    candidates = retrieve(query, collection, top_k=top_k)

    filtered = []
    for c in candidates:
        # Age guardrail — uses is_age_sensitive (from categories table) only.
        # products.age_restricted is intentionally ignored here; it was
        # confirmed to be a duplicate field on the current dataset.
        if c["is_age_sensitive"] and customer_age is not None and customer_age < 18:
            continue
        filtered.append(c)

    return filtered