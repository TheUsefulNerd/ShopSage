#!/usr/bin/env python3
"""Retrieval logic for ShopSage — Qdrant hybrid search (dense + sparse BM25).

No Groq/LLM code lives here. All LLM calls are in streamlit_app.py.

Public API:
  - retrieve(query, top_k, score_threshold) -> list of candidate dicts
  - build_candidate_block(query, top_k, customer_age) -> age-filtered candidates

Hybrid search design:
  Each query is encoded into both a dense vector (semantic meaning) and a sparse
  BM25 vector (exact keyword matching) using FastEmbed. Qdrant fuses both result
  lists server-side using Reciprocal Rank Fusion (RRF) — a single query_points()
  call handles it.

Age filtering:
  products.age_restricted is a duplicate of categories.is_age_sensitive.
  Filtering uses is_age_sensitive only.
"""

from __future__ import annotations

from typing import Optional

from qdrant_client.models import (
    Prefetch,
    FusionQuery,
    Fusion,
    SparseVector,
)

from scripts.rag.ingestion import (
    COLLECTION_NAME,
    get_qdrant_client,
    get_dense_model,
    get_sparse_model,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCORE_THRESHOLD: Optional[float] = 0.02


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_k: int = 8,
    score_threshold: Optional[float] = SCORE_THRESHOLD,
) -> list[dict]:
    client = get_qdrant_client()
    dense_model = get_dense_model()
    sparse_model = get_sparse_model()

    dense_vec = list(dense_model.embed([query]))[0].tolist()
    sparse_result = list(sparse_model.embed([query]))[0]
    sparse_vec = SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist(),
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=top_k * 2),
            Prefetch(query=sparse_vec, using="sparse", limit=top_k * 2),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
    )

    hits = []
    for point in results.points:
        payload = point.payload or {}
        hits.append({**payload, "score": point.score})
    return hits


# ---------------------------------------------------------------------------
# Candidate finder (age-filtered)
# ---------------------------------------------------------------------------

def build_candidate_block(
    query: str,
    top_k: int = 8,
    customer_age: Optional[int] = None,
) -> list[dict]:
    candidates = retrieve(query, top_k=top_k)
    if customer_age is None:
        return candidates
    return [
        c for c in candidates
        if not (c.get("is_age_sensitive") and customer_age < 18)
    ]