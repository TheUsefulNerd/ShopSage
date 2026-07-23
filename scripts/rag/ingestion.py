#!/usr/bin/env python3
"""Ingestion pipeline — pulls products from Supabase, embeds them, and upserts
into a Qdrant Cloud collection with HYBRID search support (dense + sparse BM25).

What this module does:
  1. Loads products joined with brands + categories from Supabase (paginated).
  2. Loads variant colors from product_variants (paginated), keyed by product_id.
  3. Flattens nested brand/category dicts into a flat DataFrame.
  4. Builds a chunk_text string per product.
  5. Encodes all chunks with:
       - Dense:  all-MiniLM-L6-v2 via FastEmbed (normalize_embeddings=True)
       - Sparse: BM25 via FastEmbed (for exact keyword/brand/color matching)
     Upserts both into a Qdrant collection configured for hybrid search.

Usage:
    from scripts.rag.ingestion import build_collection
    build_collection()

Or run standalone:
    python -m scripts.rag.ingestion --force-rebuild
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
)
from fastembed import TextEmbedding, SparseTextEmbedding
from supabase import create_client, Client

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COLLECTION_NAME = "shopsage_products"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"
DENSE_DIM = 384
EMBED_BATCH_SIZE = 64
VARIANT_COLOR_COLUMN = "color_name"

# ---------------------------------------------------------------------------
# Clients (lazy singletons)
# ---------------------------------------------------------------------------

_qdrant_client: Optional[QdrantClient] = None
_dense_model: Optional[TextEmbedding] = None
_sparse_model: Optional[SparseTextEmbedding] = None


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        url = os.environ.get("QDRANT_URL")
        api_key = os.environ.get("QDRANT_API")
        if not url or not api_key:
            raise EnvironmentError("Missing QDRANT_URL and/or QDRANT_API in .env")
        _qdrant_client = QdrantClient(url=url, api_key=api_key)
        print(f"Connected to Qdrant Cloud: {url}")
    return _qdrant_client


def get_qdrant_client() -> QdrantClient:
    return _get_qdrant_client()


def _get_dense_model() -> TextEmbedding:
    global _dense_model
    if _dense_model is None:
        print(f"Loading dense embedding model: {DENSE_MODEL}...")
        _dense_model = TextEmbedding(model_name=DENSE_MODEL)
        print("Dense model ready.")
    return _dense_model


def get_dense_model() -> TextEmbedding:
    return _get_dense_model()


def _get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        print(f"Loading sparse (BM25) model: {SPARSE_MODEL}...")
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
        print("Sparse model ready.")
    return _sparse_model


def get_sparse_model() -> SparseTextEmbedding:
    return _get_sparse_model()


# ---------------------------------------------------------------------------
# Supabase data loading
# ---------------------------------------------------------------------------

def _get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise EnvironmentError("Missing SUPABASE_URL and/or SUPABASE_KEY.")
    return create_client(url, key)


def load_products(supabase: Client, page_size: int = 1000) -> list[dict]:
    all_rows: list[dict] = []
    start = 0
    while True:
        resp = (
            supabase.table("products")
            .select(
                "product_id, title, description, base_price, rating_avg, "
                "review_count, age_restricted, min_age, color_family, material, "
                "brands(brand_name, tier), "
                "categories(category_name, is_age_sensitive)"
            )
            .range(start, start + page_size - 1)
            .execute()
        )
        rows = resp.data
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        start += page_size
    return all_rows


def load_variant_colors(supabase: Client, page_size: int = 1000) -> dict[str, list[str]]:
    all_rows: list[dict] = []
    start = 0
    while True:
        resp = (
            supabase.table("product_variants")
            .select(f"product_id, {VARIANT_COLOR_COLUMN}")
            .range(start, start + page_size - 1)
            .execute()
        )
        rows = resp.data
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        start += page_size

    if not all_rows:
        return {}
    variants_df = pd.DataFrame(all_rows)
    return (
        variants_df.dropna(subset=[VARIANT_COLOR_COLUMN])
        .groupby("product_id")[VARIANT_COLOR_COLUMN]
        .apply(lambda s: sorted(set(s)))
        .to_dict()
    )


def flatten_product(row: dict, variant_colors: dict[str, list[str]]) -> dict:
    brand = row.get("brands") or {}
    category = row.get("categories") or {}
    pid = row["product_id"]
    return {
        "product_id": pid,
        "title": row["title"],
        "description": row["description"],
        "base_price": row["base_price"],
        "rating_avg": row["rating_avg"],
        "review_count": row["review_count"],
        "age_restricted": row["age_restricted"],
        "min_age": row["min_age"],
        "color_family": row["color_family"],
        "material": row.get("material"),
        "brand_name": brand.get("brand_name"),
        "brand_tier": brand.get("tier"),
        "category_name": category.get("category_name"),
        "is_age_sensitive": category.get("is_age_sensitive"),
        "variant_colors": variant_colors.get(pid, []),
    }


def load_products_df(supabase: Optional[Client] = None) -> pd.DataFrame:
    if supabase is None:
        supabase = _get_supabase_client()
    print("Loading products from Supabase...")
    raw_products = load_products(supabase)
    print(f"  Loaded {len(raw_products)} products")
    print("Loading variant colors from Supabase...")
    variant_colors = load_variant_colors(supabase)
    print(f"  Loaded variant colors for {len(variant_colors)} products")
    return pd.DataFrame([flatten_product(r, variant_colors) for r in raw_products])


# ---------------------------------------------------------------------------
# Chunk text builder
# ---------------------------------------------------------------------------

def build_chunk_text(row: pd.Series) -> str:
    colors = row["variant_colors"] or (
        [row["color_family"]] if row["color_family"] else []
    )
    color_text = f"Available colors: {', '.join(colors)}. " if colors else ""
    material_text = f"Material: {row['material']}. " if row.get("material") else ""
    return (
        f"{row['title']} by {row['brand_name']} ({row['brand_tier']} tier). "
        f"Category: {row['category_name']}. {color_text}{material_text}"
        f"{row['description']}"
    )


# ---------------------------------------------------------------------------
# Qdrant collection builder
# ---------------------------------------------------------------------------

def build_collection(
    supabase: Optional[Client] = None,
    force_rebuild: bool = False,
) -> str:
    """Build (or reuse) the Qdrant Cloud collection with dense + sparse vectors."""
    client = _get_qdrant_client()

    if not force_rebuild:
        try:
            info = client.get_collection(COLLECTION_NAME)
            count = info.points_count
            if count and count > 0:
                print(
                    f"Reusing existing Qdrant collection '{COLLECTION_NAME}' "
                    f"({count} points). Pass force_rebuild=True to rebuild."
                )
                return COLLECTION_NAME
        except Exception:
            pass

    print(f"Building Qdrant collection '{COLLECTION_NAME}'...")
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            ),
        },
    )
    print(f"  Created collection '{COLLECTION_NAME}' with dense + sparse vectors")

    products_df = load_products_df(supabase)
    products_df["chunk_text"] = products_df.apply(build_chunk_text, axis=1)
    texts = products_df["chunk_text"].tolist()

    dense_model = _get_dense_model()
    sparse_model = _get_sparse_model()

    print(f"  Encoding {len(texts)} products (dense + sparse)...")
    dense_embeddings = list(dense_model.embed(texts, batch_size=EMBED_BATCH_SIZE))
    sparse_embeddings = list(sparse_model.embed(texts, batch_size=EMBED_BATCH_SIZE))

    points = []
    for i, (_, row) in enumerate(products_df.iterrows()):
        sparse_vec = sparse_embeddings[i]
        points.append(PointStruct(
            id=i,
            vector={
                "dense": dense_embeddings[i].tolist(),
                "sparse": SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
            },
            payload={
                "product_id": row["product_id"],
                "title": row["title"],
                "brand_name": row["brand_name"],
                "category_name": row["category_name"],
                "color_family": row["color_family"],
                "is_age_sensitive": row["is_age_sensitive"],
                "age_restricted": row["age_restricted"],
                "min_age": row["min_age"],
                "chunk_text": row["chunk_text"],
            },
        ))

    for i in range(0, len(points), EMBED_BATCH_SIZE):
        batch = points[i: i + EMBED_BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"  Upserted {min(i + EMBED_BATCH_SIZE, len(points))}/{len(points)} points")

    total = client.get_collection(COLLECTION_NAME).points_count
    print(f"Done — {total} points in '{COLLECTION_NAME}' (dense + sparse hybrid ready).")
    return COLLECTION_NAME


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()
    build_collection(force_rebuild=args.force_rebuild)