#!/usr/bin/env python3
"""Ingestion pipeline — pulls products from Supabase, embeds them, and loads
into a Chroma vector store.

What this module does:
  1. Loads products joined with brands + categories from Supabase (paginated).
  2. Loads variant colors from product_variants (paginated), keyed by product_id.
  3. Flattens nested brand/category dicts into a flat DataFrame.
  4. Builds a chunk_text string per product (title, brand, category, colors,
     description). base_price/rating_avg are intentionally NOT embedded or
     stored in Chroma metadata — they change independently of the catalog
     (pricing updates, new reviews) and would go stale until the next
     --force-rebuild otherwise. A live tool call (get_product_details, in
     tools.py) fetches them fresh instead, called on final candidates only.
  5. Encodes all chunks with all-MiniLM-L6-v2 (normalize_embeddings=True) and
     upserts them into a Chroma collection using cosine distance.

If persist_path is given and a populated collection already exists there,
build_collection() reuses it instead of re-pulling from Supabase and
re-embedding everything — pass force_rebuild=True (or --force-rebuild on the
CLI) to rebuild anyway, e.g. after catalog data has changed.

Usage:
    from scripts.rag.ingestion import build_collection
    collection, products_df = build_collection(persist_path=".chroma")

Or run standalone:
    python -m scripts.rag.ingestion --persist-path .chroma
    python -m scripts.rag.ingestion --persist-path .chroma --force-rebuild
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import chromadb
import pandas as pd
from supabase import create_client, Client
from scripts.rag.retrieval import _get_embedder

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
EMBED_BATCH_SIZE = 100

# Column name for color on product_variants — verified against live schema.
# If this throws a KeyError on load, check your Supabase schema and update it.
VARIANT_COLOR_COLUMN = "color_name"


# ---------------------------------------------------------------------------
# Supabase data loading
# ---------------------------------------------------------------------------

def _get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise EnvironmentError(
            "Missing SUPABASE_URL and/or SUPABASE_KEY. "
            "Set both in your .env file or as environment variables."
        )
    return create_client(url, key)


def load_products(supabase: Client, page_size: int = 1000) -> list[dict]:
    """Pulls all products joined with brand and category info (paginated)."""
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
    """Returns {product_id: [distinct variant color strings]}.

    color_family on products is the BASE color only. Joining product_variants
    lets queries like "green yoga mat" surface a product whose base color is
    black but has a green variant — retrieval matching only, NOT a claim that
    the item is currently in stock in green (that needs a live tool call).
    """
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
    """Flatten the nested brand/category dicts returned by Supabase into a
    single flat dict, and attach the variant colors list."""
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
        # age_restricted from products is intentionally kept in the DataFrame
        # for reference but is NOT used for filtering — confirmed duplicate of
        # categories.is_age_sensitive on current data (nunique()==1 per category).
        # See notebook for the evidence. Filtering uses is_age_sensitive only.
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
    """Full pipeline: load raw rows → flatten → return products DataFrame."""
    if supabase is None:
        supabase = _get_supabase_client()

    print("Loading products from Supabase...")
    raw_products = load_products(supabase)
    print(f"  Loaded {len(raw_products)} products")

    print("Loading variant colors from Supabase...")
    variant_colors = load_variant_colors(supabase)
    print(f"  Loaded variant colors for {len(variant_colors)} products")

    products_df = pd.DataFrame(
        [flatten_product(r, variant_colors) for r in raw_products]
    )
    return products_df


# ---------------------------------------------------------------------------
# Chunk text + embedding
# ---------------------------------------------------------------------------

def build_chunk_text(row: pd.Series) -> str:
    """Build the text string that gets embedded for a product.

    Design decisions:
    - base_price/rating_avg are deliberately excluded — they change
      independently of the catalog and would go stale until the next
      --force-rebuild. They're fetched live instead (get_product_details,
      tools.py), never baked into the embedded text or Chroma metadata.
    - variant_colors are included so semantic queries like "green yoga mat"
      can surface products that don't have green as their base color_family.
      This is retrieval-matching only — it never reaches the LLM as a claim
      (see app.py's chat_fn, which builds the candidate_block text that's
      actually shown to the LLM from live-fetched data, not this).
    - material is included for the same retrieval-matching reason as colors —
      lets queries like "leather boots" or "cotton shirt" match on material,
      even though material is essentially static (unlike price/stock) so
      there's no live-fetch staleness concern here either way.
    """
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


def build_collection(
    supabase: Optional[Client] = None,
    chroma_client: Optional[chromadb.Client] = None,
    persist_path: Optional[str] = None,
    force_rebuild: bool = False,
) -> tuple[chromadb.Collection, Optional[pd.DataFrame]]:
    """Build (or load) the Chroma vector store.

    Args:
        supabase: Optional pre-built Supabase client (uses env vars if None).
        chroma_client: Optional pre-built Chroma client. If None:
            - Uses PersistentClient at persist_path if persist_path is given.
            - Falls back to in-memory Client (resets on restart) otherwise.
        persist_path: Path for a PersistentClient (e.g. ".chroma"). Ignored
            if chroma_client is supplied directly.
        force_rebuild: If True, always re-pull from Supabase and re-embed,
            even if a populated collection already exists at persist_path.
            Use this after catalog data has actually changed.

    Returns:
        (collection, products_df) — products_df is None when an existing
        collection was reused (no DataFrame was built in that case).
    """
    if chroma_client is None:
        if persist_path:
            chroma_client = chromadb.PersistentClient(path=persist_path)
            print(f"Using persistent Chroma at: {persist_path}")
        else:
            chroma_client = chromadb.Client()
            print("Using in-memory Chroma (index resets on restart).")

    # Skip the expensive rebuild (Supabase pull + re-embedding every product)
    # if a populated collection already exists — this is what actually makes
    # persist_path worth using. Without this check, build_collection()
    # unconditionally deleted and rebuilt the collection every single call,
    # even against a persistent path, defeating the point of persisting it.
    if not force_rebuild:
        try:
            existing = chroma_client.get_collection(COLLECTION_NAME)
            if existing.count() > 0:
                print(
                    f"Found existing collection '{COLLECTION_NAME}' with "
                    f"{existing.count()} items — reusing it, skipping rebuild. "
                    f"Pass force_rebuild=True to rebuild anyway."
                )
                return existing, None
        except Exception:
            pass  # collection doesn't exist yet — fall through and build it

    products_df = load_products_df(supabase)
    products_df["chunk_text"] = products_df.apply(build_chunk_text, axis=1)

    # Drop and recreate so re-running doesn't hit duplicate-collection errors
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # cosine distance — MiniLM vectors aren't unit-normalized by default,
    # so we normalize on encode() below and use cosine here for consistency.
    collection = chroma_client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    embedder = _get_embedder()

    total = len(products_df)
    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = products_df.iloc[i : i + EMBED_BATCH_SIZE]
        embeddings = embedder.embed_documents(batch["chunk_text"].tolist())
        collection.add(
            ids=batch["product_id"].tolist(),
            embeddings=embeddings,
            documents=batch["chunk_text"].tolist(),
            metadatas=batch[
                [
                    # base_price and rating_avg deliberately excluded — they
                    # change independently of the catalog (pricing updates,
                    # new reviews) and would go stale until the next
                    # --force-rebuild otherwise. Fetched live instead, via
                    # get_product_details (tools.py), on final candidates only.
                    "product_id", "title", "brand_name",
                    "category_name", "color_family", "age_restricted",
                    "min_age", "is_age_sensitive",
                ]
            ].to_dict(orient="records"),
        )
        print(f"  Embedded {min(i + EMBED_BATCH_SIZE, total)}/{total} products")

    print(f"Done — indexed {collection.count()} chunks into '{COLLECTION_NAME}'.")
    return collection, products_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the ShopSage Chroma index from Supabase.")
    parser.add_argument(
        "--persist-path",
        default=".chroma",
        help="Directory for a persistent Chroma store. Defaults to .chroma. "
             "Automatically reuses an existing populated index there if one "
             "exists — pass --force-rebuild to rebuild anyway.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild even if a populated collection already exists at --persist-path.",
    )
    args = parser.parse_args()

    collection, df = build_collection(persist_path=args.persist_path, force_rebuild=args.force_rebuild)
    if df is not None:
        print(f"\nproducts_df shape: {df.shape}")
        print(df[["product_id", "title", "category_name", "base_price"]].head())
    else:
        print(f"\nReused existing collection — {collection.count()} items, no DataFrame built.")