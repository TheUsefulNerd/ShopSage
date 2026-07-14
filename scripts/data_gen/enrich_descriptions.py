#!/usr/bin/env python3
"""Standalone CLI: replace products.csv's templated `description` column with
LLM-generated copy. Reads/writes CSVs produced by generate.py — run that
first. This script needs langchain-core plus whichever provider package you
use (langchain-anthropic / langchain-openai / langchain-groq)."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_common import add_common_llm_cli_args, build_llm_client, run_llm_enrichment_loop


def build_description_enrichment_chain(llm):
    """Build the description-enrichment chain from a caller-supplied langchain
    chat model."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    description_enrichment_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are enriching a real e-commerce product record for a shopping assistant's catalog.
Given the product's real fields below, generate ONLY the missing fields, staying realistic
and consistent with the existing data (brand, category, gender, color, article type).

INPUT PRODUCT:
- Product ID: {product_id}
- Product Title: {title}

GENERATE THE FOLLOWING FIELDS:

1. "description" — a 2-3 sentence product description written in natural retail copy style.
   It MUST explicitly mention any relevant technical/functional attributes implied by the
   product name, season, and usage (e.g. if the name or season suggests rain/cold-weather
   use, explicitly use words like "waterproof," "insulated," or "cold-weather" rather than
   leaving them implied). Do not invent brand claims or certifications that aren't reasonable
   for the category.


Output ONLY valid JSON in this exact structure, no other text:
{{
  "description": ""
}}
"""
        ),
        ("human", "Generate missing fields for Product ID: {product_id}."),
    ])

    return description_enrichment_prompt | llm | StrOutputParser()


def enrich_descriptions_with_llm(
    out: Path,
    llm,
    limit: int | None = None,
    verbose: bool = True,
    concurrency: int = 1,
    max_retries: int = 5,
    min_interval: float = 0.0,
    fresh: bool = False,
) -> None:
    """Post-processing pass: replace the templated `description` column in
    products.csv with LLM-generated copy.

    Safe for large/full runs: progress is checkpointed to
    `<out>/.descriptions_llm_checkpoint.jsonl` as each product succeeds, so
    an interrupted run (rate limit, crash, timeout) can just be re-run and
    will pick up where it left off instead of starting over or losing prior
    work.

    `llm` must be a langchain chat model instance (e.g. ChatAnthropic(...),
    ChatOpenAI(...)) that the caller constructs and passes in — this function
    doesn't instantiate any provider client itself.
    """
    import pandas as pd

    products_csv_path = out / "products.csv"
    checkpoint_path = out / ".descriptions_llm_checkpoint.jsonl"

    products_df = pd.read_csv(products_csv_path)

    chain = build_description_enrichment_chain(llm)

    def build_input(row):
        return {"product_id": row.product_id, "title": row.title}

    checkpoint = run_llm_enrichment_loop(
        rows=list(products_df.itertuples(index=False)),
        id_field="product_id",
        build_input_fn=build_input,
        chain=chain,
        checkpoint_path=checkpoint_path,
        limit=limit,
        concurrency=concurrency,
        max_retries=max_retries,
        min_interval=min_interval,
        fresh=fresh,
        verbose=verbose,
    )

    description_enriched_df = pd.DataFrame(
        [{"product_id": pid, "description": entry.get("description")} for pid, entry in checkpoint.items()],
        columns=["product_id", "description"],
    )

    # Left merge so products not yet enriched keep their original templated
    # description instead of becoming blank.
    merged = pd.merge(
        products_df,
        description_enriched_df[["product_id", "description"]],
        on="product_id",
        how="left",
        suffixes=("_original", "_new"),
    )
    merged["description"] = merged["description_new"].combine_first(merged["description_original"])
    merged.drop(columns=["description_original", "description_new"], inplace=True)
    merged = merged[products_df.columns.tolist()]  # restore original column order

    merged.to_csv(products_csv_path, index=False)
    print(f"Wrote LLM-enriched descriptions for {len(description_enriched_df)}/{len(products_df)} products to {products_csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-enrich products.csv's description column.")
    parser.add_argument("--out", required=True, help="The warehouse output folder (same one passed to generate.py's --out).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only enrich the first N products (useful for testing before a full run).")
    add_common_llm_cli_args(parser)
    args = parser.parse_args()

    out = Path(args.out)
    llm = build_llm_client(args.llm_provider, args.llm_model, args.llm_api_key)

    enrich_descriptions_with_llm(
        out, llm,
        limit=args.limit,
        verbose=not args.llm_quiet,
        concurrency=args.llm_concurrency,
        max_retries=args.llm_max_retries,
        min_interval=args.llm_min_interval,
        fresh=args.llm_fresh,
    )


if __name__ == "__main__":
    main()