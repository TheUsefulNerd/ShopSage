#!/usr/bin/env python3
"""Standalone CLI: replace reviews.csv's templated `review_title` and
`review_body` columns with LLM-generated copy grounded in the actual
product title and star rating. Reads/writes CSVs produced by
generate.py — run that first. This script needs langchain-core plus
whichever provider package you use (langchain-anthropic / langchain-openai
/ langchain-groq)."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_common import add_common_llm_cli_args, build_llm_client, run_llm_enrichment_loop


def build_review_enrichment_chain(llm):
    """Build the review-enrichment chain from a caller-supplied langchain
    chat model. Generates a review_title and review body grounded in the
    actual product title and star rating."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    review_enrichment_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are generating a realistic product review, including a title and body, for an e-commerce product.
Given the product's details below, generate ONLY the missing fields, staying realistic
and consistent with the product title.

INPUT PRODUCT:
- Product ID: {product_id}
- Product Title: {title}
- Product Rating: {rating}

GENERATE THE FOLLOWING FIELDS:

1. "review_title" — a concise and engaging title for the review.
2. "review" — a 2-3 sentence review body that reflects a plausible customer experience with the product.

Output ONLY valid JSON in this exact structure, no other text:
{{ "review_title": "", "review": "" }}
"""
        ),
        ("human", "Generate a review title and review for Product ID: {product_id} with Product Title: {title} and Rating: {rating}."),
    ])

    return review_enrichment_prompt | llm | StrOutputParser()


def enrich_reviews_with_llm(
    out: Path,
    llm,
    limit: int | None = None,
    verbose: bool = True,
    concurrency: int = 1,
    max_retries: int = 5,
    min_interval: float = 0.0,
    fresh: bool = False,
) -> None:
    """Post-processing pass: replace the templated `review_title` and
    `review_body` columns in reviews.csv with LLM-generated copy that's
    grounded in the actual product title and star rating.

    Safe for large/full runs: progress is checkpointed to
    `<out>/.reviews_llm_checkpoint.jsonl` as each review succeeds, so an
    interrupted run (rate limit, crash, timeout) can just be re-run and will
    pick up where it left off instead of starting over or losing prior work.
    """
    import pandas as pd

    reviews_csv_path = out / "reviews.csv"
    products_csv_path = out / "products.csv"
    checkpoint_path = out / ".reviews_llm_checkpoint.jsonl"

    reviews_df = pd.read_csv(reviews_csv_path)
    products_df = pd.read_csv(products_csv_path)[["product_id", "title"]]

    review_product_merged = reviews_df.merge(products_df, on="product_id", how="left")

    chain = build_review_enrichment_chain(llm)

    def build_input(row):
        return {
            "product_id": row.product_id,
            "title": row.title,
            "rating": row.rating,
        }

    checkpoint = run_llm_enrichment_loop(
        rows=list(review_product_merged.itertuples(index=False)),
        id_field="review_id",
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

    generated_review_data_list = [
        {
            "review_id": rid,
            "review_title_generated": entry.get("review_title"),
            "review_body_generated": entry.get("review"),
        }
        for rid, entry in checkpoint.items()
    ]
    llm_generated_reviews_df = pd.DataFrame(
        generated_review_data_list,
        columns=["review_id", "review_title_generated", "review_body_generated"],
    )

    # Left merge onto the ORIGINAL reviews_df (not review_product_merged) so we
    # only touch review_title/review_body, and use combine_first so rows not
    # yet enriched keep their original templated text instead of going blank.
    merged = pd.merge(reviews_df, llm_generated_reviews_df, on="review_id", how="left")
    merged["review_title"] = merged["review_title_generated"].combine_first(merged["review_title"])
    merged["review_body"] = merged["review_body_generated"].combine_first(merged["review_body"])
    merged.drop(columns=["review_title_generated", "review_body_generated"], inplace=True)
    merged = merged[reviews_df.columns.tolist()]  # restore original column order

    merged.to_csv(reviews_csv_path, index=False)
    print(f"Wrote LLM-enriched reviews for {len(llm_generated_reviews_df)}/{len(reviews_df)} reviews to {reviews_csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-enrich reviews.csv's review_title/review_body columns.")
    parser.add_argument("--out", required=True, help="The warehouse output folder (same one passed to generate.py's --out).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only enrich the first N reviews (useful for testing before a full run).")
    add_common_llm_cli_args(parser)
    args = parser.parse_args()

    out = Path(args.out)
    llm = build_llm_client(args.llm_provider, args.llm_model, args.llm_api_key)

    enrich_reviews_with_llm(
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