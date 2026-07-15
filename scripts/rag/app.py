#!/usr/bin/env python3
"""Gradio chat UI for ShopSage Week 1 RAG prototype.

Wires get_shortlist() into a Gradio ChatInterface.

Usage:
    python -m scripts.rag.app
    python -m scripts.rag.app --persist-path .chroma   # use a saved index
    python -m scripts.rag.app --share                   # get a public URL

The first run will:
  1. Connect to Supabase and embed all 500 products (takes ~1-2 min).
  2. Start the Gradio server.

Subsequent runs with --persist-path reuse the saved index and skip re-embedding.

Week 1 scope:
  - No tool calls (inventory/order tools are Week 2).
  - No conversation memory (stateless per-turn).
  - Stock, size, and color claims are NOT verified — RAG-grounded descriptions
    only. Add the guardrail note in the UI description so users know.
"""

from __future__ import annotations

import argparse

import gradio as gr

from scripts.rag.ingestion import build_collection
from scripts.rag.retrieval import get_shortlist


def build_chat_fn(collection):
    """Return a Gradio-compatible chat function bound to the given collection."""
    def chat_fn(message: str, history: list) -> str:
        answer, _ = get_shortlist(message, collection)
        return answer
    return chat_fn


def launch(persist_path: str | None = None, share: bool = False, debug: bool = False):
    """Build the index (or load from disk) and launch the Gradio UI."""
    collection, _ = build_collection(persist_path=persist_path)

    chat_fn = build_chat_fn(collection)

    demo = gr.ChatInterface(
        fn=chat_fn,
        title="ShopSage — Shopping Assistant (Week 1 Prototype)",
        description=(
            "RAG-grounded product search over the sample catalog. "
            "No live inventory/order tools yet — stock, size, and color claims "
            "are not verified in this build."
        ),
        examples=[
            "Show me smart plugs to control my home appliances remotely.",
            "I need hiking boots for a cold weather trip.",
            "Sleeping bag for camping under $150.",
            "What fitness gear do you have under $100?",
        ],
    )

    demo.launch(share=share, debug=debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShopSage Gradio chat UI.")
    parser.add_argument(
        "--persist-path",
        default=None,
        help="Load/save the Chroma index from this directory (e.g. .chroma). "
             "Omit to use an in-memory index (rebuilt every run).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Generate a public Gradio share link.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Gradio in debug mode (shows server logs).",
    )
    args = parser.parse_args()
    launch(persist_path=args.persist_path, share=args.share, debug=args.debug)
