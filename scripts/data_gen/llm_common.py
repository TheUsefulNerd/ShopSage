#!/usr/bin/env python3
"""Shared infrastructure for LLM-based enrichment of the synthetic warehouse
CSVs produced by generate.py.

This module has no CLI of its own — it's imported by enrich_descriptions.py
and enrich_reviews.py, which each add their own dataset-specific prompt and
merge logic on top of the generic driver defined here.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import threading
import time
from pathlib import Path

# Load variables from a .env file in the current directory (or a parent of
# it) if python-dotenv is installed, so GROQ_API_KEY/ANTHROPIC_API_KEY/
# OPENAI_API_KEY can just live in a .env file instead of needing to be
# exported manually in every shell session. Silently does nothing if
# python-dotenv isn't installed or no .env file is found.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def invoke_llm_with_retry(chain, input_data: dict, max_retries: int = 5, base_delay: float = 2.0) -> dict:
    """Call chain.invoke(input_data) and parse the JSON result, retrying with
    exponential backoff + jitter on ANY failure (rate limits, transient
    network/API errors, or the model returning non-JSON text). Raises the
    last error once retries are exhausted, so the caller can decide whether
    to skip that one row rather than crash the whole run."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            raw = chain.invoke(input_data)
            return json.loads(raw)
        except Exception as e:  # noqa: BLE001 - deliberately broad, see docstring
            last_exc = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def load_checkpoint(path: Path) -> dict:
    """Load a JSONL checkpoint file into {row_id: generated_fields}. Missing
    file or corrupt lines are tolerated (corrupt lines just get skipped —
    they represent a write that was interrupted mid-flush)."""
    data: dict = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    data[entry["_id"]] = entry
                except (json.JSONDecodeError, KeyError):
                    continue
    return data


class _RateLimiter:
    """Paces call starts to at least `min_interval` seconds apart, shared
    across all threads regardless of `concurrency`. This is a PROACTIVE
    throttle rather than reactive retry-after-429 — spacing calls out so you
    never approach the provider's tokens-per-minute ceiling in the first
    place, instead of bursting past it and eating repeated 429s + backoff
    delays that all count against the same per-minute window anyway."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait_for = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self.min_interval
        if wait_for > 0:
            time.sleep(wait_for)


def run_llm_enrichment_loop(
    rows: list,
    id_field: str,
    build_input_fn,
    chain,
    checkpoint_path: Path,
    limit: int | None = None,
    concurrency: int = 1,
    max_retries: int = 5,
    min_interval: float = 0.0,
    fresh: bool = False,
    verbose: bool = True,
) -> dict:
    """Generic driver shared by both description and review enrichment.

    - Resumable: progress is appended to `checkpoint_path` (JSONL) as each row
      succeeds, so a crash partway through a large run can be resumed by
      re-running the same command — already-checkpointed rows are skipped.
    - Retries transient failures (rate limits, malformed JSON) with backoff
      instead of letting one bad call kill the whole run; a row that still
      fails after `max_retries` is skipped (keeps its original templated
      text) rather than aborting everything else.
    - `concurrency > 1` runs calls in a thread pool (LLM calls are I/O-bound,
      so threads are sufficient here) with a lock around checkpoint writes.
    - `min_interval > 0` proactively paces call starts to at least that many
      seconds apart (shared across all threads), to stay under a provider's
      tokens-per-minute limit instead of bursting past it and relying on
      retries. Use this when you know your TPM budget and per-call token
      cost roughly — e.g. a 6000 TPM limit with ~350 tokens/call supports
      about 17 calls/minute, so min_interval=3.5 keeps you safely under that.

    Returns {row_id: generated_fields_dict} covering every successfully
    enriched row, including ones loaded from a prior run's checkpoint.
    """
    if fresh and checkpoint_path.exists():
        checkpoint_path.unlink()

    checkpoint = load_checkpoint(checkpoint_path)
    lock = threading.Lock()
    rate_limiter = _RateLimiter(min_interval)

    def row_id(row):
        return getattr(row, id_field)

    todo = []
    for row in rows:
        rid = row_id(row)
        if rid in checkpoint:
            continue
        todo.append(row)
        if limit is not None and (len(checkpoint) + len(todo)) >= limit:
            break

    if verbose:
        print(f"{len(checkpoint)} rows already checkpointed, {len(todo)} to process this run "
              f"(checkpoint: {checkpoint_path}).")

    def process(row) -> None:
        rid = row_id(row)
        input_data = build_input_fn(row)
        if verbose:
            print(f"--- Generating for {id_field}={rid} ---")
            print(json.dumps(input_data, indent=2, default=str))
        rate_limiter.wait()
        try:
            generated = invoke_llm_with_retry(chain, input_data, max_retries=max_retries)
        except Exception as e:
            print(f"WARNING: giving up on {id_field}={rid} after {max_retries} attempts ({e}); "
                  f"leaving its original text in place.")
            return
        entry = dict(generated)
        entry["_id"] = rid
        with lock:
            with checkpoint_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()
            checkpoint[rid] = entry
        if verbose:
            print(f"--- Generated for {id_field}={rid} ---")
            print(json.dumps(generated, indent=2))

    if concurrency <= 1:
        for row in todo:
            process(row)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(process, row) for row in todo]
            for f in concurrent.futures.as_completed(futures):
                f.result()  # re-raise anything unexpected (process() itself already catches LLM errors)

    return checkpoint


def build_llm_client(provider: str, model: str | None, api_key: str | None):
    """Construct a langchain chat model for the given provider. Imports the
    provider-specific package lazily, so you only need e.g. langchain-groq
    installed if you actually use --llm-provider groq."""
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        return ChatAnthropic(model=model or "claude-sonnet-4-6", api_key=key)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        key = api_key or os.environ.get("OPENAI_API_KEY")
        return ChatOpenAI(model=model or "gpt-4o-mini", api_key=key)
    elif provider == "groq":
        from langchain_groq import ChatGroq
        key = api_key or os.environ.get("GROQ_API_KEY")
        return ChatGroq(temperature=0, groq_api_key=key, model_name=model or "openai/gpt-oss-20b")
    else:
        raise ValueError(f"Unknown llm provider: {provider}")


def add_common_llm_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add the CLI flags shared by enrich_descriptions.py and
    enrich_reviews.py, so both scripts stay consistent without duplicating
    argparse definitions."""
    parser.add_argument("--llm-provider", choices=["anthropic", "openai", "groq"], default="groq",
                         help="Which langchain chat model provider to use.")
    parser.add_argument("--llm-model", default=None,
                         help="Model name for the provider (defaults: anthropic='claude-sonnet-4-6', "
                              "openai='gpt-4o-mini', groq='openai/gpt-oss-20b').")
    parser.add_argument("--llm-api-key", default=None,
                         help="API key for the chosen provider. Falls back to the provider's standard "
                              "env var if omitted (ANTHROPIC_API_KEY / OPENAI_API_KEY / GROQ_API_KEY).")
    parser.add_argument("--llm-quiet", action="store_true",
                         help="Suppress the per-row input/output print statements "
                              "(still prints a one-line summary at the end).")
    parser.add_argument("--llm-concurrency", type=int, default=1,
                         help="Number of concurrent LLM calls. Start small (e.g. 4-8) and raise it "
                              "based on your provider's rate limits.")
    parser.add_argument("--llm-max-retries", type=int, default=5,
                         help="Retries (with exponential backoff) per row before giving up and leaving "
                              "its original templated text in place.")
    parser.add_argument("--llm-min-interval", type=float, default=0.0,
                         help="Minimum seconds between call starts, shared across all concurrent workers. "
                              "Use this to proactively stay under a provider's tokens-per-minute limit "
                              "instead of bursting past it and relying on retries (e.g. a 6000 TPM limit "
                              "with ~350 tokens/call supports ~17 calls/min, so try --llm-min-interval 3.5).")
    parser.add_argument("--llm-fresh", action="store_true",
                         help="Ignore any existing enrichment checkpoint and start over from scratch, "
                              "instead of resuming from where a previous run left off.")