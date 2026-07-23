#!/usr/bin/env python3
"""Gradio chat UI for ShopSage Week 1 RAG prototype.

chat_fn (built in build_chat_fn) classifies every message via
extract_user_intent() BEFORE deciding what to do. RAG (build_candidate_block)
only runs for "new_search" — this covers both fresh searches and refinements
like "under $250". "follow_up" (e.g. "is it available in red?") and
"order_status" (e.g. "where's my order?") do NOT call RAG at all right now;
track_order isn't wired in yet, so these branches just return a placeholder
message (get_product_details, which replaces what check_inventory was meant
to do, IS wired in — see below).

For "new_search", every age-filtered candidate is deterministically checked
and filtered — no LLM/tool-calling judgment involved, since "never recommend
an out-of-stock/over-budget/wrong-color item" has no exceptions:
  1. Live stock-check via get_product_details (tools.py) — out-of-stock
     candidates are dropped, backfilled from the larger candidate pool.
  2. Budget filter — hard exclude on live base_price if min/max_budget stated.
  3. Color/size filter — only keeps candidates with an ACTUAL IN-STOCK VARIANT
     matching what was asked; relaxes back to the broader set (rather than
     showing nothing) if no exact match exists, so real alternatives are
     still shown.
  4. A deterministic mismatch_note (plain Python text, not LLM-dependent) is
     prepended to the final answer whenever a stated color/size/budget
     constraint couldn't be exactly satisfied — this fact reaches the user
     regardless of whether the LLM remembers to mention it.
  5. If NOTHING survives all filtering, chat_fn returns a fallback message
     directly and skips the Groq narration call entirely.

All Groq/LLM code lives here — retrieval.py has zero Groq dependency (see
that file's module docstring), so this is the only file that needs a
GROQ_API_KEY.

Usage:
    python -m scripts.rag.app
    python -m scripts.rag.app --persist-path .chroma   # use a saved index
    python -m scripts.rag.app --share                   # get a public URL

The first run will:
  1. Connect to Supabase and embed all 500 products (takes ~1-2 min).
  2. Start the Gradio server.

Subsequent runs with --persist-path reuse the saved index and skip re-embedding.

Cost/latency note: every "new_search" message costs 1 Groq call (intent
extraction) + 1 Groq call (narration) + up to top_k Supabase calls (one
get_product_details lookup per age-filtered candidate, before the top-3
slice). "follow_up"/"order_status" messages cost only the intent-extraction
call, since they short-circuit before reaching RAG.

Current scope / known gaps:
  - "order_status" is now wired in (track_order.invoke, tools.py) — if
    intent.order_id is known, invokes it; if not, asks the user for it.
  - "follow_up" (e.g. "is it available in red?" about an ALREADY-SHOWN
    product) is not wired in yet either (intent.referenced_product_id is
    extracted but not acted on) — also returns a placeholder. Note this is
    different from the color/size filtering above, which applies to
    "new_search" results, not follow-ups about a specific known product.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import gradio as gr
from groq import Groq
from pydantic import BaseModel, ValidationError

from scripts.rag.ingestion import build_collection
from scripts.rag.retrieval import build_candidate_block, _get_embedder
from scripts.rag.tools import (
    get_product_details,
    get_product_details_bulk,
    track_order,
    get_customer_by_email,
    get_recent_orders,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Groq config + client (moved here from retrieval.py — see that file's
# module docstring for why)
# ---------------------------------------------------------------------------

GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are ShopSage, a shopping assistant. You are given a user query and a "
    "list of candidate products retrieved from the catalog (age-restricted "
    "items are excluded when the customer's age is known). All candidates "
    "given to you have ALREADY been verified to satisfy any stated budget, "
    "color, size, and rating constraints — you do not need to re-check or "
    "second-guess this; it's guaranteed. "
    "Answer using only the candidates given — never invent products or attributes "
    "not present in the candidate list. If the candidate list is empty, say so plainly.\n\n"
    "MANDATORY FORMAT — every single candidate MUST be listed on its own "
    "bullet, in EXACTLY this format, with NO field skipped:\n"
    "<Product name>, $<price>, Rating: <rating>/5, In stock: <availability>\n\n"
    "Worked example — given these two candidates:\n"
    "**Osprey Hiking Boots**\n"
    "Price: $207.69\n"
    "Rating: 4.2/5\n"
    "In stock: Size 8: Black, White\n\n"
    "**Salomon Hiking Boots**\n"
    "Price: $246.34\n"
    "Rating: not rated yet\n"
    "In stock: Size 8: Black; Size 9: Blue\n\n"
    "...your response's bullet list must look EXACTLY like this — note "
    "BOTH candidates show their own rating, not just one of them:\n"
    "- Osprey Hiking Boots, $207.69, Rating: 4.2/5, In stock: Size 8: Black, White\n"
    "- Salomon Hiking Boots, $246.34, Rating: not rated yet, In stock: Size 8: Black; Size 9: Blue\n\n"
    "A rating mentioned for only one candidate, or missing from any bullet, "
    "is WRONG — every bullet needs its own rating, even when it says "
    "\"not rated yet\". Preserve the size-to-color grouping in "
    "<availability> exactly as given — do NOT flatten \"Size 9: Black; "
    "Size 10: Blue\" into \"available in Black and Blue\", which loses "
    "which color belongs to which size. Do not omit any candidate, and do "
    "not merge multiple products into one sentence.\n\n"
    "If the user asked about a specific color/size/budget/rating, you may "
    "naturally acknowledge it in your own words — but the MANDATORY FORMAT "
    "above is not optional and takes priority over phrasing style.\n\n"
    "After the list, end with a short follow-up question inviting the user "
    "to narrow down or compare the options."
)

_groq_client: Optional[Groq] = None


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
# User intent + attribute extraction
#
# Fixes the bug where a follow-up like "I need under 250" gets searched
# against Chroma in isolation, losing all context about what's actually
# being discussed (e.g. hiking boots) — search_query is a reconstructed,
# standalone query that folds in relevant context from history.
#
# Also classifies intent so chat_fn can skip RAG entirely for follow_up/
# order_status (see build_chat_fn) — these don't need a fresh product
# search, they need a DB lookup on something already known.
#
# Known limitation: distinguishing "follow_up" (about an already-shown
# product) from "new_search" (a refinement that still needs RAG) is
# genuinely ambiguous in natural language — there's no wording that makes
# this 100% reliable. The prompt is written to fail toward "new_search"
# when uncertain, since that degrades to "slightly wasteful extra RAG call"
# rather than a broken/nonsensical response.
# ---------------------------------------------------------------------------

class UserIntent(BaseModel):
    # A standalone, context-complete version of the user's request — e.g.
    # "under 250" after "hiking boots for cold weather" becomes
    # "hiking boots for cold weather under $250". Not used for "follow_up"
    # or "order_status" intents.
    search_query: str
    # "new_search": a genuinely new product request, OR a refinement of the
    #               current search thread (budget/color/etc.) — either way,
    #               still needs a fresh RAG call via search_query.
    # "follow_up": a question about ONE SPECIFIC product already shown (e.g.
    #              "is it available in red?", "do you have it in stock?") —
    #              no RAG needed; would route to get_product_details (using
    #              referenced_product_id) once wired.
    # "order_status": asking about an existing order — NOT a product search;
    #                 would route to track_order once wired (not yet built).
    intent: str = "new_search"
    color: Optional[str] = None
    size: Optional[str] = None
    min_budget: Optional[float] = None
    max_budget: Optional[float] = None
    # Deterministic threshold, like budget — "rating higher than 4",
    # "rated above 4 stars", "best rated" all map here. Never treated as a
    # soft preference; a candidate below this is simply excluded.
    min_rating: Optional[float] = None
    recipient_age: Optional[int] = None
    order_id: Optional[str] = None
    # For "order_status" questions about the customer's ORDER HISTORY
    # (e.g. "my last order", "my last 3 orders", "my last few orders") —
    # rather than ONE specific order_id. No login/session system exists,
    # so email is the only way to identify which customer is asking.
    wants_order_history: bool = False
    customer_email: Optional[str] = None
    # None = no explicit count given (e.g. "my last few orders") — chat_fn
    # applies a deterministic default (5) rather than trusting the model to
    # guess a number for "a few". 1 = singular ("my last order").
    order_history_limit: Optional[int] = None
    # Which previously-shown product a "follow_up" refers to, chosen from
    # the REAL product_ids in known_products — never invented.
    referenced_product_id: Optional[str] = None


def _build_intent_extraction_prompt(known_products: dict[str, dict]) -> str:
    """known_products: {product_id: {"title": ..., "base_price": ...,
    "sizes_shown": [...], "colors_shown": [...]}}, accumulated across the
    session (every product shown so far, not just the most recent search).

    Enriched with price, sizes, AND colors (not just title) so the model can
    resolve references like "the $207 one" or "the size 8 Osprey" or "the
    red one" by DIRECT lookup against this structured list — evidence
    showed the model reliably extracts attributes like price/size/color
    from a message, but was NOT reliably cross-referencing them against its
    own unstructured prior narration text to find the matching product_id.
    Giving it the needed attributes directly, right here, removes that
    unreliable extra hop. Color was added after finding a concrete case
    where two candidates both had "Red" available and the model picked the
    wrong one (defaulted to whichever was shown last), since color wasn't
    part of this structured list before.
    """
    if known_products:
        known_list = "\n".join(
            f'  "{pid}": "{info["title"]}", ${info["base_price"]}, '
            f'sizes: {", ".join(info["sizes_shown"]) if info.get("sizes_shown") else "n/a"}, '
            f'colors: {", ".join(info.get("colors_shown") or []) or "n/a"}'
            for pid, info in known_products.items()
        )
        known_section = (
            f"Products shown so far in this conversation "
            f"(product_id: title, price, sizes, colors):\n{known_list}\n\n"
        )
    else:
        known_section = "No products have been shown yet in this conversation.\n\n"

    return (
        f"{known_section}"
        "You are analyzing a shopping conversation to prepare for a product "
        "search, order-status lookup, or inventory check. Given the "
        "conversation so far and the user's LATEST message, extract JSON "
        "with exactly these fields: "
        '{"search_query": string, "intent": "new_search" or "follow_up" or '
        '"order_status", "color": string or null, "size": string or null, '
        '"min_budget": number or '
        'null, "max_budget": number or null, "min_rating": number or null, '
        '"recipient_age": integer or '
        'null, "order_id": string or null, "wants_order_history": true or '
        'false, "customer_email": string or null, "order_history_limit": '
        'integer or null, "referenced_product_id": string '
        'or null}. '
        "\n\n"
        "search_query is the MOST IMPORTANT field for \"new_search\": it "
        "must be a standalone, self-contained product search query that "
        "makes sense with NO other context. If the latest message narrows/"
        "refines something already being discussed (e.g. \"under $250\", "
        "\"in red\", \"something cheaper\" after \"hiking boots\"), COMBINE "
        "them — e.g. \"hiking boots under $250\". For \"follow_up\" or "
        "\"order_status\" intent, search_query can just restate the "
        "request as-is. "
        "\n\n"
        "intent is \"order_status\" if the message is about an EXISTING "
        "order — tracking, delivery/shipping status, \"where's my order\", "
        "etc. — NOT a product search. Set wants_order_history=true if the "
        "message asks about the customer's order HISTORY rather than one "
        "specific order_id — e.g. \"my last order\", \"my last 3 orders\", "
        "\"my last few orders\", \"my recent orders\". If a specific number "
        "is stated (\"last 3\"), set order_history_limit to that number. If "
        "singular (\"my last order\"), set order_history_limit=1. If vague "
        "(\"my last few orders\", \"my recent orders\", no number given), "
        "leave order_history_limit as null — do NOT guess a number "
        "yourself. If the message contains an email address, set "
        "customer_email to it (needed to identify which customer's orders "
        "to look up, since there's no login system). "
        "intent is \"follow_up\" ONLY if the message clearly asks about ONE "
        "of the products listed above (e.g. \"is it available in red?\", "
        "\"do you have it in size 9?\", \"do you have it in stock?\", "
        "\"is the $207 one in stock?\", \"the cheaper Osprey one\", "
        "\"the size 8 one\", "
        "size/color/stock questions with no "
        "new product description). If so, set referenced_product_id to the "
        "matching product_id from the list above — MATCH DIRECTLY against "
        "the title, price, sizes, AND colors shown for each product_id "
        "above (e.g. if the message names a price, size, color, brand, or "
        "price threshold, find the product_id whose title/price/sizes/"
        "colors actually satisfies it "
        "— don't just guess based on recency). Never invent a product_id "
        "that isn't in the list; if you can't confidently match one, use "
        "\"new_search\" instead. "
        "Otherwise intent is \"new_search\" — this covers new requests and "
        "refinements to the current search thread. When genuinely unsure "
        "between \"follow_up\" and \"new_search\", prefer \"new_search\". "
        "\n\n"
        "min_rating is a rating threshold, ONLY if an explicit number is "
        "stated — e.g. \"rating higher than 4\", \"rated above 4 stars\" -> "
        "min_rating=4. Ratings are out of 5. If the user says something "
        "vague like \"best rated\" or \"highly rated\" with NO number, leave "
        "min_rating as null — do not invent a threshold. "
        "\n\n"
        "recipient_age is the age of whoever the product is FOR, not "
        "necessarily the person writing the message (e.g. a gift for a "
        "child). order_id is the order identifier if the user stated one "
        "(e.g. \"O00000123\"), otherwise null — do not guess or invent one. "
        "Only include color/size/budget/rating/age/order_id values explicitly "
        "stated or clearly implied — use null when unsure. "
        "Respond with ONLY the JSON object, no other text."
    )


def extract_user_intent(
    message: str,
    history: list,
    known_products: Optional[dict[str, dict]] = None,
) -> UserIntent:
    """One Groq call: reconstructs a context-complete search_query and
    extracts color/budget/age/intent/order_id/referenced_product_id.

    Falls back to UserIntent(search_query=message) if extraction/parsing
    fails, so a bad model response degrades to plain-message behavior
    (treated as new_search) rather than crashing.
    """
    known_products = known_products or {}

    messages = [{"role": "system", "content": _build_intent_extraction_prompt(known_products)}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    groq = _get_groq_client()
    completion = groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = completion.choices[0].message.content
    try:
        data = json.loads(raw)
        intent = UserIntent(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"[extract_user_intent] Failed to parse '{raw}': {e}")
        return UserIntent(search_query=message)

    # Defensive check 1: never trust a referenced_product_id the model
    # invented that isn't actually in known_products, even if it claimed one.
    if intent.referenced_product_id is not None and intent.referenced_product_id not in known_products:
        print(
            f"[extract_user_intent] Model referenced unknown "
            f"product_id {intent.referenced_product_id!r} — discarding it."
        )
        intent.referenced_product_id = None
        if intent.intent == "follow_up":
            intent.intent = "new_search"
            # Always overwrite (not just "if empty") — a downgrade means the
            # model's search_query was built under the WRONG assumption
            # (follow_up doesn't need reconstruction), so even a non-empty
            # value here is untrustworthy, not just a missing one.
            intent.search_query = message

    # Defensive check 2: "follow_up" with NO referenced_product_id at all is
    # just as invalid as an invented one — per the prompt's own instruction
    # ("if you can't confidently match one, use new_search instead"), the
    # model should never produce this combination, but it sometimes does
    # anyway (observed: intent='follow_up', referenced_product_id=None for
    # "in black"). Check 1 above only catches an INVENTED id that's wrong;
    # this catches the model never resolving one in the first place.
    if intent.intent == "follow_up" and intent.referenced_product_id is None:
        print(
            f"[extract_user_intent] Model said 'follow_up' but never "
            f"resolved a referenced_product_id — downgrading to new_search."
        )
        intent.intent = "new_search"
        intent.search_query = message  # same reasoning as check 1 above

    return intent


# ---------------------------------------------------------------------------
# Deterministic candidate filtering — stock/budget/color/size
#
# Extracted out of chat_fn as its own function for readability — takes
# plain arguments (max_budget, min_budget, color, size) rather than a
# UserIntent object, so this function has no dependency on UserIntent or
# anything else specific to chat_fn's scope. (Not moved to retrieval.py or
# a separate file: there's only one caller today — chat_fn — so a new
# module would just be overhead with no actual reuse benefit yet. Revisit
# if/when a second consumer shows up, e.g. an MCP tool or a test suite.)
# ---------------------------------------------------------------------------

def apply_deterministic_filters(
    enriched_candidates: list[dict],
    max_budget: Optional[float] = None,
    min_budget: Optional[float] = None,
    min_rating: Optional[float] = None,
    color: Optional[str] = None,
    size: Optional[str] = None,
) -> tuple[list[dict], dict]:
    """Filter already-enriched candidates (must have "in_stock", "base_price",
    "rating_avg", "variants" keys — i.e. already merged with
    get_product_details_bulk's output) by stock, budget, rating, and
    color/size — all deterministic, no LLM judgment involved.

    NOTE: no top-N slicing happens here (removed for now) — returns the FULL
    filtered set. Caller is responsible for any limiting, and match_info is
    computed on this same full set, so it's always consistent with whatever
    the caller then does with the returned candidates.

    Args:
        enriched_candidates: candidates already merged with live product
            details (base_price, in_stock, variants, etc.).
        max_budget, min_budget: hard price constraints, if stated.
        min_rating: hard rating threshold, if stated (e.g. "rating higher
            than 4"). A product with rating_avg=0.0 (no reviews yet) is
            correctly excluded by any positive threshold — 0.0 genuinely
            isn't >= 4, no special-casing needed.
        color, size: if stated, only keep candidates with an ACTUAL
            IN-STOCK VARIANT matching what was asked (not just "has this
            color somewhere in its variant list" — must be the SAME variant
            if both color AND size were given, and it must be in stock).
            If this would empty the pool entirely, relaxes back to the
            budget/stock-filtered set instead — better to show real,
            in-stock alternatives than nothing at all.

    Returns:
        (filtered_candidates, match_info)
        - filtered_candidates: ALL survivors after stock → budget →
          color/size filtering (with relaxation applied if needed).
        - match_info: {
              "exact_match_found": bool,  # some candidate has BOTH color AND
                                          # size (whichever were requested) on
                                          # the SAME in-stock variant
              "color_matched": bool,      # some candidate has the requested
                                          # color in stock, in ANY size
                                          # (True if color wasn't requested)
              "size_matched": bool,       # some candidate has the requested
                                          # size in stock, in ANY color
                                          # (True if size wasn't requested)
          }
    """
    in_stock_only = [c for c in enriched_candidates if c["in_stock"]]

    # Deterministic budget filter — hard constraint, same category as
    # age/stock (never recommend something explicitly outside the stated
    # budget). Uses live base_price from get_product_details_bulk, not a
    # stale/embedded value.
    if max_budget is not None:
        in_stock_only = [
            c for c in in_stock_only
            if c["base_price"] is not None and c["base_price"] <= max_budget
        ]
    if min_budget is not None:
        in_stock_only = [
            c for c in in_stock_only
            if c["base_price"] is not None and c["base_price"] >= min_budget
        ]
    if min_rating is not None:
        in_stock_only = [
            c for c in in_stock_only
            if c["rating_avg"] is not None and c["rating_avg"] >= min_rating
        ]

    def _match_flags(c: dict) -> tuple[bool, bool, bool]:
        """Per-candidate: (color_matched, size_matched, exact_matched),
        each checked ONLY against this candidate's in-stock variants."""
        in_stock_variants = [v for v in c["variants"] if v["in_stock"]]
        color_matched = not color or any(
            v["color_name"] and v["color_name"].lower() == color.lower()
            for v in in_stock_variants
        )
        size_matched = not size or any(
            v["size_label"] and str(v["size_label"]).lower() == str(size).lower()
            for v in in_stock_variants
        )
        exact_matched = (not color and not size) or any(
            (not color or (v["color_name"] and v["color_name"].lower() == color.lower()))
            and (not size or (v["size_label"] and str(v["size_label"]).lower() == str(size).lower()))
            for v in in_stock_variants
        )
        return color_matched, size_matched, exact_matched

    exact_matched_candidates = [c for c in in_stock_only if _match_flags(c)[2]]
    if exact_matched_candidates:
        in_stock_only = exact_matched_candidates
    elif size:
        # Size is a HARD constraint (unlike color, which is a soft
        # preference) — if size was requested, prioritize candidates that
        # at least match the size, even if color doesn't. If size matches
        # NOTHING at all, don't fall back to showing unrelated products —
        # return EMPTY, since recommending the wrong size isn't a
        # meaningful "alternative" the way a different color would be.
        size_matched_candidates = [c for c in in_stock_only if _match_flags(c)[1]]
        in_stock_only = size_matched_candidates  # may be [] — that's intentional
    # else (size wasn't requested at all): keep in_stock_only as the full
    # budget/stock-filtered set — this is the soft relaxation path for a
    # color-only mismatch, still showing real alternatives.

    any_color_matched = (not color) or any(_match_flags(c)[0] for c in in_stock_only)
    any_size_matched = (not size) or any(_match_flags(c)[1] for c in in_stock_only)
    exact_match_found = bool(in_stock_only) and any(_match_flags(c)[2] for c in in_stock_only)

    return in_stock_only, {
        "exact_match_found": exact_match_found,
        "color_matched": any_color_matched,
        "size_matched": any_size_matched,
    }


# ---------------------------------------------------------------------------
# Gradio wiring
# ---------------------------------------------------------------------------

def build_chat_fn(collection):
    """Return a Gradio-compatible chat function bound to the given collection.

    Uses gr.State (wired in launch(), single shared instance for both
    additional_inputs and additional_outputs) to accumulate known_products —
    {product_id: title} for every candidate shown so far this session.
    extract_user_intent() reads known_products to resolve follow-ups.

    NOTE: a visible gr.JSON debug panel showing the extracted intent was
    tried and reverted — it caused a silent crash under Gradio 6.20.0
    (confirmed via a standalone script that chat_fn's own logic is 100%
    correct; the failure was isolated to Gradio's handling of the extra
    gr.JSON additional_output, likely a version-specific compatibility
    issue in this very new major release). Use the terminal print() below
    to inspect intent during testing instead.

    Routing: RAG (build_candidate_block) only runs for "new_search". Every
    message still costs 1 Groq call (intent extraction); "new_search"
    messages cost a 2nd Groq call (narration) on top of that.
    """
    def chat_fn(message: str, history: list, known_products: dict) -> tuple[str, dict]:
        known_products = known_products or {}

        # Gradio's ChatInterface passes history as a list of
        # {"role": ..., "content": ...} dicts (the "messages" format), not
        # (user_msg, bot_msg) tuples — re-wrap defensively in case content
        # isn't a plain string (e.g. multimodal messages).
        history = [
            {"role": msg["role"], "content": msg["content"]} for msg in history
        ]

        intent = extract_user_intent(message, history, known_products=known_products)
        print(f"[chat_fn] intent={intent}")
        debug_lines = [
            "Groq call: extract_user_intent",
            f"  Input: message={message!r}",
            f"  Output: {intent}",
        ]

        def _debug_wrap(response_text: str) -> str:
            styled_lines = []
            for line in debug_lines:
                if ":" in line:
                    key, _, rest = line.partition(":")
                    styled_lines.append(f'<b style="color:#2563eb">{key}:</b>{rest}')
                else:
                    styled_lines.append(line)  # blank separator lines, kept as-is
            debug_html = (
                "\n\n<details><summary>Debug info</summary>\n\n<pre>"
                + "\n".join(styled_lines) + "</pre>\n</details>"
            )
            return response_text + debug_html

        # --- order_status: order HISTORY (no login, so email identifies customer) ---
        if intent.intent == "order_status" and intent.wants_order_history:
            if not intent.customer_email:
                return _debug_wrap("Sure — what's the email on your account?"), known_products

            customer = get_customer_by_email.invoke({"email": intent.customer_email})
            debug_lines.append("")
            debug_lines.append("Tool: get_customer_by_email")
            debug_lines.append(f"  Input: email={intent.customer_email!r}")
            debug_lines.append(f"  Output: {customer}")
            if "error" in customer:
                return _debug_wrap(f"I couldn't find an account with email {intent.customer_email}."), known_products

            limit = intent.order_history_limit or 5  # deterministic default for "a few" — never guessed by the LLM
            orders = get_recent_orders.invoke({"customer_id": customer["customer_id"], "limit": limit})
            debug_lines.append("")
            debug_lines.append("Tool: get_recent_orders")
            debug_lines.append(f"  Input: customer_id={customer['customer_id']!r}, limit={limit}")
            debug_lines.append(f"  Output: {orders}")

            if not orders:
                return _debug_wrap(f"No orders found for {intent.customer_email}."), known_products

            # Singular ("my last order") -> full detail via track_order.
            # Plural -> lighter summary list, not full item-by-item detail
            # for every order (too verbose for e.g. "my last 5 orders").
            if limit == 1:
                full = track_order.invoke({"order_id": orders[0]["order_id"]})
                debug_lines.append("")
                debug_lines.append("Tool: track_order")
                debug_lines.append(f"  Input: order_id={orders[0]['order_id']!r}")
                debug_lines.append(f"  Output: {full}")
                customer_part = f" for {full['customer_name']}" if full.get("customer_name") else ""
                parts = [f"Your last order, {full['order_id']}{customer_part}: {full['order_status']}."]
                if full.get("tracking_status"):
                    parts.append(f"Tracking: {full['tracking_status']}.")
                if full.get("delivered_date"):
                    parts.append(f"Delivered on {full['delivered_date']}.")
                elif full.get("expected_delivery_date"):
                    parts.append(f"Expected delivery: {full['expected_delivery_date']}.")
                return _debug_wrap(" ".join(parts)), known_products

            def _format_order(o):
                # order_date comes as "YYYY-MM-DD HH:MM:SS" — the time
                # portion isn't meaningful for a summary list, so just
                # show the date.
                date_only = str(o["order_date"]).split(" ")[0].split("T")[0]
                return f"- {o['order_id']} ({date_only}): {o['order_status']} — ${o['total_amount']}"

            order_lines = "\n".join(_format_order(o) for o in orders)
            return _debug_wrap(f"Your last {len(orders)} orders:\n{order_lines}"), known_products

        # --- order_status: invoke track_order if order_id known, else ask ---
        if intent.intent == "order_status":
            if not intent.order_id:
                return _debug_wrap("Sure — what's your order ID? (e.g. O00000123)"), known_products

            result = track_order.invoke({"order_id": intent.order_id})
            debug_lines.append("")
            debug_lines.append("Tool: track_order")
            debug_lines.append(f"  Input: order_id={intent.order_id!r}")
            debug_lines.append(f"  Output: {result}")

            if "error" in result:
                return _debug_wrap(f"I couldn't find an order with ID {intent.order_id}."), known_products

            customer_part = f" ({result['customer_name']})" if result.get("customer_name") else ""
            lines = [f"**Order {result['order_id']}{customer_part}**"]
            lines.append(f"Status: {result['order_status']}")
            if result.get("tracking_status"):
                carrier_part = f" ({result['carrier']})" if result.get("carrier") else ""
                lines.append(f"Tracking: {result['tracking_status']}{carrier_part}")
            if result.get("delivered_date"):
                lines.append(f"Delivered: {result['delivered_date']}")
            elif result.get("expected_delivery_date"):
                lines.append(f"Expected delivery: {result['expected_delivery_date']}")

            items = result.get("items") or []
            if items:
                def _format_item(item):
                    variant_bits = []
                    if item.get("size_label") and item["size_label"] != "One Size":
                        variant_bits.append(f"size {item['size_label']}")
                    if item.get("color_name"):
                        variant_bits.append(item["color_name"])
                    variant_text = f" ({', '.join(variant_bits)})" if variant_bits else ""
                    discount_text = f" (${item['discount_amount']} off)" if item.get("discount_amount") else ""
                    return (
                        f"- {item['quantity']}x {item['title']}{variant_text} — "
                        f"${item['line_total']}{discount_text}, {item['item_status']}"
                    )
                lines.append("")
                lines.append("Items:")
                lines.extend(_format_item(item) for item in items)

            return _debug_wrap("\n".join(lines)), known_products

        # --- follow_up: live-check ONE specific already-shown product ---
        # By this point (after both defensive checks in extract_user_intent),
        # intent.intent == "follow_up" is only possible if referenced_product_id
        # is BOTH non-null AND a real ID from known_products — guaranteed,
        # not just hoped for.
        if intent.intent == "follow_up":
            title = known_products.get(intent.referenced_product_id, {}).get("title", "that product")
            details = get_product_details.invoke({"product_id": intent.referenced_product_id})
            debug_lines.append("")
            debug_lines.append("Tool: get_product_details")
            debug_lines.append(f"  Input: product_id={intent.referenced_product_id!r}")
            debug_lines.append(f"  Output: {details}")

            if "error" in details:
                return _debug_wrap(f"I couldn't check {title} right now."), known_products

            # Reuse the EXACT SAME deterministic matching logic as new_search
            # — a single-item list input still correctly applies the
            # exact-variant-match + relaxation rules built earlier.
            filtered, match_info = apply_deterministic_filters(
                [details], color=intent.color, size=intent.size, min_rating=intent.min_rating,
            )
            debug_lines.append("")
            debug_lines.append("Function: apply_deterministic_filters")
            debug_lines.append(f"  Input: color={intent.color!r}, size={intent.size!r}, min_rating={intent.min_rating}")
            debug_lines.append(f"  Output: filtered={[d['product_id'] for d in filtered]}, match_info={match_info}")

            asked_parts = [p for p in [intent.color, intent.size] if p]
            asked_desc = " and ".join(asked_parts)

            # For an honest "doesn't come in X" vs "X is out of stock right
            # now" distinction — check ALL variants (not just in-stock ones,
            # unlike apply_deterministic_filters' matching, which correctly
            # only considers in-stock for the "never recommend OOS" rule).
            def _exists_at_all(field_name, value):
                if not value:
                    return True
                return any(
                    v.get(field_name) and str(v[field_name]).lower() == str(value).lower()
                    for v in details["variants"]
                )
            color_exists_at_all = _exists_at_all("color_name", intent.color)
            size_exists_at_all = _exists_at_all("size_label", intent.size)

            if not filtered:
                if intent.min_rating is not None and details.get("rating_avg") is not None and details["rating_avg"] < intent.min_rating:
                    answer = f"{title} is rated {details['rating_avg']}/5, which is below {intent.min_rating}."
                else:
                    answer = f"{title} is currently out of stock."
            elif match_info["exact_match_found"]:
                answer = f"Yes, {title} is available" + (f" in {asked_desc}" if asked_desc else "") + "."
            elif intent.size and not match_info["size_matched"]:
                if size_exists_at_all:
                    answer = f"Size {intent.size} for {title} is currently out of stock."
                else:
                    answer = f"{title} doesn't have size {intent.size} available."
            elif intent.color and not match_info["color_matched"]:
                if color_exists_at_all:
                    answer = f"{title} in {intent.color} is currently out of stock."
                else:
                    answer = f"{title} doesn't come in {intent.color}."
            elif asked_desc:
                answer = f"{title} doesn't have {asked_desc} available together."
            else:
                answer = f"{title} is in stock."

            return _debug_wrap(answer), known_products

        # --- new_search (covers new requests + refinements): RAG + narration ---
        candidates = build_candidate_block(
            intent.search_query, collection, customer_age=intent.recipient_age
        )
        debug_lines.append("")
        debug_lines.append("Chroma search: build_candidate_block")
        debug_lines.append(f"  Input: search_query={intent.search_query!r}, customer_age={intent.recipient_age}")
        debug_lines.append(f"  Output: {[(c['product_id'], c['title']) for c in candidates]}")

        # Deterministic stock-check — NOT an LLM/tool-calling decision, since
        # "never recommend an out-of-stock item" has no exceptions. Runs on
        # every age-filtered candidate (build_candidate_block no longer
        # slices to 3 internally — see its docstring), so there's a real
        # pool to backfill from if some turn out to be out of stock.
        #
        # Uses get_product_details_bulk (3 Supabase queries TOTAL) instead of
        # calling get_product_details in a loop (3 queries PER candidate,
        # i.e. up to 24 sequential round-trips for top_k=8) — the per-
        # candidate loop was the main cause of ~15s response times in testing.
        product_ids = [c["product_id"] for c in candidates]
        details_by_id = get_product_details_bulk.invoke({"product_ids": product_ids})
        print(f"[chat_fn] get_product_details_bulk called for {len(product_ids)} product_ids: {product_ids}")
        debug_lines.append("")
        debug_lines.append("Tool: get_product_details_bulk (Supabase)")
        debug_lines.append(f"  Input: product_ids={product_ids}")
        debug_lines.append("  Output:")
        for pid, d in details_by_id.items():
            line = (
                f"{pid}: base_price={d['base_price']}, in_stock={d['in_stock']}, "
                f"variants={[(v['size_label'], v['color_name'], v['available_qty']) for v in d['variants']]}"
            )
            print(f"  {line}")
            debug_lines.append(f"    {line}")
        missing = set(product_ids) - set(details_by_id.keys())
        if missing:
            print(f"  (not found in Supabase, skipped: {missing})")
            debug_lines.append(f"    (not found in Supabase, skipped: {missing})")

        enriched = [
            {**c, **details_by_id[c["product_id"]]}
            for c in candidates
            if c["product_id"] in details_by_id  # skip stale/missing since last rebuild
        ]

        # Deterministic stock/budget/color/size filtering — see
        # apply_deterministic_filters' own docstring for the full logic
        # (relaxation behavior, why color/size must match the SAME variant,
        # etc.). Passing plain values here, not `intent` itself, keeps that
        # function decoupled from UserIntent.
        #
        # NOTE: top-3 limiting REMOVED for now — top_picks is currently the
        # FULL filtered set, however many candidates that is. Revisit adding
        # a limit back later.
        filter_input = {
            "max_budget": intent.max_budget, "min_budget": intent.min_budget,
            "min_rating": intent.min_rating, "color": intent.color, "size": intent.size,
        }
        top_picks, match_info = apply_deterministic_filters(
            enriched,
            max_budget=intent.max_budget,
            min_budget=intent.min_budget,
            min_rating=intent.min_rating,
            color=intent.color,
            size=intent.size,
        )
        print(f"[chat_fn] apply_deterministic_filters -> match_info={match_info}")
        print(f"[chat_fn] top_picks: {[(c['product_id'], c['title']) for c in top_picks]}")
        debug_lines.append("")
        debug_lines.append("Function: apply_deterministic_filters")
        debug_lines.append(f"  Input: {filter_input}")
        debug_lines.append(f"  Output: match_info={match_info}")
        debug_lines.append(f"          top_picks={[(c['product_id'], c['title']) for c in top_picks]}")

        # Deterministic mismatch note — built in Python, not left to the LLM
        # to remember to mention. Prepended directly to the final answer
        # below, so this fact is guaranteed to reach the user regardless of
        # how the model behaves.
        #
        # Size vs. color get DIFFERENT framing, not the same treatment:
        # - Size is a hard constraint — the wrong size genuinely doesn't
        #   work for the customer, so a size mismatch is stated plainly as
        #   a real limitation ("size 9 isn't currently available").
        # - Color is more of a preference — a different color is still a
        #   fully usable product, so a color mismatch gets a softer framing
        #   ("showing other available colors") rather than being presented
        #   as a problem.
        mismatch_parts = []
        if intent.color or intent.size:
            if not match_info["exact_match_found"]:
                size_ok = match_info["size_matched"]  # True if size wasn't requested, or it matched
                if intent.size and not size_ok:
                    mismatch_parts.append(f"Size {intent.size} isn't currently available.")
                elif intent.color and not match_info["color_matched"]:
                    mismatch_parts.append(f"Showing other available colors (not {intent.color}).")
                elif intent.size and intent.color:
                    # Both individually exist somewhere, just never on the
                    # same in-stock variant together.
                    mismatch_parts.append(
                        f"Size {intent.size} and color {intent.color} aren't available "
                        f"together — showing other in-stock options instead."
                    )
        if not top_picks and not mismatch_parts:
            # Only reached if NOTHING more specific was already found above
            # (e.g. size hard-fail already produced its own message) — this
            # is purely a fallback for when budget/rating alone is why
            # nothing survived, not a blanket overwrite of a better reason.
            if intent.max_budget is not None or intent.min_budget is not None or intent.min_rating is not None:
                constraint_desc = " and ".join(
                    p for p in [
                        f"under ${intent.max_budget}" if intent.max_budget is not None else None,
                        f"over ${intent.min_budget}" if intent.min_budget is not None else None,
                        f"rated {intent.min_rating}+" if intent.min_rating is not None else None,
                    ] if p
                )
                mismatch_parts = [f"Nothing currently in stock is {constraint_desc}."]
        mismatch_note = (" ".join(mismatch_parts) + "\n\n") if mismatch_parts else ""

        # Fully deterministic short-circuit — if NOTHING survived all the
        # filtering (whatever the reason), skip the Groq narration call
        # entirely rather than sending an empty candidate list and hoping
        # the model follows the "say so plainly" instruction correctly.
        if not top_picks:
            fallback_msg = mismatch_note.strip() if mismatch_note else (
                "I couldn't find anything in stock matching that. "
                "Want me to widen the search?"
            )
            return _debug_wrap(fallback_msg), known_products

        def _size_sort_key(size_label: str):
            """Sort numeric sizes (e.g. '9', '10') numerically, so '10'
            doesn't come before '9' the way plain string sorting would.
            Non-numeric sizes (S/M/L/XL) fall back to string sort, placed
            after numeric ones for consistent ordering."""
            try:
                return (0, float(size_label))
            except (ValueError, TypeError):
                return (1, str(size_label))

        def _format_candidate(c: dict) -> str:
            rating_text = f"{c['rating_avg']}/5" if c["rating_avg"] else "not rated yet"

            # Build size -> [colors] instead of one flat color list — a flat
            # list loses the association (e.g. can't tell "size 9 only
            # comes in Black" from "size 10 only comes in Red" if both
            # just get merged into "Colors in stock: Black, Red").
            size_to_colors: dict[str, list[str]] = {}
            for v in c["variants"]:
                if v["in_stock"] and v["size_label"] and v["color_name"]:
                    size_to_colors.setdefault(v["size_label"], [])
                    if v["color_name"] not in size_to_colors[v["size_label"]]:
                        size_to_colors[v["size_label"]].append(v["color_name"])

            if size_to_colors:
                # "One Size" is a real value for products where sizing
                # genuinely doesn't apply (beauty, most camping gear, etc.)
                # — showing "Size One Size: Green" is technically consistent
                # with genuinely-sized items but reads oddly, especially
                # when a search mixes sized and unsized product types.
                # Collapse it to just the colors, no size label at all.
                if set(size_to_colors.keys()) == {"One Size"}:
                    availability_text = ", ".join(sorted(size_to_colors["One Size"]))
                else:
                    availability_text = "; ".join(
                        f"Size {size}: {', '.join(sorted(colors))}"
                        for size, colors in sorted(size_to_colors.items(), key=lambda item: _size_sort_key(item[0]))
                    )
            else:
                availability_text = "not available"

            return (
                f"**{c['title']}**\n"
                f"Price: ${c['base_price']}\n"
                f"Brand: {c['brand_name']} | Category: {c['category_name']}\n"
                f"Rating: {rating_text}\n"
                f"Material: {c['material'] or 'not specified'}\n"
                f"In stock: {availability_text}"
            )

        candidate_block = "\n\n".join(_format_candidate(c) for c in top_picks)

        # Explicit, unambiguous signal for anything the user specifically
        # asked about — don't rely on the LLM noticing "red"/"under $250"
        # buried in the raw message and correctly cross-referencing it
        # against candidate_block's actual data on its own.
        requested_attrs = []
        if intent.color:
            requested_attrs.append(f"color: {intent.color}")
        if intent.size:
            requested_attrs.append(f"size: {intent.size}")
        if intent.max_budget is not None:
            requested_attrs.append(f"max budget: ${intent.max_budget}")
        if intent.min_budget is not None:
            requested_attrs.append(f"min budget: ${intent.min_budget}")
        if intent.min_rating is not None:
            requested_attrs.append(f"min rating: {intent.min_rating}/5")
        requested_attrs_text = (
            f"User specifically asked about: {', '.join(requested_attrs)}\n\n"
            if requested_attrs
            else ""
        )

        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + history
            + [
                {
                    "role": "user",
                    "content": (
                        f"User query: {message}\n\n"
                        f"{requested_attrs_text}"
                        f"Live product data (from inventory/pricing lookup — price, "
                        f"stock, and colors below are current, not assumed from "
                        f"catalog description):\n{candidate_block}"
                    ),
                }
            ]
        )

        groq = _get_groq_client()
        completion = groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.1,
        )

        # Accumulate every candidate ACTUALLY SHOWN, keyed by product_id —
        # never overwrite/clear existing entries, so earlier turns stay
        # resolvable. Uses top_picks (post-stock-filter), not the raw
        # `candidates` list — no point tracking products that got filtered
        # out for being out of stock, since a later "is it available?"
        # follow-up wouldn't be about something never shown.
        #
        # Stores base_price AND sizes_shown alongside title (not just
        # title) — evidence showed the model reliably extracts attributes
        # like price/size from a message, but wasn't reliably
        # cross-referencing them against unstructured prior narration text
        # to resolve referenced_product_id. Giving these directly,
        # structured, right in the prompt, removes that unreliable extra
        # hop. Note: sizes_shown is a snapshot at display time — a real
        # follow-up still re-checks live stock via get_product_details,
        # this is ONLY used to help resolve WHICH product is meant, not
        # trusted as current stock fact.
        for c in top_picks:
            known_products[c["product_id"]] = {
                "title": c["title"],
                "base_price": c["base_price"],
                "sizes_shown": sorted(
                    {v["size_label"] for v in c["variants"] if v["in_stock"] and v["size_label"]}
                ),
                "colors_shown": sorted(
                    {v["color_name"] for v in c["variants"] if v["in_stock"] and v["color_name"]}
                ),
            }

        return _debug_wrap(mismatch_note + completion.choices[0].message.content), known_products
    return chat_fn


def launch(persist_path: str | None = ".chroma", share: bool = False, debug: bool = False, force_rebuild: bool = False):
    """Build the index (or load from disk) and launch the Gradio UI."""
    collection, _ = build_collection(persist_path=persist_path, force_rebuild=force_rebuild)

    # Pre-warm the embedder NOW, at startup — not on the first real user
    # message. Without this, the very first chat message pays for BOTH Groq
    # calls (intent extraction + narration) AND the one-time embedder model
    # load all in a single request, which is slow enough to risk hitting a
    # timeout (this was likely the cause of the "Error" seen consistently on
    # the first request in a session).
    print("Pre-warming embedder...")
    _get_embedder()
    print("Embedder ready.")

    chat_fn = build_chat_fn(collection)

    # One shared gr.State instance, used as BOTH the additional input and
    # additional output — guarantees chat_fn always reads back whatever it
    # last returned for known_products, rather than relying on Gradio to
    # link two separate State objects by position.
    known_products_state = gr.State(value={})

    demo = gr.ChatInterface(
        fn=chat_fn,
        additional_inputs=[known_products_state],
        additional_outputs=[known_products_state],
        title="ShopSage — Shopping Assistant (Week 1 Prototype)",
        description=(
            "RAG-grounded product search over the sample catalog. "
            "No live inventory/order tools yet — stock, size, and color claims "
            "are not verified in this build."
        ),
        examples=[
            ["Show me smart plugs to control my home appliances remotely.", None],
            ["I need hiking boots for a cold weather trip.", None],
            ["Sleeping bag for camping under $150.", None],
            ["What fitness gear do you have under $100?", None],
        ],
    )

    demo.launch(share=share, debug=debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShopSage Gradio chat UI.")
    parser.add_argument(
        "--persist-path",
        default=".chroma",
        help="Load/save the Chroma index from this directory. Defaults to .chroma.",
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
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild even if a populated collection already exists at --persist-path.",
    )
    args = parser.parse_args()
    launch(persist_path=args.persist_path, share=args.share, debug=args.debug, force_rebuild=args.force_rebuild)