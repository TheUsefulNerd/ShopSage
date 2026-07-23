#!/usr/bin/env python3
"""ShopSage — Streamlit chat UI with hybrid retrieval, live inventory,
order tracking, and cross-session memory.

Run:
    streamlit run scripts/rag/streamlit_app.py

Deploy:
    Push to GitHub -> connect on share.streamlit.io -> set secrets in the
    Streamlit dashboard (SUPABASE_URL, SUPABASE_KEY, GROQ_API_KEY,
    QDRANT_URL, QDRANT_API).
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Optional

import streamlit as st
from groq import Groq
from pydantic import BaseModel, ValidationError

import sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.append(_project_root)

from scripts.rag.ingestion import build_collection
from scripts.rag.retrieval import build_candidate_block
from scripts.rag.tools import (
    get_product_details,
    get_product_details_bulk,
    track_order,
    get_customer_by_email,
    get_recent_orders,
)
from scripts.memory.store import MemoryStore

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ShopSage — AI Shopping Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — retail-forward design
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.04);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 8px;
    padding: 4px;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    color: white !important;
}

/* Scenario buttons */
.stButton > button {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 10px;
    color: white;
    font-size: 0.82rem;
    padding: 8px 12px;
    width: 100%;
    text-align: left;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: rgba(138, 43, 226, 0.3);
    border-color: rgba(138, 43, 226, 0.6);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(138, 43, 226, 0.2);
}

/* Header */
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #ec4899, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.hero-subtitle {
    color: rgba(255, 255, 255, 0.55);
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}

/* Memory card */
.memory-card {
    background: rgba(167, 139, 250, 0.1);
    border: 1px solid rgba(167, 139, 250, 0.3);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 12px;
    font-size: 0.85rem;
    color: rgba(255,255,255,0.85);
}

/* Scenario group label */
.scenario-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(255, 255, 255, 0.4);
    margin: 14px 0 6px 2px;
}

/* Tech badge row */
.tech-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 8px;
}
.tech-badge {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.7rem;
    color: rgba(255,255,255,0.6);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Groq config
# ---------------------------------------------------------------------------

GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are ShopSage, a helpful, conversational AI shopping assistant.\n"
    "You are given a user query and a block of 'Live product data' representing the current search results or context.\n"
    "All candidates in the live product data have ALREADY been verified against budget, color, and size constraints — you do not need to re-verify this.\n"
    "Answer using ONLY the information in the Live product data. Never invent products, colors, sizes, or prices.\n\n"
    "If the user is asking a direct question (e.g., 'what colors are available?', 'is it in stock?'), ANSWER their question naturally based on the data.\n"
    "If the user is doing a general search, present the top options clearly with their price, rating, and relevant details, and end with a short question inviting them to narrow it down."
)

_groq_client: Optional[Groq] = None


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ---------------------------------------------------------------------------
# Intent extraction (same UserIntent as before)
# ---------------------------------------------------------------------------

class UserIntent(BaseModel):
    search_query: str
    intent: str = "new_search"
    color: Optional[str] = None
    size: Optional[str] = None
    min_budget: Optional[float] = None
    max_budget: Optional[float] = None
    min_rating: Optional[float] = None
    recipient_age: Optional[int] = None
    order_id: Optional[str] = None
    wants_order_history: bool = False
    customer_email: Optional[str] = None
    order_history_limit: Optional[int] = None
    referenced_product_id: Optional[str] = None


def _build_intent_prompt(known_products: dict) -> str:
    if known_products:
        known_list = "\n".join(
            f'  "{pid}": "{info["title"]}", ${info["base_price"]}, '
            f'sizes: {", ".join(info["sizes_shown"]) if info.get("sizes_shown") else "n/a"}, '
            f'colors: {", ".join(info.get("colors_shown") or []) or "n/a"}'
            for pid, info in known_products.items()
        )
        known_section = f"Products shown so far:\n{known_list}\n\n"
    else:
        known_section = "No products shown yet.\n\n"

    return (
        f"{known_section}"
        "Extract JSON with fields: search_query (string), intent (new_search/follow_up/order_status), "
        "color (string|null), size (string|null), min_budget (number|null), max_budget (number|null), "
        "min_rating (number|null), recipient_age (int|null), order_id (string|null), "
        "wants_order_history (bool), customer_email (string|null), order_history_limit (int|null), "
        "referenced_product_id (string|null). "
        "search_query must be standalone and self-contained. "
        "For follow_up, referenced_product_id must be a REAL id from the list above. "
        "When genuinely unsure between follow_up and new_search, prefer new_search. "
        "Respond with ONLY the JSON object."
    )


def extract_intent(message: str, history: list, known_products: dict) -> UserIntent:
    messages = [{"role": "system", "content": _build_intent_prompt(known_products)}]
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
        intent = UserIntent(**json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"[extract_intent] parse failed: {e}")
        intent = UserIntent(search_query=message)

    # Defensive checks
    if intent.referenced_product_id and intent.referenced_product_id not in known_products:
        intent.referenced_product_id = None
        intent.intent = "new_search"
        intent.search_query = message
    if intent.intent == "follow_up" and not intent.referenced_product_id:
        intent.intent = "new_search"
        intent.search_query = message

    return intent


# ---------------------------------------------------------------------------
# Deterministic filtering (identical logic to Kasturi's app.py)
# ---------------------------------------------------------------------------

def apply_deterministic_filters(
    enriched: list[dict],
    max_budget: Optional[float] = None,
    min_budget: Optional[float] = None,
    min_rating: Optional[float] = None,
    color: Optional[str] = None,
    size: Optional[str] = None,
) -> tuple[list[dict], dict]:
    in_stock = [c for c in enriched if c["in_stock"]]

    if max_budget is not None:
        in_stock = [c for c in in_stock if c["base_price"] is not None and c["base_price"] <= max_budget]
    if min_budget is not None:
        in_stock = [c for c in in_stock if c["base_price"] is not None and c["base_price"] >= min_budget]
    if min_rating is not None:
        in_stock = [c for c in in_stock if c["rating_avg"] is not None and c["rating_avg"] >= min_rating]

    def _flags(c):
        in_stock_v = [v for v in c["variants"] if v["in_stock"]]
        color_ok = not color or any(
            v["color_name"] and v["color_name"].lower() == color.lower() for v in in_stock_v
        )
        size_ok = not size or any(
            v["size_label"] and str(v["size_label"]).lower() == str(size).lower() for v in in_stock_v
        )
        exact = (not color and not size) or (color_ok and size_ok)
        return color_ok, size_ok, exact

    exact_set = [c for c in in_stock if _flags(c)[2]]
    if exact_set:
        in_stock = exact_set
    elif size:
        in_stock = [c for c in in_stock if _flags(c)[1]]

    any_color = (not color) or any(_flags(c)[0] for c in in_stock)
    any_size = (not size) or any(_flags(c)[1] for c in in_stock)
    exact_found = bool(in_stock) and any(_flags(c)[2] for c in in_stock)

    return in_stock, {"exact_match_found": exact_found, "color_matched": any_color, "size_matched": any_size}


def _format_candidate(c: dict) -> str:
    rating_text = f"{c['rating_avg']}/5" if c["rating_avg"] else "not rated yet"
    size_to_colors: dict[str, list[str]] = {}
    for v in c["variants"]:
        if v["in_stock"] and v["size_label"] and v["color_name"]:
            size_to_colors.setdefault(v["size_label"], [])
            if v["color_name"] not in size_to_colors[v["size_label"]]:
                size_to_colors[v["size_label"]].append(v["color_name"])

    if size_to_colors:
        if set(size_to_colors.keys()) == {"One Size"}:
            availability = ", ".join(sorted(size_to_colors["One Size"]))
        else:
            def _sort_key(s):
                try:
                    return (0, float(s))
                except (ValueError, TypeError):
                    return (1, str(s))
            availability = "; ".join(
                f"Size {sz}: {', '.join(sorted(cols))}"
                for sz, cols in sorted(size_to_colors.items(), key=lambda x: _sort_key(x[0]))
            )
    else:
        availability = "not available"

    return (
        f"**{c['title']}**\n"
        f"Price: ${c['base_price']}\n"
        f"Brand: {c['brand_name']} | Category: {c['category_name']}\n"
        f"Rating: {rating_text}\n"
        f"Material: {c['material'] or 'not specified'}\n"
        f"In stock: {availability}"
    )


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def _init_session():
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "known_products" not in st.session_state:
        st.session_state.known_products = {}
    if "memory_store" not in st.session_state:
        store = MemoryStore(st.session_state.user_id)
        store.increment_session()
        st.session_state.memory_store = store
    if "collection_built" not in st.session_state:
        with st.spinner("Connecting to ShopSage catalog..."):
            build_collection()
        st.session_state.collection_built = True
    if "greeted" not in st.session_state:
        st.session_state.greeted = False
        greeting = st.session_state.memory_store.build_greeting()
        if greeting:
            st.session_state.messages.append({"role": "assistant", "content": greeting})
        else:
            welcome = (
                "👋 Hi! I'm **ShopSage**, your AI shopping assistant.\n\n"
                "I can help you find products across **Outdoor**, **Clothing**, "
                "**Footwear**, **Hiking**, and **Beauty** categories. "
                "I check live inventory, respect your budget, and can track your orders.\n\n"
                "What are you shopping for today?"
            )
            st.session_state.messages.append({"role": "assistant", "content": welcome})
        st.session_state.greeted = True


# ---------------------------------------------------------------------------
# Message processing
# ---------------------------------------------------------------------------

def process_message(user_message: str) -> str:
    """Core chat logic — returns the assistant's response string."""
    history = st.session_state.messages[:-1]  # exclude the message we just appended
    known_products = st.session_state.known_products
    memory_store: MemoryStore = st.session_state.memory_store

    intent = extract_intent(user_message, history, known_products)
    print(f"[process_message] intent={intent}")

    # --- Order history ---
    if intent.intent == "order_status" and intent.wants_order_history:
        if not intent.customer_email:
            return "Sure — what's the email on your account?"
        customer = get_customer_by_email.invoke({"email": intent.customer_email})
        if "error" in customer:
            return f"I couldn't find an account with email {intent.customer_email}."
        limit = intent.order_history_limit or 5
        orders = get_recent_orders.invoke({"customer_id": customer["customer_id"], "limit": limit})
        if not orders:
            return f"No orders found for {intent.customer_email}."
        if limit == 1:
            full = track_order.invoke({"order_id": orders[0]["order_id"]})
            parts = [f"Your last order **{full['order_id']}**: {full['order_status']}."]
            if full.get("tracking_status"):
                parts.append(f"Tracking: {full['tracking_status']}.")
            if full.get("expected_delivery_date"):
                parts.append(f"Expected delivery: {full['expected_delivery_date']}.")
            return " ".join(parts)
        lines = [f"Your last {len(orders)} orders:"]
        for o in orders:
            date = str(o["order_date"]).split(" ")[0].split("T")[0]
            lines.append(f"- **{o['order_id']}** ({date}): {o['order_status']} — ${o['total_amount']}")
        return "\n".join(lines)

    # --- Single order status ---
    if intent.intent == "order_status":
        if not intent.order_id:
            return "Sure — what's your order ID? (e.g. O00000123)"
        result = track_order.invoke({"order_id": intent.order_id})
        if "error" in result:
            return f"I couldn't find an order with ID {intent.order_id}."
        lines = [f"**Order {result['order_id']}**"]
        lines.append(f"Status: {result['order_status']}")
        if result.get("tracking_status"):
            lines.append(f"Tracking: {result['tracking_status']}" + (f" ({result['carrier']})" if result.get("carrier") else ""))
        if result.get("expected_delivery_date"):
            lines.append(f"Expected delivery: {result['expected_delivery_date']}")
        items = result.get("items") or []
        if items:
            lines.append("\nItems:")
            for item in items:
                variant = ", ".join(filter(None, [
                    f"size {item['size_label']}" if item.get("size_label") and item["size_label"] != "One Size" else None,
                    item.get("color_name"),
                ]))
                lines.append(f"- {item['quantity']}x {item['title']}" + (f" ({variant})" if variant else "") + f" — ${item['line_total']}, {item['item_status']}")
        return "\n".join(lines)

    # --- Candidate Generation ---
    if intent.intent == "follow_up" and intent.referenced_product_id:
        candidates = [{"product_id": intent.referenced_product_id}]
    else:
        # RAG Search
        candidates = build_candidate_block(intent.search_query, customer_age=intent.recipient_age)

    product_ids = [c["product_id"] for c in candidates]
    details_by_id = get_product_details_bulk.invoke({"product_ids": product_ids})

    enriched = [
        {**c, **details_by_id[c["product_id"]]}
        for c in candidates
        if c["product_id"] in details_by_id
    ]

    top_picks, match_info = apply_deterministic_filters(
        enriched,
        max_budget=intent.max_budget,
        min_budget=intent.min_budget,
        min_rating=intent.min_rating,
        color=intent.color,
        size=intent.size,
    )

    # Build mismatch note
    mismatch_parts = []
    if intent.color or intent.size:
        if not match_info["exact_match_found"]:
            if intent.size and not match_info["size_matched"]:
                mismatch_parts.append(f"Size {intent.size} isn't currently available.")
            elif intent.color and not match_info["color_matched"]:
                mismatch_parts.append(f"Showing other available colors (not {intent.color}).")
            elif intent.size and intent.color:
                mismatch_parts.append(
                    f"Size {intent.size} and {intent.color} aren't available together — showing other options."
                )
    if not top_picks and not mismatch_parts:
        if intent.max_budget is not None or intent.min_budget is not None or intent.min_rating is not None:
            constraint = " and ".join(filter(None, [
                f"under ${intent.max_budget}" if intent.max_budget else None,
                f"over ${intent.min_budget}" if intent.min_budget else None,
                f"rated {intent.min_rating}+" if intent.min_rating else None,
            ]))
            mismatch_parts = [f"Nothing currently in stock is {constraint}."]

    mismatch_note = (" ".join(mismatch_parts) + "\n\n") if mismatch_parts else ""

    if not top_picks:
        return (mismatch_note.strip() if mismatch_note else
                "I couldn't find anything in stock matching that. Want me to widen the search?")

    candidate_block = "\n\n".join(_format_candidate(c) for c in top_picks)

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
    attrs_text = f"User specifically asked about: {', '.join(requested_attrs)}\n\n" if requested_attrs else ""

    groq_messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": f"User query: {user_message}\n\n{attrs_text}Live product data:\n{candidate_block}"}]
    )
    groq = _get_groq_client()
    completion = groq.chat.completions.create(model=GROQ_MODEL, messages=groq_messages, temperature=0.1)
    answer = completion.choices[0].message.content

    # Update known_products for follow-up resolution
    for c in top_picks:
        known_products[c["product_id"]] = {
            "title": c["title"],
            "base_price": c["base_price"],
            "sizes_shown": sorted({v["size_label"] for v in c["variants"] if v["in_stock"] and v["size_label"]}),
            "colors_shown": sorted({v["color_name"] for v in c["variants"] if v["in_stock"] and v["color_name"]}),
        }
    st.session_state.known_products = known_products

    # Persist preferences to Supabase
    memory_store.extract_and_save(intent)

    return mismatch_note + answer


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="hero-title">🛍️ ShopSage</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-subtitle">AI-Powered Shopping Assistant</div>', unsafe_allow_html=True)

        # Memory card
        store: MemoryStore = st.session_state.get("memory_store")
        if store:
            prefs = store.load()
            if prefs:
                parts = []
                if prefs.get("max_budget"):
                    parts.append(f"💰 Budget: under ${prefs['max_budget']:.0f}")
                cats = prefs.get("preferred_categories") or []
                if cats:
                    parts.append(f"📂 Interests: {', '.join(cats[:3])}")
                sessions = prefs.get("session_count", 1)
                parts.append(f"🔁 Sessions: {sessions}")
                if parts:
                    st.markdown(
                        f'<div class="memory-card">🧠 <b>Your Preferences</b><br>' +
                        "<br>".join(parts) + "</div>",
                        unsafe_allow_html=True,
                    )

        st.divider()

        # Scenario buttons
        st.markdown('<div class="scenario-label">🔍 Product Search</div>', unsafe_allow_html=True)
        scenarios_search = [
            ("🏕️ Camping tent", "I need a good tent for camping trips"),
            ("👟 Running shoes", "Show me running shoes for everyday use"),
            ("🧥 Winter jacket", "I need a warm jacket for cold weather"),
        ]
        for label, prompt in scenarios_search:
            if st.button(label, key=f"btn_{label}"):
                st.session_state.pending_prompt = prompt

        st.markdown('<div class="scenario-label">💰 Budget & Filters</div>', unsafe_allow_html=True)
        scenarios_budget = [
            ("💸 Boots under $150", "Hiking boots under $150"),
            ("⭐ Top rated gear", "Best rated camping gear above 4 stars"),
            ("🎨 Red hiking boots sz 9", "Red hiking boots in size 9"),
        ]
        for label, prompt in scenarios_budget:
            if st.button(label, key=f"btn_{label}"):
                st.session_state.pending_prompt = prompt

        st.markdown('<div class="scenario-label">📦 Orders & More</div>', unsafe_allow_html=True)
        scenarios_orders = [
            ("📬 Track my order", "Where is my order?"),
            ("📋 My recent orders", "Show me my last 3 orders"),
            ("🎁 Gift for a child", "Gift ideas for a 10-year-old child"),
        ]
        for label, prompt in scenarios_orders:
            if st.button(label, key=f"btn_{label}"):
                st.session_state.pending_prompt = prompt

        st.divider()

        # How it works accordion
        with st.expander("ℹ️ How ShopSage works"):
            st.markdown("""
**What I can do:**
- 🔍 Search 250 products across 5 categories
- 💰 Respect your exact budget (guaranteed — never over)
- 📦 Check live inventory and track orders
- 🎨 Filter by color, size, rating
- 🧠 Remember your preferences across sessions

**Categories:** Outdoor · Clothing · Footwear · Hiking · Beauty

**What I can't do:**
- Process payments or place orders
- Search outside the catalog
- Access real-time pricing beyond the catalog

**Tech stack:**
            """)
            st.markdown("""
<div class="tech-badges">
  <span class="tech-badge">Qdrant Hybrid Search</span>
  <span class="tech-badge">BM25 + Semantic</span>
  <span class="tech-badge">Groq LLaMA-3</span>
  <span class="tech-badge">Supabase</span>
  <span class="tech-badge">Live Inventory</span>
</div>
            """, unsafe_allow_html=True)

        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.known_products = {}
            st.rerun()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    _init_session()
    render_sidebar()

    # Main content area
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="hero-title" style="font-size:1.6rem;">Your AI Shopping Assistant</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-subtitle">Powered by hybrid vector search · Live inventory · Budget-aware guardrails</div>',
            unsafe_allow_html=True
        )

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # Handle scenario button presses
    if "pending_prompt" in st.session_state:
        prompt = st.session_state.pop("pending_prompt")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching catalog..."):
                response = process_message(prompt)
            st.markdown(response, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    # Chat input
    if user_input := st.chat_input("Ask me anything — 'hiking boots under $150', 'track my order'..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Searching catalog..."):
                response = process_message(user_input)
            st.markdown(response, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    main()
