#!/usr/bin/env python3
"""Tool definitions for ShopSage — all agent-facing @tool wrappers live here,
regardless of what subsystem their underlying logic touches (Supabase for
inventory/orders, Chroma for product search). This is the one file to check
for "what can the agent do."

Three public functions so far:
  - get_product_details(product_id) -> dict
    Consolidated live lookup: price, rating, material, and per-variant
    stock (size/color/available_qty) for ONE product. Built with future MCP
    exposure in mind — one clean input, one complete output. Use this for
    single-product lookups (e.g. a future "follow_up" tool call).
  - get_product_details_bulk(product_ids) -> dict[product_id, dict]
    Same data as get_product_details, but for MANY products in 3 TOTAL
    Supabase queries (using .in_()), not 3 queries PER product. Use this
    for enriching a whole candidate list (e.g. app.py's chat_fn, checking
    stock/price for every RAG candidate before narration) — calling
    get_product_details in a loop for N candidates means 3xN sequential
    network round-trips, which is what caused real, noticeable latency
    (~15s) in testing. This function exists specifically to fix that.
  - track_order(order_id) -> dict
    Joins orders + order_tracking for one order_id — current status,
    carrier, and delivery dates. Used by app.py's "order_status" branch.

NOTE: check_inventory and make_search_products_tool were built earlier and
have since been removed — they were never wired into the live app
(chat_fn), and get_product_details replaces what check_inventory was doing
(plus more, in one call). track_order was ALSO removed at that point, then
rebuilt here once order-status actually got wired into chat_fn.

Design notes:
  - All tools here return plain dicts, never raise on "not found" — callers
    (and the LLM agent) get a structured {"error": "..."} instead of an
    exception, so a bad ID doesn't crash the whole turn.
  - Inventory-related tools sum available_qty across warehouses for a given
    variant, since inventory_snapshots is one row per (product, variant,
    warehouse, date) — a variant can have stock in multiple warehouses. Only
    the MOST RECENT snapshot per warehouse is used (snapshot_date desc).
"""

from __future__ import annotations

from langchain_core.tools import tool

from scripts.rag.ingestion import _get_supabase_client


# ---------------------------------------------------------------------------
# Product details (price, rating, material, per-variant stock) — consolidated
# ---------------------------------------------------------------------------

@tool
def get_product_details(product_id: str) -> dict:
    """Get full current details for a product: live price, rating, material,
    and stock status for every active variant (size/color combination).

    This is the single source of truth for anything that must come from a
    live lookup rather than being assumed from a (possibly stale) RAG/
    catalog description — price, color availability, and stock all come
    from here in one call, rather than three separate ones.

    For enriching MANY candidates at once, use get_product_details_bulk
    instead — calling this in a loop costs 3 Supabase queries PER product,
    which adds up to real, noticeable latency for anything beyond 1-2 items.

    Args:
        product_id: The product to look up.

    Returns:
        On success:
            {
                "product_id": ...,
                "base_price": float,
                "rating_avg": float | None,
                "material": str | None,
                "in_stock": bool,          # True if ANY active variant has stock
                "variants": [
                    {
                        "variant_id": ...,
                        "size_label": ...,
                        "color_name": ...,
                        "available_qty": int,
                        "in_stock": bool,
                    },
                    ...
                ],
            }
        On failure:
            {"error": "product_not_found", "product_id": ...}
    """
    supabase = _get_supabase_client()

    product_resp = (
        supabase.table("products")
        .select("product_id, base_price, rating_avg, material")
        .eq("product_id", product_id)
        .execute()
    )
    if not product_resp.data:
        return {"error": "product_not_found", "product_id": product_id}
    product = product_resp.data[0]

    variant_resp = (
        supabase.table("product_variants")
        .select("variant_id, size_label, color_name")
        .eq("product_id", product_id)
        .eq("is_active", True)
        .execute()
    )
    variant_rows = variant_resp.data

    variants = []
    if variant_rows:
        variant_ids = [v["variant_id"] for v in variant_rows]
        snapshot_resp = (
            supabase.table("inventory_snapshots")
            .select("variant_id, warehouse_id, available_qty, snapshot_date")
            .eq("product_id", product_id)
            .in_("variant_id", variant_ids)
            .order("snapshot_date", desc=True)
            .execute()
        )

        # Most recent snapshot per (variant_id, warehouse_id), then sum
        # available_qty across warehouses for each variant.
        latest_per_variant_warehouse: dict = {}
        for row in snapshot_resp.data:
            key = (row["variant_id"], row["warehouse_id"])
            if key not in latest_per_variant_warehouse:  # first hit = most recent, sorted desc
                latest_per_variant_warehouse[key] = row

        qty_by_variant: dict = {}
        for (variant_id, _wh), row in latest_per_variant_warehouse.items():
            qty_by_variant[variant_id] = qty_by_variant.get(variant_id, 0) + row["available_qty"]

        for v in variant_rows:
            available_qty = qty_by_variant.get(v["variant_id"], 0)
            variants.append(
                {
                    "variant_id": v["variant_id"],
                    "size_label": v.get("size_label"),
                    "color_name": v.get("color_name"),
                    "available_qty": available_qty,
                    "in_stock": available_qty > 0,
                }
            )

    return {
        "product_id": product_id,
        "base_price": product.get("base_price"),
        "rating_avg": product.get("rating_avg"),
        "material": product.get("material"),
        "in_stock": any(v["in_stock"] for v in variants),
        "variants": variants,
    }


# ---------------------------------------------------------------------------
# Batched version — for enriching a whole candidate list, not one product.
# Not @tool-decorated: this is an internal efficiency helper, not something
# an LLM would ever choose to call (it wouldn't know the ideal batch of
# product_ids to pass) — app.py's chat_fn calls it directly.
# ---------------------------------------------------------------------------

def get_product_details_bulk(product_ids: list[str]) -> dict[str, dict]:
    """Same data as get_product_details, for MANY products, in 3 TOTAL
    Supabase queries (not 3 per product).

    Args:
        product_ids: List of product IDs to look up.

    Returns:
        {product_id: details_dict} — details_dict has the exact same shape
        as get_product_details' success return (base_price, rating_avg,
        material, in_stock, variants). Product IDs not found in Supabase
        are simply ABSENT from the returned dict (no per-item error dict,
        since the whole point here is batch efficiency) — callers should
        use .get(product_id) and treat a missing key as "not found."
    """
    if not product_ids:
        return {}

    supabase = _get_supabase_client()

    products_resp = (
        supabase.table("products")
        .select("product_id, base_price, rating_avg, material")
        .in_("product_id", product_ids)
        .execute()
    )
    products_by_id = {row["product_id"]: row for row in products_resp.data}

    variants_resp = (
        supabase.table("product_variants")
        .select("product_id, variant_id, size_label, color_name")
        .in_("product_id", product_ids)
        .eq("is_active", True)
        .execute()
    )
    variant_rows_by_product: dict[str, list[dict]] = {}
    all_variant_ids = []
    for row in variants_resp.data:
        variant_rows_by_product.setdefault(row["product_id"], []).append(row)
        all_variant_ids.append(row["variant_id"])

    qty_by_variant: dict[str, int] = {}
    if all_variant_ids:
        snapshots_resp = (
            supabase.table("inventory_snapshots")
            .select("variant_id, warehouse_id, available_qty, snapshot_date")
            .in_("variant_id", all_variant_ids)
            .order("snapshot_date", desc=True)
            .execute()
        )
        # Most recent snapshot per (variant_id, warehouse_id) across ALL
        # products at once, then sum available_qty per variant.
        latest_per_variant_warehouse: dict = {}
        for row in snapshots_resp.data:
            key = (row["variant_id"], row["warehouse_id"])
            if key not in latest_per_variant_warehouse:
                latest_per_variant_warehouse[key] = row
        for (variant_id, _wh), row in latest_per_variant_warehouse.items():
            qty_by_variant[variant_id] = qty_by_variant.get(variant_id, 0) + row["available_qty"]

    results: dict[str, dict] = {}
    for pid in product_ids:
        product = products_by_id.get(pid)
        if product is None:
            continue  # not found — simply absent from results, per docstring

        variants = []
        for v in variant_rows_by_product.get(pid, []):
            available_qty = qty_by_variant.get(v["variant_id"], 0)
            variants.append(
                {
                    "variant_id": v["variant_id"],
                    "size_label": v.get("size_label"),
                    "color_name": v.get("color_name"),
                    "available_qty": available_qty,
                    "in_stock": available_qty > 0,
                }
            )

        results[pid] = {
            "product_id": pid,
            "base_price": product.get("base_price"),
            "rating_avg": product.get("rating_avg"),
            "material": product.get("material"),
            "in_stock": any(v["in_stock"] for v in variants),
            "variants": variants,
        }

    return results


# ---------------------------------------------------------------------------
# Order status — joins orders + order_tracking for one order_id
# ---------------------------------------------------------------------------

@tool
def track_order(order_id: str) -> dict:
    """Get current status, tracking info, AND items for an order.

    Args:
        order_id: The order to look up (e.g. "O00000123").

    Returns:
        On success:
            {
                "order_id": ...,
                "customer_name": str | None,
                "customer_email": str | None,
                "order_status": str,       # from orders table (e.g. "shipped")
                "order_date": str,
                "total_amount": float,
                "carrier": str | None,
                "tracking_status": str | None,
                "shipped_date": str | None,
                "expected_delivery_date": str | None,
                "delivered_date": str | None,
                "items": [
                    {
                        "product_id": ...,
                        "title": str,       # "Unknown product" if the
                                            # product was since removed
                        "size_label": str | None,
                        "color_name": str | None,
                        "quantity": int,
                        "unit_price": float,
                        "discount_amount": float,  # per-line discount, 0 if none
                        "line_total": float,       # quantity*unit_price - discount
                        "item_status": str,    # e.g. "fulfilled", "cancelled"
                    },
                    ...
                ],
            }
        On failure:
            {"error": "order_not_found", "order_id": ...}
    """
    supabase = _get_supabase_client()

    order_resp = (
        supabase.table("orders")
        .select("order_id, customer_id, order_date, order_status, total_amount")
        .eq("order_id", order_id)
        .execute()
    )
    if not order_resp.data:
        return {"error": "order_not_found", "order_id": order_id}
    order = order_resp.data[0]

    customer_resp = (
        supabase.table("customers")
        .select("first_name, last_name, email")
        .eq("customer_id", order["customer_id"])
        .execute()
    )
    customer = customer_resp.data[0] if customer_resp.data else {}

    tracking_resp = (
        supabase.table("order_tracking")
        .select("carrier, tracking_status, shipped_date, expected_delivery_date, delivered_date")
        .eq("order_id", order_id)
        .execute()
    )
    tracking = tracking_resp.data[0] if tracking_resp.data else {}

    items_resp = (
        supabase.table("order_items")
        .select("product_id, variant_id, quantity, unit_price, discount_amount, item_status")
        .eq("order_id", order_id)
        .execute()
    )
    item_rows = items_resp.data

    title_by_id: dict = {}
    variant_by_id: dict = {}
    if item_rows:
        product_ids = list({row["product_id"] for row in item_rows})
        products_resp = (
            supabase.table("products")
            .select("product_id, title")
            .in_("product_id", product_ids)
            .execute()
        )
        title_by_id = {p["product_id"]: p["title"] for p in products_resp.data}

        variant_ids = list({row["variant_id"] for row in item_rows if row.get("variant_id")})
        if variant_ids:
            variants_resp = (
                supabase.table("product_variants")
                .select("variant_id, size_label, color_name")
                .in_("variant_id", variant_ids)
                .execute()
            )
            variant_by_id = {v["variant_id"]: v for v in variants_resp.data}

    items = [
        {
            "product_id": row["product_id"],
            "title": title_by_id.get(row["product_id"], "Unknown product"),
            "size_label": variant_by_id.get(row.get("variant_id"), {}).get("size_label"),
            "color_name": variant_by_id.get(row.get("variant_id"), {}).get("color_name"),
            "quantity": row["quantity"],
            "unit_price": row["unit_price"],
            "discount_amount": row.get("discount_amount") or 0,
            "line_total": round(row["quantity"] * row["unit_price"] - (row.get("discount_amount") or 0), 2),
            "item_status": row["item_status"],
        }
        for row in item_rows
    ]

    return {
        "order_id": order_id,
        "customer_name": f"{customer.get('first_name', '')} {customer.get('last_name', '')}".strip() or None,
        "customer_email": customer.get("email"),
        "order_status": order.get("order_status"),
        "order_date": order.get("order_date"),
        "total_amount": order.get("total_amount"),
        "carrier": tracking.get("carrier"),
        "tracking_status": tracking.get("tracking_status"),
        "shipped_date": tracking.get("shipped_date"),
        "expected_delivery_date": tracking.get("expected_delivery_date"),
        "delivered_date": tracking.get("delivered_date"),
        "items": items,
    }


# ---------------------------------------------------------------------------
# Customer identification + order history — for "give me my last N orders",
# which needs to know WHICH customer is asking. There's no login/session
# system in ShopSage, so email is used as the identifier: ask the user for
# it once, resolve to customer_id here, then cache customer_id for the rest
# of the session (see app.py's chat_fn) so it's only asked once per session.
# ---------------------------------------------------------------------------

@tool
def get_customer_by_email(email: str) -> dict:
    """Resolve a customer's email to their customer_id (+ name), so order
    history can be looked up. Case-insensitive match.

    Args:
        email: The email address to look up.

    Returns:
        On success: {"customer_id": ..., "first_name": ..., "last_name": ...}
        On failure: {"error": "customer_not_found", "email": ...}
    """
    supabase = _get_supabase_client()
    resp = (
        supabase.table("customers")
        .select("customer_id, first_name, last_name")
        .ilike("email", email)
        .execute()
    )
    if not resp.data:
        return {"error": "customer_not_found", "email": email}
    row = resp.data[0]
    return {
        "customer_id": row["customer_id"],
        "first_name": row.get("first_name"),
        "last_name": row.get("last_name"),
    }


@tool
def get_recent_orders(customer_id: str, limit: int = 5) -> list[dict]:
    """Get a customer's most recent orders, newest first.

    Args:
        customer_id: The customer to look up (from get_customer_by_email).
        limit: Max number of orders to return (default 5).

    Returns:
        list of {"order_id", "order_date", "order_status", "total_amount"},
        newest first. Empty list if the customer has no orders (not an
        error — a customer genuinely having zero orders is a valid state).
    """
    supabase = _get_supabase_client()
    resp = (
        supabase.table("orders")
        .select("order_id, order_date, order_status, total_amount")
        .eq("customer_id", customer_id)
        .order("order_date", desc=True)
        .limit(limit)
        .execute()
    )
    return resp.data or []