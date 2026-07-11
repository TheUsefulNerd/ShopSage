#!/usr/bin/env python3
"""Deterministic synthetic retail warehouse generator.

Writes normalized CSV files with stable foreign keys and realistic relationships.
The generator is designed to scale to 25M+ rows without holding the full warehouse
in memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator, Sequence, Tuple

SEED = 42

CATEGORIES = [
    "Outdoor",
    "Clothing",
    "Electronics",
    "Home",
    "Fitness",
    "Kids",
    "Beauty",
    "Sports",
    "Grocery",
    "Toys",
    "Office",
    "Pets",
    "Automotive",
    "Garden",
    "Health",
    "Books",
    "Footwear",
    "Accessories",
    "Travel",
    "Gaming",
    "Appliances",
    "Bedding",
    "Kitchen",
    "Decor",
    "Lighting",
    "Storage",
    "Cleaning",
    "Stationery",
    "Baby",
    "Luggage",
    "Audio",
    "Wearables",
    "Smart Home",
    "Cycling",
    "Running",
    "Camping",
    "Hiking",
    "Fishing",
    "Swimming",
    "Yoga",
    "Skincare",
    "Haircare",
    "Supplements",
    "Gadgets",
    "Computing",
    "Networking",
    "Monitors",
    "Peripherals",
    "Fragrance",
    "Jewelry",
]

BRANDS = [f"Brand{n:03d}" for n in range(1, 201)]
WAREHOUSES = [
    ("North", "US-North"),
    ("South", "US-South"),
    ("East", "US-East"),
    ("West", "US-West"),
    ("Central", "US-Central"),
]
WAREHOUSES = [(f"W{idx+1:03d}", f"Warehouse {name}", region) for idx, (name, region) in enumerate(WAREHOUSES * 5)][:25]

COLORS = ["Black", "White", "Blue", "Green", "Red", "Gray", "Navy", "Pink", "Brown", "Olive", "Teal", "Purple"]
SIZES = ["XS", "S", "M", "L", "XL", "XXL", "One Size"]
MATERIALS = ["Polyester", "Cotton", "Nylon", "Stainless Steel", "Aluminum", "Leather", "Glass", "Ceramic", "Silicone", "Wood", "Plastic", "Wool"]
TIER = ["Value", "Mid", "Premium"]
CHANNELS = ["web", "mobile", "marketplace", "store"]
ORDER_STATUSES = ["placed", "packed", "shipped", "out_for_delivery", "delivered", "cancelled", "returned"]
ITEM_STATUSES = ["fulfilled", "backordered", "cancelled", "returned"]
TRACKING_STATUSES = ["label_created", "in_transit", "out_for_delivery", "delivered", "exception"]
EVENT_TYPES = ["search", "view", "click", "add_to_cart", "wishlist", "purchase"]
TOOL_NAMES = ["inventory", "order_tracking", "recommendation", "cart", "memory"]
CARRIERS = ["BlueDart", "Delhivery", "DHL", "FedEx", "UPS", "USPS", "Aramex"]
FIRST_NAMES = ["Aarav", "Priya", "Isha", "Rohan", "Anika", "Kabir", "Meera", "Arjun", "Sara", "Nikhil", "Sana", "Vikram", "Neha", "Rahul", "Pooja", "Dev", "Kavya", "Anushka", "Aditya", "Simran"]
LAST_NAMES = ["Kapoor", "Sharma", "Verma", "Patel", "Iyer", "Gupta", "Reddy", "Nair", "Singh", "Malhotra", "Mehta", "Bose", "Chopra", "Agarwal", "Jain"]

PRODUCTS = [
    "Jacket", "Boots", "Backpack", "Headphones", "Coffee Maker", "Yoga Mat", "T-shirt", "Lamp", "Bottle", "Tent",
    "Microwave", "Chair", "Desk", "Monitor", "Mouse", "Keyboard", "Smartwatch", "Speaker", "Skincare Set", "Hair Dryer",
    "Blanket", "Pillow", "Pan", "Knife Set", "Sunglasses", "Wallet", "Suitcase", "Bike Helmet", "Running Shoes", "Fishing Rod"
]

ROOT = Path(".")


def dt(days_back: int = 0, hours_back: int = 0) -> datetime:
    return datetime(2026, 7, 11, 12, 0, 0) - timedelta(days=days_back, hours=hours_back)


def stable_choice(seq: Sequence, idx: int):
    return seq[idx % len(seq)]


def chunked_range(n: int, chunk_size: int) -> Iterator[Tuple[int, int]]:
    for start in range(0, n, chunk_size):
        yield start, min(start + chunk_size, n)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence], mode: str = "w") -> None:
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def p_id(i: int) -> str:
    return f"P{i:07d}"


def v_id(i: int) -> str:
    return f"V{i:08d}"


def c_id(i: int) -> str:
    return f"C{i:07d}"


def o_id(i: int) -> str:
    return f"O{i:08d}"


def oi_id(i: int) -> str:
    return f"OI{i:09d}"


def r_id(i: int) -> str:
    return f"R{i:08d}"


def e_id(i: int) -> str:
    return f"E{i:09d}"


def t_id(i: int) -> str:
    return f"T{i:08d}"


def s_id(i: int) -> str:
    return f"S{i:08d}"


def generate_categories(out: Path) -> List[Tuple[str, str, str, bool]]:
    rows = []
    header = ["category_id", "category_name", "parent_category_id", "is_age_sensitive"]
    for i, name in enumerate(CATEGORIES, start=1):
        cat_id = f"CAT{i:03d}"
        parent = f"CAT{((i - 1) // 7) + 1:03d}" if i > 7 else ""
        age_sensitive = name in {"Beauty", "Supplements", "Gadgets", "Fragrance", "Jewelry"}
        rows.append((cat_id, name, parent, age_sensitive))
    write_csv(out / "categories.csv", header, rows)
    return rows


def generate_brands(out: Path) -> List[Tuple[str, str, str]]:
    rows = []
    header = ["brand_id", "brand_name", "tier"]
    for i, name in enumerate(BRANDS, start=1):
        rows.append((f"BR{i:03d}", name, TIER[i % len(TIER)]))
    write_csv(out / "brands.csv", header, rows)
    return rows


def generate_warehouses(out: Path) -> List[Tuple[str, str, str]]:
    header = ["warehouse_id", "warehouse_name", "region"]
    write_csv(out / "warehouses.csv", header, WAREHOUSES)
    return WAREHOUSES


def product_attributes(category: str, i: int) -> Tuple[str, str, bool, int, str, str]:
    title_base = stable_choice(PRODUCTS, i)
    color = stable_choice(COLORS, i * 3 + 1)
    material = stable_choice(MATERIALS, i * 5 + 2)
    age_sensitive = category in {"Beauty", "Supplements", "Fragrance", "Jewelry"}
    min_age = 18 if age_sensitive else 0
    description = (
        f"{color} {title_base.lower()} for {category.lower()} use with durable {material.lower()} construction. "
        f"Designed for search, compare, and recommendation workflows."
    )
    return title_base, description, age_sensitive, min_age, color, material


def generate_products(out: Path, n_products: int, rng: random.Random, chunk_size: int = 100_000):
    header = ["product_id", "category_id", "brand_id", "title", "description", "base_price", "rating_avg", "review_count", "age_restricted", "min_age", "color_family", "material", "created_at"]
    path = out / "products.csv"
    first = True
    for start, end in chunked_range(n_products, chunk_size):
        rows = []
        for i in range(start, end):
            category_id = f"CAT{(i % len(CATEGORIES)) + 1:03d}"
            category_name = CATEGORIES[i % len(CATEGORIES)]
            brand_id = f"BR{(i % len(BRANDS)) + 1:03d}"
            title_base, description, age_sensitive, min_age, color, material = product_attributes(category_name, i)
            base_price = round(8 + ((i * 7) % 492) + ((i % 100) / 100), 2)
            rating_avg = round(2.8 + ((i * 17) % 22) / 10, 2)
            review_count = 5 + (i % 5000)
            created_at = (dt(days_back=(i % 1460)) - timedelta(hours=i % 24)).isoformat(sep=" ")
            rows.append((p_id(i), category_id, brand_id, f"{stable_choice(BRANDS, i)} {title_base}", description, base_price, rating_avg, review_count, age_sensitive, min_age, color, material, created_at))
        write_csv(path, header, rows, mode="w" if first else "a")
        first = False


def generate_variants(out: Path, n_products: int, n_variants: int, rng: random.Random, chunk_size: int = 100_000):
    header = ["variant_id", "product_id", "size_label", "color_name", "sku", "weight_grams", "is_active"]
    path = out / "product_variants.csv"
    first = True
    for start, end in chunked_range(n_variants, chunk_size):
        rows = []
        for i in range(start, end):
            product_idx = i % n_products
            color = stable_choice(COLORS, i * 7 + 3)
            size = stable_choice(SIZES, i * 5 + 1)
            sku = f"SKU-{product_idx:07d}-{i % 9999:04d}"
            weight = 100 + (i % 4000)
            rows.append((v_id(i), p_id(product_idx), size, color, sku, weight, (i % 17) != 0))
        write_csv(path, header, rows, mode="w" if first else "a")
        first = False


def generate_customers(out: Path, n_customers: int, rng: random.Random, chunk_size: int = 100_000):
    header = ["customer_id", "first_name", "last_name", "email", "signup_date", "home_region", "age"]
    path = out / "customers.csv"
    first = True
    regions = [r[2] for r in WAREHOUSES]
    for start, end in chunked_range(n_customers, chunk_size):
        rows = []
        for i in range(start, end):
            fn = stable_choice(FIRST_NAMES, i)
            ln = stable_choice(LAST_NAMES, i * 2 + 1)
            email = f"{fn.lower()}.{ln.lower()}{i % 1000:03d}@example.com"
            signup = (date(2022, 1, 1) + timedelta(days=i % 1280)).isoformat()
            region = stable_choice(regions, i)
            age = 18 + (i % 47)
            rows.append((c_id(i), fn, ln, email, signup, region, age))
        write_csv(path, header, rows, mode="w" if first else "a")
        first = False


def generate_preferences(out: Path, n_customers: int, chunk_size: int = 100_000):
    header = ["customer_id", "preferred_categories", "preferred_brands", "budget_max", "style_notes", "updated_at"]
    path = out / "customer_preferences.csv"
    first = True
    for start, end in chunked_range(n_customers, chunk_size):
        rows = []
        for i in range(start, end):
            cats = ",".join([CATEGORIES[(i + j) % len(CATEGORIES)] for j in range(3)])
            brands = ",".join([BRANDS[(i + j * 11) % len(BRANDS)] for j in range(3)])
            budget = round(25 + (i % 275), 2)
            note = ["minimal", "sporty", "technical", "casual", "premium", "budget", "outdoor", "travel"][(i * 3) % 8]
            updated = (dt(days_back=(i % 60), hours_back=(i % 24))).isoformat(sep=" ")
            rows.append((c_id(i), cats, brands, budget, note, updated))
        write_csv(path, header, rows, mode="w" if first else "a")
        first = False


def generate_orders(out: Path, n_orders: int, n_customers: int, rng: random.Random, chunk_size: int = 100_000):
    header = ["order_id", "customer_id", "order_date", "order_status", "channel", "subtotal", "shipping_fee", "tax_amount", "total_amount"]
    path = out / "orders.csv"
    first = True
    for start, end in chunked_range(n_orders, chunk_size):
        rows = []
        for i in range(start, end):
            cust = c_id(i % n_customers)
            order_date = (dt(days_back=(i % 365), hours_back=(i % 24))).isoformat(sep=" ")
            status = stable_choice(ORDER_STATUSES, i)
            channel = stable_choice(CHANNELS, i * 2 + 1)
            subtotal = round(20 + ((i * 13) % 480) + ((i % 100) / 10), 2)
            shipping = round(3 + (i % 15), 2)
            tax = round(subtotal * 0.08, 2)
            total = round(subtotal + shipping + tax, 2)
            rows.append((o_id(i), cust, order_date, status, channel, subtotal, shipping, tax, total))
        write_csv(path, header, rows, mode="w" if first else "a")
        first = False


def generate_order_items(out: Path, n_orders: int, n_items: int, n_products: int, n_variants: int, chunk_size: int = 100_000):
    header = ["order_item_id", "order_id", "customer_id", "product_id", "variant_id", "quantity", "unit_price", "discount_amount", "item_status"]
    path = out / "order_items.csv"
    first = True
    for start, end in chunked_range(n_items, chunk_size):
        rows = []
        for i in range(start, end):
            order_idx = i % n_orders
            customer_idx = order_idx % 500_000
            product_idx = i % n_products
            variant_idx = i % n_variants
            quantity = 1 + (i % 3)
            unit_price = round(10 + ((product_idx * 7) % 490) + ((i % 100) / 100), 2)
            discount = round((i % 20) * 0.5, 2)
            item_status = stable_choice(ITEM_STATUSES, i)
            rows.append((oi_id(i), o_id(order_idx), c_id(customer_idx), p_id(product_idx), v_id(variant_idx), quantity, unit_price, discount, item_status))
        write_csv(path, header, rows, mode="w" if first else "a")
        first = False


def generate_inventory(out: Path, n_variants: int, n_products: int, n_snapshots: int, chunk_size: int = 100_000):
    header = ["snapshot_id", "warehouse_id", "product_id", "variant_id", "snapshot_date", "stock_qty", "reserved_qty", "available_qty", "backorder_qty", "restock_eta_date"]
    path = out / "inventory_snapshots.csv"
    first = True
    for start, end in chunked_range(n_snapshots, chunk_size):
        rows = []
        for i in range(start, end):
            variant_idx = i % n_variants
            product_idx = variant_idx % n_products
            warehouse_id = WAREHOUSES[i % len(WAREHOUSES)][0]
            snapshot_date = (date(2026, 1, 1) + timedelta(days=i % 181)).isoformat()
            stock = (i * 11) % 250
            reserved = i % 30
            available = max(stock - reserved, 0)
            backorder = 0 if available > 0 else (i % 5)
            restock_eta = (date(2026, 1, 1) + timedelta(days=(i % 14) + 1)).isoformat() if available == 0 else ""
            rows.append((s_id(i), warehouse_id, p_id(product_idx), v_id(variant_idx), snapshot_date, stock, reserved, available, backorder, restock_eta))
        write_csv(path, header, rows, mode="w" if first else "a")
        first = False


def generate_tracking(out: Path, n_orders: int, chunk_size: int = 100_000):
    header = ["order_id", "carrier", "tracking_status", "shipped_date", "expected_delivery_date", "delivered_date", "last_event_at"]
    path = out / "order_tracking.csv"
    first = True
    for start, end in chunked_range(n_orders, chunk_size):
        rows = []
        for i in range(start, end):
            status = stable_choice(TRACKING_STATUSES, i)
            shipped = (date(2026, 1, 1) + timedelta(days=i % 120)).isoformat() if status != "label_created" else ""
            expected = (date(2026, 1, 1) + timedelta(days=(i % 120) + 5)).isoformat()
            delivered = (date(2026, 1, 1) + timedelta(days=(i % 120) + 7)).isoformat() if status == "delivered" else ""
            last_event = (dt(days_back=(i % 120), hours_back=(i % 24))).isoformat(sep=" ")
            rows.append((o_id(i), stable_choice(CARRIERS, i), status, shipped, expected, delivered, last_event))
        write_csv(path, header, rows, mode="w" if first else "a")
        first = False


def generate_reviews(out: Path, n_reviews: int, n_order_items: int, chunk_size: int = 100_000):
    header = ["review_id", "order_item_id", "order_id", "customer_id", "product_id", "rating", "review_title", "review_body", "review_date", "verified_purchase"]
    path = out / "reviews.csv"
    first = True
    for start, end in chunked_range(n_reviews, chunk_size):
        rows = []
        for i in range(start, end):
            order_item_idx = i % n_order_items
            order_idx = order_item_idx // 3
            customer_idx = order_idx % 500_000
            product_idx = order_item_idx % 1_000_000
            rating = 1 + (i % 5)
            title = ["Great fit", "Good value", "Solid quality", "Nice style", "Works well"][i % 5]
            body = f"Synthetic review {i} for product {product_idx:07d}. This item matched the stated use case and budget constraints."
            review_date = (date(2026, 1, 1) + timedelta(days=i % 365)).isoformat()
            rows.append((r_id(i), oi_id(order_item_idx), o_id(order_idx), c_id(customer_idx), p_id(product_idx), rating, title, body, review_date, True))
        write_csv(path, header, rows, mode="w" if first else "a")
        first = False


def generate_events(out: Path, n_events: int, n_customers: int, n_products: int, n_variants: int, chunk_size: int = 100_000):
    header = ["event_id", "customer_id", "session_id", "event_type", "query_text", "product_id", "variant_id", "event_ts", "dwell_ms"]
    path = out / "behavior_events.csv"
    first = True
    query_templates = [
        "waterproof jacket under 80",
        "cheap hiking boots for cold weather",
        "similar but cheaper",
        "does it come in green",
        "size M in stock",
        "order tracking last Tuesday",
        "budget hiking gear",
        "lightweight backpack",
    ]
    for start, end in chunked_range(n_events, chunk_size):
        rows = []
        for i in range(start, end):
            cust = c_id(i % n_customers)
            session = f"S{i // 8:010d}"
            etype = stable_choice(EVENT_TYPES, i)
            query = query_templates[i % len(query_templates)] if etype in {"search", "view"} else ""
            product_idx = i % n_products if etype != "purchase" else (i * 3) % n_products
            variant_idx = i % n_variants if etype in {"click", "add_to_cart", "purchase"} else ""
            event_ts = (dt(days_back=(i % 90), hours_back=(i % 24))).isoformat(sep=" ")
            dwell = 500 + (i % 120000)
            rows.append((e_id(i), cust, session, etype, query, p_id(product_idx), v_id(int(variant_idx)) if variant_idx != "" else "", event_ts, dwell))
        write_csv(path, header, rows, mode="w" if first else "a")
        first = False


def generate_tool_logs(out: Path, n_logs: int, n_customers: int, chunk_size: int = 100_000):
    header = ["call_id", "customer_id", "session_id", "tool_name", "success", "latency_ms", "error_type", "called_at", "payload_summary"]
    path = out / "tool_call_logs.csv"
    first = True
    errors = ["timeout", "rate_limited", "upstream_5xx", "invalid_request", "none"]
    for start, end in chunked_range(n_logs, chunk_size):
        rows = []
        for i in range(start, end):
            success = (i % 13) != 0
            err = "" if success else stable_choice(errors[:-1], i)
            rows.append((t_id(i), c_id(i % n_customers), f"S{i // 5:010d}", stable_choice(TOOL_NAMES, i), success, 80 + (i % 1200), err, (dt(days_back=(i % 14), hours_back=(i % 24))).isoformat(sep=" "), f"{{'table':'{stable_choice(TOOL_NAMES, i)}'}}"))
        write_csv(path, header, rows, mode="w" if first else "a")
        first = False


def scale_counts(counts: dict, scale: float) -> dict:
    if scale <= 0:
        raise ValueError("scale must be positive")
    if scale == 1:
        return counts
    scaled = {}
    for k, v in counts.items():
        if k in {"categories", "brands", "warehouses"}:
            scaled[k] = v
        else:
            scaled[k] = max(1, int(math.ceil(v * scale)))
    return scaled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--format", choices=["csv"], default="csv")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--counts", default=None, help="Optional JSON counts file")
    args = parser.parse_args()

    out = Path(args.out)
    ensure_dir(out)
    rng = random.Random(args.seed)

    if args.counts:
        with open(args.counts, "r", encoding="utf-8") as f:
            counts = json.load(f)
    else:
        with open(Path(__file__).with_name("counts.json"), "r", encoding="utf-8") as f:
            counts = json.load(f)

    counts = scale_counts(counts, args.scale)

    generate_categories(out)
    generate_brands(out)
    generate_warehouses(out)
    generate_products(out, counts["products"], rng)
    generate_variants(out, counts["products"], counts["product_variants"], rng)
    generate_customers(out, counts["customers"], rng)
    generate_preferences(out, counts["customer_preferences"])
    generate_orders(out, counts["orders"], counts["customers"], rng)
    generate_order_items(out, counts["orders"], counts["order_items"], counts["products"], counts["product_variants"])
    generate_inventory(out, counts["product_variants"], counts["products"], counts["inventory_snapshots"])
    generate_tracking(out, counts["orders"])
    generate_reviews(out, counts["reviews"], counts["order_items"])
    generate_events(out, counts["behavior_events"], counts["customers"], counts["products"], counts["product_variants"])
    generate_tool_logs(out, counts["tool_call_logs"], counts["customers"])

    print(json.dumps({"output": str(out.resolve()), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
