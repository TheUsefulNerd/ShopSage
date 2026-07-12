#!/usr/bin/env python3
"""Load the CSVs produced by generate.py into Supabase tables via the
supabase-py client.

Prerequisites:
    1. Run schema.sql once in the Supabase SQL Editor to create the tables.
    2. pip install supabase pandas python-dotenv
    3. Have SUPABASE_URL and SUPABASE_KEY set (in your .env or the
       environment). Use a secret-privilege key — the newer sb_secret_...
       key, or a legacy service_role JWT — not the anon/publishable key, so
       this bulk load isn't blocked by any Row Level Security policies you
       may add later for your app's normal read/write paths.

Usage:
    python load_to_supabase.py --data-dir ./data/test_output
    python load_to_supabase.py --data-dir ./data/test_output --tables products reviews
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from supabase import create_client, Client

# Load order matters: parent tables must be loaded before any child table
# that has a foreign key referencing them.
TABLE_LOAD_ORDER = [
    "categories",
    "brands",
    "warehouses",
    "products",
    "product_variants",
    "customers",
    "orders",
    "order_items",
    "inventory_snapshots",
    "order_tracking",
    "reviews",
]

PRIMARY_KEYS = {
    "categories": "category_id",
    "brands": "brand_id",
    "warehouses": "warehouse_id",
    "products": "product_id",
    "product_variants": "variant_id",
    "customers": "customer_id",
    "orders": "order_id",
    "order_items": "order_item_id",
    "inventory_snapshots": "snapshot_id",
    "order_tracking": "order_id",
    "reviews": "review_id",
}


def load_csv_records(csv_path: Path) -> list[dict]:
    """Read a CSV with pandas and convert to a list of JSON-safe dicts.
    Round-tripping through JSON converts NaN -> None and numpy types
    (int64/float64/bool_) -> native Python types, both of which the
    supabase-py client needs for its HTTP request body."""
    df = pd.read_csv(csv_path)
    return json.loads(df.to_json(orient="records"))


def upload_table(supabase: Client, table: str, records: list[dict], batch_size: int) -> None:
    if not records:
        print(f"  {table}: 0 rows, skipping.")
        return
    pk = PRIMARY_KEYS[table]
    total = len(records)
    for start in range(0, total, batch_size):
        batch = records[start:start + batch_size]
        supabase.table(table).upsert(batch, on_conflict=pk).execute()
        print(f"  {table}: {min(start + batch_size, total)}/{total} rows uploaded")


def report_mapping(data_dir: Path, tables_to_load: list[str]) -> list[str]:
    """Print the CSV-file -> table mapping explicitly before touching the
    network, so it's never a silent assumption. Returns the subset of
    tables_to_load that actually have a matching CSV on disk (in order)."""
    csvs_found = {p.stem for p in data_dir.glob("*.csv")}

    print("=== CSV -> Supabase table mapping ===")
    ready = []
    for table in tables_to_load:
        if table not in PRIMARY_KEYS:
            print(f"  ??? {table:24s} -> not a known table (expected one of {sorted(PRIMARY_KEYS)}); skipping")
            continue
        csv_path = data_dir / f"{table}.csv"
        if csv_path.exists():
            print(f"  OK  {csv_path.name:24s} -> {table}")
            ready.append(table)
        else:
            print(f"  --  {table + '.csv':24s} -> {table}  (file not found in {data_dir}, will skip)")

    unrecognized = csvs_found - set(PRIMARY_KEYS)
    if unrecognized:
        print(f"  Note: found extra CSV(s) in {data_dir} with no matching table, ignored: {sorted(unrecognized)}")
    print()
    return ready


def main() -> None:
    parser = argparse.ArgumentParser(description="Load generate.py's output CSVs into Supabase.")
    parser.add_argument("--data-dir", required=True, help="Folder containing the generated CSVs (the --out folder from generate.py).")
    parser.add_argument("--batch-size", type=int, default=500, help="Rows per upsert request.")
    parser.add_argument("--tables", nargs="*", default=None,
                         help="Only load these tables (space-separated). Default: all, in FK-safe order.")
    parser.add_argument("--yes", action="store_true",
                         help="Skip the confirmation prompt after the mapping is shown.")
    args = parser.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise SystemExit(
            "Missing SUPABASE_URL and/or SUPABASE_KEY.\n"
            "Set both in your .env file or as environment variables before running this script.\n"
            "Use a secret-privilege key (the newer sb_secret_... key, or a legacy service_role JWT), "
            "not the anon/publishable key, so this bulk load isn't blocked by Row Level Security."
        )

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"--data-dir {data_dir} does not exist.")

    tables_to_load = args.tables or TABLE_LOAD_ORDER
    ready_tables = report_mapping(data_dir, tables_to_load)

    if not ready_tables:
        raise SystemExit("Nothing to load — no matching CSVs found. Check --data-dir.")

    if not args.yes:
        answer = input(f"Load {len(ready_tables)} table(s) into Supabase project at {url}? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    supabase: Client = create_client(url, key)

    for table in ready_tables:
        csv_path = data_dir / f"{table}.csv"
        print(f"Loading {table} from {csv_path} ...")
        records = load_csv_records(csv_path)
        upload_table(supabase, table, records, batch_size=args.batch_size)

    print("Done.")


if __name__ == "__main__":
    main()