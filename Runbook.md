# ShopSage Data Pipeline — Runbook

This documents the full flow for generating synthetic retail data, enriching it
with an LLM, and loading it into Supabase.

## Prerequisites

```powershell
pip install pandas python-dotenv langchain-core langchain-groq supabase
```

Add these to your `.env` file at the project root:

```
GROQ_API_KEY=your_groq_key
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_KEY=your_supabase_secret_key   # sb_secret_... or legacy service_role JWT — NOT anon/publishable
```

All scripts below auto-load `.env` — no need to manually `$env:...` anything.

Scripts referenced here live in `scripts\`:
`generate.py`, `llm_common.py`, `enrich_descriptions.py`, `enrich_reviews.py`,
`load_to_supabase.py`. `schema.sql` lives at the project root.

---

## Step 1 — Define how much data to generate

Create/edit `scripts\counts_test.json`:

```json
{
  "products": 200,
  "product_variants": 400,
  "customers": 100,
  "customer_preferences": 100,
  "orders": 300,
  "order_items": 500,
  "inventory_snapshots": 400,
  "reviews": 250,
  "behavior_events": 50,
  "tool_call_logs": 50
}
```

This is a "medium tier" size — big enough to surface real issues (uneven
category spread, edge cases) but cheap/fast enough to regenerate repeatedly
while iterating. Scale up once everything downstream checks out.

## Step 2 — Generate the raw CSVs

No API key needed for this step — pure offline generation.

```powershell
python scripts\generate.py --out .\data\test_output --counts scripts\counts_test.json
```

Produces `categories.csv`, `brands.csv`, `warehouses.csv`, `products.csv`,
`product_variants.csv`, `customers.csv`, `orders.csv`, `order_items.csv`,
`inventory_snapshots.csv`, `order_tracking.csv`, `reviews.csv` in
`data\test_output\`.

**To start completely fresh** (wipes CSVs and any enrichment checkpoints):

```powershell
Remove-Item .\data\test_output -Recurse -Force
```

## Step 3 — Enrich product descriptions with an LLM

```powershell
python scripts\enrich_descriptions.py --out .\data\test_output --llm-model llama-3.1-8b-instant --llm-concurrency 4 --llm-min-interval 4 --llm-quiet
```

## Step 4 — Enrich review text with an LLM

```powershell
python scripts\enrich_reviews.py --out .\data\test_output --llm-model llama-3.1-8b-instant --llm-concurrency 4 --llm-min-interval 4 --llm-quiet
```

### Useful flags for steps 3 & 4

| Flag | Purpose |
|---|---|
| `--limit N` | Only enrich the first N rows — use this to test cheaply before a full run |
| `--llm-quiet` | Suppress per-row logs, keep only the summary line |
| `--llm-concurrency N` | Parallel calls (start at 4, raise if you have rate-limit headroom) |
| `--llm-min-interval S` | Minimum seconds between call starts, shared across all workers — the main lever for staying under a provider's tokens-per-minute limit |
| `--llm-fresh` | Ignore any existing checkpoint and start that step over from scratch |
| `--llm-provider` | `groq` (default), `anthropic`, or `openai` |
| `--llm-model` | Model name (see rate-limit notes below) |

### If you hit a rate-limit error (`429`, `rate_limit_exceeded`)

Both scripts already retry failed calls with backoff automatically, and any
row that still fails after 5 attempts is skipped (keeps its original
templated text) rather than crashing the whole run — safe to just re-run
the same command afterward; it resumes from checkpoint.

To avoid hitting the limit in the first place, calculate a safe
`--llm-min-interval`:

```
safe interval (seconds) = 60 / (your TPM limit / tokens per call)
```

Example: a 6000 TPM limit with ~350 tokens/call → ~17 calls/min sustainable
→ `--llm-min-interval 4` gives a safe margin.

Check your exact limits in the error message itself, or at
console.groq.com/settings/limits.

### Resuming / re-running

Progress is checkpointed incrementally to
`data\test_output\.descriptions_llm_checkpoint.jsonl` and
`.reviews_llm_checkpoint.jsonl` as each row succeeds. Re-running the exact
same command after an interruption (crash, Ctrl+C, rate limit) picks up
only the remaining rows — it won't redo or lose completed work.

## Step 5 — Create the Supabase tables (once)

Open your Supabase project → **SQL Editor** → New query → paste the entire
contents of `schema.sql` → Run. Creates all 11 tables with the correct
columns and foreign keys, in dependency order.

## Step 6 — Load the CSVs into Supabase

```powershell
python scripts\load_to_supabase.py --data-dir .\data\test_output
```

This first prints a mapping report (which CSV maps to which table, and
flags anything missing or unrecognized) and asks for a `y/N` confirmation
before uploading anything — showing your actual Supabase project URL so you
can catch a wrong `.env` before it matters. Add `--yes` to skip the prompt
for non-interactive runs.

Re-running this is safe — it uses `upsert` (insert-or-update) keyed on each
table's primary key, so already-loaded rows just get updated, not duplicated.

---

## Full command sequence (copy-paste, medium-tier run)

```powershell
python scripts\generate.py --out .\data\test_output --counts scripts\counts_test.json

python scripts\enrich_descriptions.py --out .\data\test_output --llm-model llama-3.1-8b-instant --llm-concurrency 4 --llm-min-interval 4 --llm-quiet

python scripts\enrich_reviews.py --out .\data\test_output --llm-model llama-3.1-8b-instant --llm-concurrency 4 --llm-min-interval 4 --llm-quiet

python scripts\load_to_supabase.py --data-dir .\data\test_output
```

(Run `schema.sql` in the Supabase SQL Editor once, before the first `load_to_supabase.py` run.)