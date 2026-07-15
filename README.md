# ShopSage

A shopping assistant that helps users discover and compare products via RAG, checks live inventory/order status via tools, and remembers a shopper's style and budget preferences across sessions — while never recommending out-of-stock or age-restricted items.

Built for [requirements.md](./requirements.md) — see [tasks.md](./tasks.md) for the 4-week build plan and [docs/team.md](./docs/team.md) for roles and stack decisions.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the full branching, commit, and PR workflow before opening a pull request.

## Status

🚧 Week 1 in progress. This README will be updated as each week's demo goal lands.

## Tech Stack

- **Language:** Python
- **LLM inference:** Groq
- **Orchestration:** LangChain
- **Vector store:** Chroma or Qdrant *(TBD — see docs/team.md)*
- **Dataset storage:** Supabase
- **Backend:** FastAPI
- **Chat UI:** Gradio
- **Tools:** MCP
- **Cache:** Redis
- **Evals:** RAGAS + custom checks
- **Observability:** LangSmith or OpenTelemetry *(TBD — see docs/team.md)*

Full breakdown in [docs/team.md](./docs/team.md).

## Prerequisites

- Python 3.11+ (requirements.txt was resolved against 3.11 on Windows — older versions may fail)
- A Groq API key
- A Supabase project (URL + service key)
- Redis (local install or Docker) — needed from Week 3 onward, not required for Week 1

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd ShopSage

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies (versions are pinned — do not upgrade individually without team discussion)
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# then fill in .env with your Groq API key, Supabase URL/key, etc.

# 5. Run the Gradio chat UI (builds the vector index on first run — takes ~1 min)
python -m scripts.rag.app

# Optional: get a public share link for teammates to test without installing anything
python -m scripts.rag.app --share

# Optional: save the Chroma index to disk so it survives restarts
python -m scripts.rag.app --persist-path .chroma
```

The Gradio UI will start at `http://localhost:7860`. On first run it pulls all 500 products from
Supabase, embeds them with `all-MiniLM-L6-v2`, and loads them into Chroma. Subsequent runs with
`--persist-path` skip the embedding step.

## Week 1 Demo

The Week 1 RAG prototype supports natural-language product queries with budget and age guardrails:

```
"Show me smart plugs to control my home appliances remotely."
"I need hiking boots for a cold weather trip."
"Sleeping bag for camping under $150."
"What fitness gear do you have under $100?"
```

Budget filtering and age-restriction checks are **deterministic** (done in Python before the LLM
sees any candidates) — the LLM only writes the human-readable recommendation over the pre-filtered
shortlist. No tool calls yet — those are Week 2.

## Project Structure

```
ShopSage/
├── docs/
│   ├── team.md              # roles, stack decisions, open TBDs
│   ├── tools.md             # tool specs (Week 2+)
│   └── guardrails.md        # guardrail rules (Week 3+)
├── scripts/
│   ├── data_gen/            # synthetic dataset generation (Kasturi's branch, merged)
│   └── rag/
│       ├── ingestion.py     # Supabase → embed → Chroma
│       ├── retrieval.py     # retrieve(), extract_budget(), get_shortlist()
│       └── app.py           # Gradio chat UI entrypoint
├── notebook/
│   └── ShopSage_Week1.ipynb # Week 1 RAG prototype (reference — modular code is in scripts/rag/)
├── requirements.txt
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── requirements.md          # project spec (do not edit)
├── tasks.md                 # 4-week task plan (do not edit)
└── README.md
```

> `.env`, `.chroma/`, and `.gradio/` are gitignored — never commit secrets or local indexes.

## Team

See [docs/team.md](./docs/team.md) for roles and read-confirmation.

## Environment Variables

See `.env.example` for the full list. At minimum for Week 1 you'll need:

```
GROQ_API_KEY=
SUPABASE_URL=
SUPABASE_KEY=
```
