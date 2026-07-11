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

- Python 3.12+ (requirements.txt pins were resolved against 3.12 — older versions may fail to resolve)
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

# 5. Run the app
python app.py
```

The Gradio UI will start locally and print a shareable link in the terminal.

## Project Structure

```
ShopSage/
├── docs/
│   ├── team.md          # roles, stack decisions, open TBDs
│   ├── tools.md          # tool specs (Week 2+)
│   └── guardrails.md     # guardrail rules (Week 3+)
├── data/
│   └── ...               # synthetic dataset, product catalog
├── src/
│   └── ...               # ingestion, retrieval, tools, memory, guardrails
├── app.py                 # Gradio entrypoint
├── requirements.txt
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── requirements.md        # project spec (do not edit)
├── tasks.md               # 4-week task plan (do not edit)
└── README.md
```

> Structure will fill in as tasks land — this is the target shape, not everything exists yet.

## Team

See [docs/team.md](./docs/team.md) for roles and read-confirmation.

## Environment Variables

See `.env.example` for the full list. At minimum for Week 1 you'll need:

```
GROQ_API_KEY=
SUPABASE_URL=
SUPABASE_KEY=
```
