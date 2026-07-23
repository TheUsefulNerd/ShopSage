# ShopSage 🛍️ — AI Shopping Assistant

ShopSage is an AI-powered retail assistant demonstrating production-grade architecture: **Hybrid Vector Search**, **Live Inventory checks**, and **Cross-Session Memory**.

## Architecture & Tech Stack

This project was upgraded in Week 2 to match industry best practices:

*   **Frontend**: [Streamlit](https://streamlit.io/) — Retail-oriented UI with scenario buttons, memory display, and a modern aesthetic.
*   **LLM Engine**: Groq (`llama-3.3-70b-versatile`) — Lightning fast intent extraction and response generation.
*   **Vector Database**: [Qdrant Cloud](https://qdrant.tech/) — Cloud-hosted vector database.
*   **Hybrid Search**: Dense semantic search (FastEmbed `all-MiniLM-L6-v2`) + Sparse exact-match search (FastEmbed `BM25`), fused server-side with Reciprocal Rank Fusion (RRF).
*   **Persistent Memory**: Supabase `user_preferences` table tracks budget and category interests across sessions.
*   **Live Data**: Supabase Postgres for real-time inventory and pricing lookup.

## Core Features

1.  **Guaranteed Constraints**: The agent never hallucinates products out of budget, out of stock, or in the wrong size/color. This is enforced via a deterministic filtering layer applied *after* RRF retrieval and *before* the LLM sees the candidates.
2.  **Two-Layer Memory**:
    *   *Short-term (Session)*: Tracks the exact products shown so users can say "I'll take the red one" without repeating the brand.
    *   *Long-term (Supabase)*: Tracks the user's budget and category preferences across sessions, greeting them dynamically upon return.
3.  **Eval Pipeline**: Includes an automated evaluation script (`scripts/eval/run_eval.py`) that tests the retrieval layer against 100+ queries (budget constraints, rating checks, fuzzy matching, out-of-scope handling).

## Setup & Running

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Setup**:
    Add the following to your `.env` file:
    ```env
    GROQ_API_KEY=your_key
    SUPABASE_URL=your_url
    SUPABASE_KEY=your_key
    QDRANT_URL=your_qdrant_cloud_url
    QDRANT_API=your_qdrant_api_key
    ```

3.  **Build the Vector Index**:
    This pulls the 250 products from Supabase, embeds them (Dense + Sparse), and upserts them to Qdrant Cloud.
    ```bash
    python -m scripts.rag.ingestion
    ```
    *(Note: The embeddings run entirely locally using FastEmbed ONNX — no external embedding API tokens required).*

4.  **Run the App**:
    ```bash
    streamlit run scripts/rag/streamlit_app.py
    ```

5.  **Run the Evaluation**:
    ```bash
    python -m scripts.eval.run_eval
    ```

## Deployment

Because this project uses Qdrant Cloud (for vector storage) and Supabase (for SQL and memory), it is fully stateless. You can deploy this repository instantly to **Streamlit Community Cloud** (share.streamlit.io) for free by connecting your GitHub account. Just add your `.env` variables to the Streamlit Cloud dashboard.
