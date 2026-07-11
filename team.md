# ShopSage — Team & Stack

## Roles

| Name | Primary Role(s) | Week 1 Tasks |
|---|---|---|
| Advait Joshi | Repo/infra, RAG ingestion pipeline, Gradio UI, wiring | Task 2, 6, 8 (with Satish), 9 |
| Kasturi | Synthetic dataset, system prompt | Task 3, 4 (with Satish's guidance) |
| Satish Gauraha | Dataset guidance, product catalog | Task 4 (guidance), Task 5 |

> Roles will expand naturally in later weeks (tools/MCP, memory, guardrails/caching, observability — per tasks.md role split in Task 1). Update this table as ownership solidifies.

## Read Confirmation

Each member confirms they have read `requirements.md` in full before starting work:

- [ ] Advait Joshi
- [ ] Kasturi
- [ ] Satish Gauraha

## Tech Stack

### Week 1 — Foundations, RAG & UI
| Component | Choice | Status |
|---|---|---|
| Language | Python | Locked |
| LLM inference | Groq | Locked |
| Orchestration | LangChain | Locked |
| Vector store | Chroma **or** Qdrant | **TBD — owner: whoever builds Task 6 (ingestion pipeline), decide before that task starts** |
| Embedding model | Not yet decided | **TBD — owner: same as above; needed before Task 6** |
| Dataset storage | Supabase | Locked |
| Data cleaning | Pandas | Locked |
| Backend | FastAPI | Locked |
| Chat UI | Gradio | Locked |

### Week 2 — Tools, MCP & Memory
| Component | Choice | Status |
|---|---|---|
| Tool exposure | MCP | Locked |
| Memory store/approach | Not yet decided | **TBD — needed before Task 14 (memory schema design)** |

### Week 3 — Guardrails & Caching
| Component | Choice | Status |
|---|---|---|
| Guardrails | Manual/custom checks (no NeMo Guardrails) | Locked |
| Cache | Redis | Locked |

### Week 4 — Observability & Evals
| Component | Choice | Status |
|---|---|---|
| Eval harness | RAGAS + custom precision/recall/faithfulness checks | Locked |
| Tracing/observability | LangSmith **or** OpenTelemetry | **TBD — owner: whoever owns observability, decide before Task 24** |

## Open Decisions Log
Keep this list current — anything marked TBD above should move here once resolved, with the decision and reasoning, so Week 4 error-analysis has context on why a choice was made.

- [ ] Vector store: Chroma vs Qdrant
- [ ] Embedding model
- [ ] Memory storage approach
- [ ] LangSmith vs OpenTelemetry
