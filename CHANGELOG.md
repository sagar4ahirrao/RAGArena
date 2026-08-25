# Changelog

All notable changes to RagArena are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); versioning: [SemVer](https://semver.org/).

## [0.2.1] — 2026-08-25

Provider coverage + playground groundwork.

### Added
- **Azure AI Foundry provider** (`azure_foundry`) — calls the `models/chat/completions`
  REST endpoint with a Bearer token; verified working with `Llama-4-Maverick-17B-128E-Instruct-FP8`
  and `DeepSeek-V3.2`.
- **Azure OpenAI native handler** (`azure`) via the official `AzureOpenAI` client.
- **Current free-model entries**: `google/gemini-2.5-flash`, `groq/qwen/qwen3.6-27b`,
  `groq/openai/gpt-oss-20b`, `google/gemini-embedding-001`.
- `examples/03_provider_test.py` — runs all 18 strategies across varied datasets.

### Fixed
- Google base URL now points at the OpenAI-compatible `/v1beta/openai` endpoint.
- Removed deprecated `text-embedding-004` (replaced by `gemini-embedding-001`).

## [0.2.0] — 2026-08-25

Graph + multimodal retrieval, inspired by LightRAG and RAGAnything.

### Added
- **Graph RAG dual-level retrieval** (`graph.py`) — `GraphIndex` extracts entities per
  chunk, clusters chunks sharing entities into communities, and supports two retrieval
  levels:
  - **local** — entity-precise retrieval for "who/what" factual lookups.
  - **global** — macro-theme retrieval: summarise communities, rank by relevance,
    synthesise a cross-document answer for "how/why" questions.
  - **hybrid** / **mix** — combine both levels. The graph is built lazily and cached on
    the shared index; entity extraction degrades to a deterministic keyword fallback when
    the LLM is unavailable.
- **5 new strategies** — `graph_local`, `graph_global`, `graph_hybrid`, `graph_mix`,
  and `multimodal`. Total strategies: **18**.
- **Multimodal ingestion** (`index.py`) — `MultimodalDocument` and a `doc_type` field
  preserve tables / images / equations intact (no sentence-splitting) and tag them so the
  `multimodal` strategy retrieves and labels each content type (`TABLE`, `IMAGE`,
  `EQUATION`, `TEXT`).
- Example `examples/02_graph_multimodal.py`.

## [0.1.0] — 2026-08-25

First public release. ⚡

### Added
- **Unified provider routing** (`router.py`) — `completion()`, `embedding()`, `rerank()`
  addressed as `provider/model` across OpenAI, Anthropic, Google, Azure,
  Bedrock, Cohere, Mistral, xAI, DeepSeek, Groq, Together, Fireworks, DeepInfra,
  Perplexity, OpenRouter, NVIDIA NIM, Anyscale, AI21, HuggingFace, Ollama, vLLM, LM Studio.
- **Model catalog** (`catalog.py`) — 76 curated models across 18 chat/embedding/rerank
  providers with context windows and per-1M-token pricing; cost estimator built in.
- **13 RAG strategies** (`strategies.py`) — naive, hybrid (dense+BM25), multi-query,
  rag_fusion (RRF), hyde, rerank, compression, crag, self_rag, decomposition,
  step_back, agentic, flare.
- **10 evaluation metrics** (`metrics.py`) — context precision/recall, hit rate, MRR,
  LLM-judge faithfulness / answer relevance / answer correctness, latency, cost,
  token usage. Presets: quick · quality · full · production.
- **Zero-config VectorIndex** (`index.py`) — recursive chunking + embedding + cosine search;
  shared-index mode makes comparisons fast & cheap.
- **Evaluation engine** (`engine.py`) — `evaluate()` single-run reports and
  `compare()` head-to-head leaderboards with per-sample drill-down.
- **CLI** — `ragarena serve | models list|providers | strategies | run | compare`.
- **Web dashboard** (`api/dashboard.html`) — overview stats, visual experiment builder,
  comparison leaderboards with Chart.js, run inspector, searchable model catalog.
- **REST API** — `POST /api/evaluate`, `POST /api/compare`, `GET /api/catalog`,
  `GET /api/runs/{id}`, `GET /health`.
- Examples, contributing guide, MIT license.

[0.2.0]: https://github.com/sagar4ahirrao/ragarena/releases/tag/v0.2.0
[0.1.0]: https://github.com/sagar4ahirrao/ragarena/releases/tag/v0.1.0
