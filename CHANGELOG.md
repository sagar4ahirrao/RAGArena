# Changelog

All notable changes to RagArena are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); versioning: [SemVer](https://semver.org/).

## [0.1.0] — 2026-08-25

First public release. ⚡

### Added
- **Unified provider routing** (`router.py`) — `completion()`, `embedding()`, `rerank()`
  addressed as `provider/model` (LiteLLM-style) across OpenAI, Anthropic, Google, Azure,
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
- **CLI** — `RagArena serve | models list|providers | strategies | run | compare`.
- **Web dashboard** (`api/dashboard.html`) — overview stats, visual experiment builder,
  comparison leaderboards with Chart.js, run inspector, searchable model catalog.
- **REST API** — `POST /api/evaluate`, `POST /api/compare`, `GET /api/catalog`,
  `GET /api/runs/{id}`, `GET /health`.
- Examples, contributing guide, MIT license.

[0.1.0]: https://github.com/RagArena/RagArena/releases/tag/v0.1.0
