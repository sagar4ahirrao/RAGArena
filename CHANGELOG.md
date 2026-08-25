# Changelog

All notable changes to RagArena are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); versioning: [SemVer](https://semver.org/).

## [0.2.2] — 2026-08-25

Professional Next.js playground, multi-format ingestion, benchmark datasets, strategy
recommendation, and a round of live-provider bugfixes.

### Added
- **Next.js 14 playground UI** (`web/`, static-export, served by FastAPI as `ui_dist/`) —
  Overview, Playground (upload/dataset/paste corpus → chunking → strategy/model picker →
  run → results table), Compare (multi-config leaderboard), Datasets browser, Model
  catalog, Runs. Dark, LiteLLM-style dashboard; no Node required at runtime for installed
  packages.
- **Multi-format ingestion** (`ingest.py`) — pdf, docx, pptx, html, csv, json/jsonl, xlsx,
  md, txt, images → document dicts, with graceful degradation when optional parser libs
  are missing. New `ragarena[ingest]` extra.
- **Popular benchmark datasets** (`datasets.py`) — bundled offline sets (`capitals`,
  `rag_faq`) plus HuggingFace-backed loaders for `squad`, `hotpotqa`,
  `natural_questions`, `triviaqa`, `ms_marco`. New `ragarena[datasets]` extra.
- **`recommend_strategy()`** (`engine.py`) + `ragarena recommend` CLI + `POST
  /api/recommend` — runs every strategy (or a chosen subset) against the *same* corpus
  and questions and returns a composite-scored (quality/cost/latency-weighted)
  leaderboard with a single recommended best strategy for that dataset.
- `GET /api/env-status` — which providers have a usable API key in the current
  environment, for a quick "what can I actually run right now" check.
- `POST /api/upload` (multipart file → parsed documents) and `GET /api/datasets/{name}`.
- Auto-loads a project-local `.env` (via `python-dotenv`, `override=True` so it wins over
  stray same-named vars already in the shell).
- `Dockerfile` (multi-stage: Next.js static export + Python runtime) and GitHub Actions
  workflow to publish to Docker Hub + GHCR on tag push.

### Fixed
- **Windows console crash**: several `print()` calls used box-drawing/arrow Unicode
  (`▶ → ·`) that crash outright on the default Windows `cp1252` console encoding,
  breaking `ragarena strategies`, `ragarena models providers`, and `compare()` entirely
  on Windows. `cli.py`'s entrypoint now reconfigures stdout/stderr to UTF-8 with
  `errors="replace"`; `engine.py` gained an encoding-safe `_print()` used throughout.
- `rerank` strategy: the default cross-encoder id was passed to `sentence-transformers`
  including the `huggingface/` provider prefix (`huggingface/BAAI/bge-reranker-v2-m3`),
  which HuggingFace Hub rejects — now correctly strips the prefix before loading.
- `hyde` strategy: an empty/whitespace-only LLM-generated hypothetical answer was passed
  straight to `embedding()`, which some providers (e.g. Gemini) reject outright — now
  falls back to the original query when the draft is empty.
- Per-sample `cost_usd` metric was rounded to 4 decimal places, silently zeroing out
  real (sub-cent) costs in reports — cost now keeps 8 decimals of precision.
- `evaluate(chunk_size=..., chunk_overlap=...)` passed those kwargs straight into
  `VectorIndex()`, which doesn't accept them, raising `TypeError` on any call that set
  custom chunking — now builds a `TextChunker` and passes it via `chunker=`.
- `ingest.to_multimodal()` used a relative import (`from ..index import
  MultimodalDocument`) one level too high for its own package, silently disabling
  multimodal expansion; also fixed the constructor call to match
  `MultimodalDocument`'s actual `content`/`doc_type`/`metadata` fields.
- Retired Groq model ids (`llama-3.1-70b-versatile`, `llama-3.1-8b-instant`,
  `mixtral-8x7b-32768`) replaced with currently-live models
  (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.6-27b`, `compound`,
  `compound-mini`), verified against Groq's live `/v1/models`.
- `google/text-embedding-004` removed from the catalog (410/404 on the OpenAI-compatible
  embeddings path) — `google/gemini-embedding-001` is the working replacement.

All 18 strategies verified end-to-end against live providers (Groq generation, Google
Gemini embeddings) via `evaluate()`, `compare()`, and `recommend_strategy()`.

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

[0.2.2]: https://github.com/sagar4ahirrao/ragarena/releases/tag/v0.2.2
[0.2.0]: https://github.com/sagar4ahirrao/ragarena/releases/tag/v0.2.0
[0.1.0]: https://github.com/sagar4ahirrao/ragarena/releases/tag/v0.1.0
