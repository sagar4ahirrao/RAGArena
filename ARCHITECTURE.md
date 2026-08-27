# RagArena — Architecture & Feature Reference

This document describes how RagArena is put together: the module map, the request/data
flow through an evaluation, and the full feature surface. For install/usage, see
[README.md](README.md); for Docker, see [DOCKER.md](DOCKER.md); for the version history,
see [CHANGELOG.md](CHANGELOG.md).

## System overview

```
                                   ┌─────────────────────────────┐
                                   │   Next.js playground (web/)  │
                                   │  Overview·Playground·Compare │
                                   │  Recommend·Datasets·Catalog  │
                                   │           ·Runs              │
                                   └───────────────┬───────────────┘
                                                    │ static export (build-time)
                                                    ▼
┌──────────────┐   HTTP    ┌──────────────────────────────────────┐
│  CLI (cli.py) │◄─────────►│   FastAPI server (api/server.py)      │
│ ragarena run  │           │  /api/evaluate  /api/compare          │
│ compare       │           │  /api/recommend /api/datasets         │
│ recommend     │           │  /api/upload    /api/env-status       │
│ serve · models│           │  serves ui_dist/ (compiled UI)        │
└───────┬───────┘           └───────────────────┬────────────────────┘
        │                                        │
        └──────────────────┬─────────────────────┘
                            ▼
              ┌─────────────────────────────┐
              │   engine.py — evaluate() /   │
              │   compare() / recommend_     │
              │   strategy()                 │
              └───────┬─────────────┬─────────┘
                      │             │
        ┌─────────────▼───┐   ┌─────▼──────────────┐
        │  strategies.py    │   │   metrics.py        │
        │  18 RAG pipelines │   │  10 metrics, 4      │
        │  (naive→graph_mix)│   │  presets, LLM-judge  │
        └────────┬──────────┘   └─────────────────────┘
                 │
   ┌─────────────┼───────────────────────┐
   ▼             ▼                       ▼
┌──────────┐ ┌──────────┐        ┌───────────────┐
│ index.py │ │ graph.py │        │  router.py     │
│VectorIndex│ │GraphIndex│        │ completion()   │
│ chunking  │ │ entities/│        │ embedding()    │
│ + cosine  │ │communities│       │ rerank()       │
└──────────┘ └──────────┘        │ 20+ providers  │
                                  └───────────────┘
   ▲
   │ documents in
┌──┴────────────┐   ┌────────────────┐
│  ingest.py     │   │  datasets.py    │
│ pdf·docx·pptx  │   │ bundled +       │
│ html·csv·json  │   │ HuggingFace     │
│ xlsx·sql·      │   │ (squad,hotpotqa,│
│ sqlite·images  │   │ nq,triviaqa,    │
└────────────────┘   │ ms_marco)       │
                      └────────────────┘
```

## Module map

| Module | Responsibility |
| --- | --- |
| `catalog.py` | Model/provider registry — 100+ chat/embedding/rerank models across 20+ providers, with context windows and per-1M-token pricing; `estimate_cost()`. |
| `router.py` | Unified `completion()` / `embedding()` / `rerank()` addressed as `provider/model`. `completion()` is backed by LiteLLM for the actual provider call (auth + request shaping across 100+ providers), while ragarena still resolves API keys itself (clear `MissingAPIKeyError` UX) and computes cost from its own catalog pricing. `embedding()`/`rerank()` keep direct SDK integrations (Voyage, Cohere, local HuggingFace `sentence-transformers`). Both `completion()` and `embedding()` also accept a non-string `model` — a LangChain chat model / `Embeddings` object, or any plain callable — for bring-your-own-model use (see `_custom_model_completion` / `_custom_model_embedding`); cost/token accounting degrades gracefully since there's no catalog pricing for an arbitrary object. Auto-loads a project `.env`. |
| `index.py` | `VectorIndex` (zero-config chunk + embed + cosine search), `TextChunker` (recursive character splitter with overlap), `MultimodalDocument` (typed text/table/image/equation chunks). |
| `graph.py` | `GraphIndex` — lazy per-corpus knowledge graph: entity extraction per chunk (LLM-based, falls back to a deterministic keyword extractor), community clustering, local/global summarization. |
| `strategies.py` | 18 `Strategy` implementations, one `run(query, index, llm_model, embedding_model)` interface each — interchangeable inside `evaluate()`/`compare()`. |
| `metrics.py` | `BaseMetric` implementations (retrieval + LLM-judged + operational), `DEFAULT_METRIC_SETS` presets (`quick`/`quality`/`full`/`production`). |
| `engine.py` | `evaluate()` (one strategy, full report), `compare()` (N strategy×model configs on a *shared* index, leaderboard), `recommend_strategy()` (compares every strategy, returns a composite-scored ranking + a single recommendation), `answer()` (plug-and-play single-query answering, with `strategy="auto"` picking a strategy via `recommend_strategy()` once per document set and caching it), `diff_runs()` (per-metric delta between two `EvaluationReport`s, e.g. from `EvaluationReport.load()`). |
| `testgen.py` | `generate_testset()` / `generate_testset_detailed()` — samples chunks from source documents and has an LLM write grounded simple/reasoning/multi-passage questions + reference answers, shaped for `evaluate()`. |
| `testing.py` | `assert_metric()` / `assert_no_regression()` — pytest-style assertions over an `EvaluationReport`/`RunDiff`, for gating CI on RAG quality. |
| `ingest.py` | `parse_file()` / `parse_dir()` — multi-format document parsing (see Feature list below) into `{"text", "tables", "images", "metadata"}` dicts; `to_multimodal()` bridges into `MultimodalDocument`; `from_sql()` pulls rows from any SQLAlchemy-supported database. |
| `datasets.py` | `DATASET_REGISTRY` — bundled offline QA sets + HuggingFace-backed loaders for popular benchmarks. `load_dataset(name, n, use_bundled)`. |
| `cli.py` | `ragarena serve\|ui\|models\|strategies\|ask\|testgen\|diff\|run\|compare\|recommend`. |
| `api/server.py` | FastAPI app: catalog/strategy/metric/dataset endpoints, async job-based `/api/evaluate`, `/api/compare`, `/api/recommend` (submit → poll `/api/runs/{id}`), file upload, and serves the compiled Next.js UI (`api/ui_dist/`, falling back to the legacy `api/dashboard.html` if the UI wasn't built into the install). |
| `web/` | Next.js 14 (App Router, TypeScript, Tailwind) playground UI. Statically exported at build time (`NEXT_STATIC_EXPORT=1 next build`) — no Node needed at runtime; served by FastAPI as plain files. |

## Data flow through one evaluation

1. **Ingest** (optional): raw files/directories/SQL sources → `ingest.parse_file()` /
   `parse_dir()` / `from_sql()` → a list of `{"text", "metadata", "tables"?, "images"?}`
   document dicts. Or use `datasets.load_dataset()` for a ready-made corpus + questions.
2. **Index**: `VectorIndex(embedding_model=...)` chunks each document
   (`TextChunker`, configurable `chunk_size`/`chunk_overlap`), embeds every chunk via
   `router.embedding()`, and stores vectors for cosine search. Graph strategies
   additionally build a `GraphIndex` lazily from the same chunks, cached on the index.
3. **Retrieve + generate**: the chosen `Strategy.run(query, index, llm_model,
   embedding_model)` retrieves chunks (dense/hybrid/multi-query/graph/... depending on
   strategy) and calls `router.completion()` to generate an answer, returning a
   `StrategyResult` (answer, chunks, usage, latency).
4. **Score**: each `(question, answer, chunks, reference_answer)` sample is scored by
   every metric in the resolved preset. Retrieval metrics (`context_precision`,
   `context_recall`, `hit_rate`, `mrr`) score chunk relevance via embedding cosine
   similarity against the run's `embedding_model` when one is available (falling back to
   lexical keyword overlap otherwise; the semantic threshold is tunable via
   `RAGARENA_RELEVANCE_THRESHOLD`, default `0.55`) — no extra LLM call either way.
   LLM-judged metrics (`faithfulness`, `answer_relevance`, `answer_correctness`) call
   `judge_model` via `router.completion()` and parse a structured verdict; pass
   `judge_samples > 1` to average several independent judge calls (temperature jitter
   between samples) and report `score_stdev`/`all_scores`, reducing single-sample
   LLM-judge variance.
5. **Report**: `EvaluationReport` aggregates per-sample metrics into a summary (mean
   score per metric, p50/avg/p95 latency, total cost, total tokens) and can be printed,
   saved to JSON, or serialized for the API/UI.

`compare()` repeats step 3–5 for every `{strategy, model}` config against the **same**
pre-built index (documents are chunked+embedded once, not once per config), and returns
a `ComparisonResult` with a full metric matrix + `.best(metric)` / `.leaderboard()`.

`recommend_strategy()` calls `compare()` across every strategy (or a chosen subset), then
ranks the results by a composite score — quality (mean of retrieval + LLM-judge metrics)
minus normalized cost and latency penalties, with configurable weights — and returns the
single best strategy **for that specific dataset**, not a generic leaderboard.

## Feature list

**Strategies (18)** — naive, hybrid (dense+BM25), multi_query, hyde, rerank, rag_fusion,
compression, crag, self_rag, decomposition, step_back, agentic, flare, graph_local,
graph_global, graph_hybrid, graph_mix, multimodal.

**Providers (20+)** — OpenAI, Azure OpenAI, Azure AI Foundry, AWS Bedrock, Google Vertex,
Google AI Studio (Gemini), Anthropic, Cohere, Mistral, xAI (Grok), DeepSeek, Groq,
Together AI, Fireworks AI, DeepInfra, Anyscale, OpenRouter, Perplexity, NVIDIA NIM,
Cerebras, SambaNova, Databricks, AI21, HuggingFace (hosted + local `sentence-transformers`),
Ollama, vLLM, LM Studio, llama.cpp, any custom OpenAI-compatible endpoint, plus
embedding-only Voyage AI / Jina AI / Nomic.

**Document ingestion** — txt, md, pdf (text + tables via `pdfplumber`), docx (paragraphs +
tables), pptx (per-slide text + tables), html (text + tables via BeautifulSoup), csv, json/jsonl,
xlsx/xls (per-sheet), images (base64, for multimodal strategies), `.sql` scripts, SQLite
`.db`/`.sqlite` files (per-table), and any SQL database via `from_sql()` (SQLAlchemy —
Postgres/MySQL/SQL Server/...). `parse_dir()` walks an arbitrary directory recursively and
parses every file it recognizes, skipping unreadable ones gracefully.

**Datasets** — 2 bundled offline sets (no network) and 5 HuggingFace-backed popular
benchmarks (SQuAD, HotpotQA, Natural Questions, TriviaQA, MS MARCO).

**Metrics (10)** — context precision/recall, hit rate, MRR (retrieval); faithfulness,
answer relevance, answer correctness (LLM-judged); latency, cost, token usage
(operational). Presets: `quick` / `quality` / `full` / `production`.

**Interfaces** — Python API (`evaluate`/`compare`/`recommend_strategy`), CLI
(`ragarena run|compare|recommend|serve|models|strategies`), REST API (FastAPI,
OpenAPI docs at `/docs`), and the Next.js playground UI (Overview, Playground, Compare,
Recommend, Datasets, Model Catalog, Runs).

**Deployment** — `pip install ragarena[all]`, a single-container Docker image (static
Next.js export + FastAPI, one port), published to PyPI, Docker Hub and GHCR on every
tagged release via GitHub Actions.
