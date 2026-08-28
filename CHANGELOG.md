# Changelog

All notable changes to RagArena are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); versioning: [SemVer](https://semver.org/).

## [0.6.0] — 2026-08-28

Every input format and every vector store verified against real servers running
in Docker, not mocks.

### Added
- **Four server-backed vector stores**: `weaviate`, `elasticsearch`, `redis`
  (RediSearch), and `pgvector` join the existing `numpy`/`faiss`/`chroma`/
  `qdrant`/`lancedb` — nine backends total, each taking its connection
  details as kwargs, e.g.
  `VectorIndex(backend="pgvector", dsn="postgresql://...")`.
- `chroma` and `qdrant` now also accept server connection details
  (`host`/`port`, `url`) instead of only running in-process.
- `list_backends()` reports `needs_server` and checks server-backed stores by
  importing their client rather than dialing a default address, which
  previously produced false negatives.

### Verified
- All nine backends return **identical similarity scores** on the same corpus
  and embeddings, against live Qdrant, Chroma, Weaviate, Elasticsearch, Redis
  Stack and pgvector containers — so a store-vs-store comparison measures the
  store, not a configuration accident.
- All 18 RAG strategies run end-to-end over a corpus pulled from a live
  PostgreSQL server and indexed into live pgvector.
- `from_sql()` against live **PostgreSQL, MySQL and pgvector-Postgres**
  servers plus SQLite files.
- `parse_file()` across **all 23 supported extensions** — including PDF,
  which had never been exercised: txt, md, markdown, html, htm, xml, yaml,
  yml, json, jsonl, csv, tsv, pdf, docx, pptx, xlsx, sql, db, sqlite,
  sqlite3, png, jpg, jpeg.

### Fixed
- **Elasticsearch returned incorrectly ranked results by default.** Since 8.12
  an indexed `dense_vector` defaults to `int8_hnsw`, scalar-quantizing to 8
  bits and perturbing similarities by ~0.4%. On a corpus with two documents
  0.19% apart this ranked them backwards, so an evaluation would score a
  different retrieval than the one under test. The backend now defaults to
  exact search (`index_type="flat"`, matching exact cosine to 6 decimal
  places); pass `index_type="hnsw"`/`"int8_hnsw"` deliberately for ANN
  benchmarking.
- **Redis backend failed to import on redis-py >= 5.1**, which renamed
  `redis.commands.search.indexDefinition` to `index_definition`. Both spellings
  are now handled.
- **`VectorIndex.add_documents()` silently indexed nothing** when given
  documents with no extractable text — most commonly a folder of images,
  which `parse_file()` returns with empty `text` and bytes in `images`. It
  returned 0 with no signal, leaving an empty index and an evaluation that
  retrieved nothing for no visible reason. It now warns, and points at
  `to_multimodal()` + `strategy="multimodal"` when the documents look like
  images.

## [0.5.0] — 2026-08-28

Vector stores become swappable (and therefore benchmarkable), plus five real
bugs found by a full live verification pass over every feature.

### Added
- **Pluggable vector-store backends**: `VectorIndex(backend="numpy"|"faiss"|
  "chroma"|"qdrant"|"lancedb", **backend_kwargs)`. Backends receive
  already-embedded vectors, so swapping one changes only storage and ANN
  search — identical corpus, embeddings and strategies on top — which is
  what makes a fair "which vector DB is best for my data" comparison
  possible. All in-process: qdrant uses `:memory:`, lancedb a temp dir, so
  no server is needed for a benchmark run. Verified live: numpy/faiss/
  qdrant/lancedb return identical scores to 4 decimal places on the same
  corpus and all pass end-to-end through `evaluate()`.
- `register_backend()` as the extension point for server-backed stores
  (Weaviate, Milvus, pgvector, Elasticsearch, Redis, Pinecone);
  `list_backends()` reports availability instead of crashing on a missing
  optional dependency; `ragarena backends` surfaces it from the CLI.
- New `vectordb` extra: `pip install ragarena[vectordb]` (included in `all`).

### Fixed
- **`rag_fusion` was completely broken** — it sorted retrieved chunks by the
  `Chunk` objects themselves rather than by their Reciprocal Rank Fusion
  scores, so every run failed with `'<' not supported between instances of
  'Chunk' and 'Chunk'` and the computed RRF scores were never used. A second
  latent crash (`.items()` on a list) in the same method is fixed too.
  All 18 strategies now pass a live end-to-end run; previously 17 did.
- **`RunDiff.regressions()` had cost backwards** — the lower-is-better set
  listed `cost_usd`, but the aggregate key is `total_cost_usd`, and the
  prefix check never matched it. A cost *reduction* was reported as a
  regression while a real cost *blow-up* was silently missed, which
  undermined `assert_no_regression()` in CI. Now matched exactly against the
  keys `summary()` actually emits.
- **`load_dataset(name, use_bundled=True)` silently returned a different
  corpus** — any non-bundled name (squad, hotpotqa, natural_questions,
  triviaqa, ms_marco) got rag_faq's content back with no indication, making
  any benchmark labelled with those names wrong. It now emits a `UserWarning`
  naming the substitution. The bundled-dataset list is derived from the
  registry so it can't drift.
- **`ragarena compare --inline` could never work** — it crashed with
  `'NoneType' object is not iterable` when `--configs` was omitted, and the
  emptiness check ran before `--inline` was collected. `--inline` now works
  on its own and accepts a JSON array as well as a single object.
- **Removed the dead `google/text-embedding-004` catalog entry** — Google
  retired the model and the live API returns 404 for it, so listing it only
  offered a choice that could never work (including in the playground's
  model dropdown). Use `google/gemini-embedding-001`.

## [0.4.0] — 2026-08-27

Adds a production "use it, not just evaluate it" surface, plus test-set
generation and CI-grade regression tooling — informed by a survey of the
open-source RAG-eval landscape (Ragas, DeepEval, TruLens, RAGChecker, ARES,
Arize Phoenix, Giskard, LangSmith).

### Added
- **`answer()` — plug-and-play RAG answering**: use RagArena as the RAG
  layer inside any app, not only as an offline evaluator —
  `answer(query="...", documents=[...])` returns an answer directly.
  `strategy="auto"` (the default) evaluates candidate strategies against a
  small set of representative `auto_eval_questions` the first time it sees
  a document set (via `recommend_strategy()` under the hood), caches the
  winner, and reuses it on every later call — falls back to `"hybrid"` when
  no eval questions are given, rather than pretending to pick empirically.
  Also available via `ragarena ask --query ... --documents ...`.
- **`RecommendationResult.code_snippet()`**: `recommend_strategy()` results
  now include a ready-to-paste Python snippet using `answer()` with the
  winning strategy — closes the gap between "here's the best strategy" and
  actually shipping with it.
- **`generate_testset()` / `generate_testset_detailed()`**: synthetic
  question+reference-answer generation from your own documents (mixing
  simple/reasoning/multi-passage question types), so comparing strategies
  at scale doesn't require hand-writing a question set first. Also
  available via `ragarena testgen --documents ...`.
- **`diff_runs()` / `EvaluationReport.load()`**: compare two saved
  evaluation runs (e.g. before/after a strategy, model, or prompt change)
  and flag per-metric regressions, correctly accounting for lower-is-better
  metrics like latency/cost. Also available via
  `ragarena diff --a run1.json --b run2.json`.
- **`ragarena.testing`**: `assert_metric()` / `assert_no_regression()` —
  pytest-style assertions over an `EvaluationReport`/`RunDiff`, for gating
  CI on RAG quality the same way you'd gate on a unit test.
- **Confidence intervals on metric scores**: `EvaluationReport.confidence_intervals()`
  reports a 95% CI per quality metric across the run's samples (normal
  approximation), surfaced in `print_summary()`/`to_dict()` — makes it
  possible to tell whether a `compare()`/`recommend_strategy()` ranking
  reflects a real difference or sample noise.

### Fixed
- `EvaluationReport.save()` crashed with `TypeError: Object of type function
  is not JSON serializable` when `model`/`embedding_model`/`judge_model` was
  a bring-your-own-model object (LangChain model or plain callable) rather
  than a string — the report now stores a stable label (`custom/<name>`)
  for non-string models instead of the live object.

## [0.3.0] — 2026-08-27

Reliability and extensibility release, in direct response to feedback that
retrieval metrics were purely lexical and the framework only accepted
`provider/model` strings.

### Added
- **Embedding-based retrieval metrics**: `context_precision`, `context_recall`,
  `hit_rate`, and `mrr` now score chunk relevance using real embedding cosine
  similarity (via the run's own `embedding_model`) instead of keyword
  overlap, when an embedding model is available. Falls back to the previous
  lexical heuristic otherwise (e.g. no `embedding_model` configured). The
  semantic-relevance threshold is tunable via `RAGARENA_RELEVANCE_THRESHOLD`
  (default `0.55`). Each metric result now reports which `method` was used.
  Verified live: a pure paraphrase that keyword-overlap scored `0.0` is
  correctly caught at `0.775` similarity; an unrelated distractor correctly
  scores below threshold at `0.477`.
- **Multi-sample LLM-judge voting**: pass `judge_samples=N` to `evaluate()`,
  `compare()` (per-config), or `ragarena run --judge-samples N` to average N
  independent judge calls (with temperature jitter) instead of trusting a
  single sample. Reports `score_stdev` and `all_scores` alongside the mean.
  Verified live with `judge_samples=3`.
- **Bring-your-own-model (BYOM)**: `completion()` and `embedding()` now
  accept any LangChain chat model / `Embeddings` object, or a plain
  `callable`, in addition to the existing `"provider/name"` strings — so
  developers can plug in models from LiteLLM, LangChain, or their own stack
  interchangeably. Verified live with both a LangChain-style `.invoke()`
  object and a plain callable, through the full `evaluate()` pipeline.

## [0.2.5] — 2026-08-27

Verified with the Playwright MCP browser against the live app (not just
curl/API checks) — found and fixed a real reliability gap and remaining
light-theme contrast issues.

### Fixed
- **No rate-limit retry**: a burst of LLM calls (e.g. an 8-question run with
  the `production` metric preset, which adds 2 judge calls per question)
  against a free-tier key could exceed the provider's tokens-per-minute
  limit — half the questions in a live test failed outright with
  `RateLimitError`. `completion()` now retries on a rate-limit error (up to
  3 times), honoring the provider's own suggested wait time when the error
  message includes one (e.g. Groq's "try again in 4.47s"), falling back to
  an increasing backoff otherwise. Re-ran the exact scenario that failed
  live: 0 errors across all 8 questions afterward.
- Missing favicon caused a `/favicon.ico` 404 console error on every single
  page load — added `web/app/icon.svg`.
- A handful of remaining hardcoded `text-slate-200/300` headings (light
  grays meant for dark backgrounds) were still barely legible in light
  theme; swapped to the theme-aware `text-fg`/`text-fg-muted` tokens.

Verified via the Playwright MCP browser: navigated all 7 pages (zero
console errors), then drove a real interactive evaluation end-to-end
through the actual UI (select dataset → pick live Groq/Gemini models → run
→ confirm results render) rather than only checking the API layer.

## [0.2.4] — 2026-08-26

`completion()` now routes through LiteLLM instead of hand-rolled per-provider
SDK integrations.

### Changed
- Removed the bespoke `_anthropic_completion`/`_cohere_completion`/
  `_bedrock_completion`/`_azure_completion` implementations (~180 lines) in
  favor of `litellm.completion()`, which already normalizes auth and request
  shape across 100+ providers. ragarena still does its own upfront API-key
  resolution (for the `MissingAPIKeyError` UX) and cost estimation (from its
  own catalog pricing, not LiteLLM's), so behavior for callers is unchanged.
  Azure AI Foundry keeps its direct REST implementation (bespoke Bearer-token
  surface, no fixed deployment) rather than going through LiteLLM.
- Added automatic retry-with-param-adjustment for newer "reasoning" model
  deployments (o1/o3/gpt-5-style) that reject `max_tokens` (need
  `max_completion_tokens`) or non-default `temperature` — verified live
  against an Azure `gpt-5.5` deployment.

Verified end-to-end against Groq, Google Gemini, Azure OpenAI, and
OpenRouter (the 4 providers with live credentials available), plus all 18
strategies re-run against Groq + Gemini with zero regressions.

## [0.2.3] — 2026-08-26

Real browser testing (Playwright) uncovered and fixed critical playground bugs;
light/dark theme; usability and metric-accuracy fixes.

### Fixed
- **Critical**: `GET /api/runs/{id}` never returned `status: "done"` for a
  completed run, so the Playground/Compare/Recommend UI's poll loop
  (`s.status === "done"`) was never satisfied — every browser-driven
  evaluation appeared to hang forever ("...judging… (undefined)") until it
  eventually timed out. Found only via a real headless-browser test; curl/API
  testing alone couldn't surface it since the JSON was otherwise valid.
- FastAPI's UI catch-all route only matched single-segment paths and only
  ever served `index.html`, so Next's client-router prefetch requests
  (`/playground/index.txt`, etc.) 404'd on every single page load. Rewrote to
  a proper `{path:path}` catch-all serving any static-export file.
- Missing API key: previously fell through to a dummy key and surfaced a
  confusing raw 401 from the provider. New `MissingAPIKeyError` names the
  exact environment variable(s) to set, everywhere (Python API, CLI, REST API).
- `evaluate()` now fails fast on `MissingAPIKeyError`/`UnknownModelError`
  instead of silently repeating the same failure for every question in the
  batch; `compare()` now records a bad config's failure in a new
  `ComparisonResult.errors` dict and keeps the results already computed for
  the other configs, instead of crashing the whole comparison.
- `ContextRecall` rescaled raw keyword overlap by an undocumented `/0.5`,
  inflating a 50% overlap into a reported "1.0 perfect recall" — now reports
  the raw overlap fraction.
- Next.js bumped 14.2.5 → 14.2.35 and postcss pinned via `overrides`,
  clearing the critical/high npm audit findings.

### Added
- **"Get code" panel** on the Playground page — generates a ready-to-run
  Python, cURL, or JavaScript snippet reproducing the current strategy/model/
  corpus configuration.
- **Light / dark theme toggle** (system → light → dark), CSS-variable-driven
  so it applies across every page without per-component dark: variants.
- `scripts_browser_test.py` / `scripts_browser_e2e.py` — real Playwright
  browser tests (page loads + console-error checks + a full interactive
  playground run), not just API-level curl checks.

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

[0.2.5]: https://github.com/sagar4ahirrao/ragarena/releases/tag/v0.2.5
[0.2.4]: https://github.com/sagar4ahirrao/ragarena/releases/tag/v0.2.4
[0.2.3]: https://github.com/sagar4ahirrao/ragarena/releases/tag/v0.2.3
[0.2.2]: https://github.com/sagar4ahirrao/ragarena/releases/tag/v0.2.2
[0.2.0]: https://github.com/sagar4ahirrao/ragarena/releases/tag/v0.2.0
[0.1.0]: https://github.com/sagar4ahirrao/ragarena/releases/tag/v0.1.0
