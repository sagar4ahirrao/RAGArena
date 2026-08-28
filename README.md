<div align="center">

# ⚡ RagArena

### Evaluate & benchmark every RAG strategy × LLM × embedding model — with one unified API

**18 strategies · 100+ models · 20+ providers · 10 metrics · Next.js playground · Docker/PyPI/GitHub**

[![PyPI](https://img.shields.io/pypi/v/ragarena?color=blue&logo=pypi)](https://pypi.org/project/ragarena/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen?logo=githubactions)]()

📐 [Architecture & full feature reference](ARCHITECTURE.md) · 🐳 [Docker guide](DOCKER.md) · 📝 [Changelog](CHANGELOG.md)

*One unified API across **every RAG strategy, LLM and embedding model**:*

```python
from ragarena import evaluate

evaluate(questions=[...], documents=[...],
         strategy="hybrid",                      # any of 18 strategies
         model="openai/gpt-4o-mini",             # swap with claude/gemini/llama/...
         embedding_model="voyage/voyage-3",      # swap with openai/cohere/jina/...
         metrics="quality").print_summary()
```

</div>

---

## Why RagArena?

Choosing a RAG stack is guesswork today. *"Is hybrid retrieval actually better than naive for
my data? Is GPT-4o worth 17× the price of GPT-4o-mini for answer faithfulness? Do Voyage-3
embeddings beat OpenAI's on my legal corpus?"*

**RagArena turns those guesses into a leaderboard.**

| | Ragas | DeepEval | TruLens | **RagArena** |
|---|---|---|---|---|
| Score *your existing* pipeline | ✅ | ✅ | ✅ | ✅ |
| **Run the pipelines themselves** (18 strategies) | ❌ | ❌ | ❌ | ✅ |
| **Swap LLM/embedding providers per run** (`provider/model` syntax) | partial | partial | partial | ✅ 100+ models |
| Built-in cost + latency accounting per strategy | ❌ | ❌ | partial | ✅ |
| Head-to-head leaderboard w/ shared index | ❌ | ❌ | ❌ | ✅ |
| Zero-config web dashboard | ❌ | ❌ | ❌ | ✅ |

### vs. RAG-Anything (LightRAG-based multimodal pipelines)

[RAG-Anything](https://github.com/HKUDS/RAG-Anything) is a great *single-pipeline* multimodal
document processor (MinerU parsing, VLM captioning, multimodal knowledge graph). RagArena
attacks a different — and complementary — problem: **which RAG design should you ship?**

| Capability | RAG-Anything | **RagArena** |
|---|---|---|
| Retrieval strategies you can choose from | 1 (fixed LightRAG pipeline) | **18** (naive → hybrid → HyDE → CRAG → Self-RAG → agentic → graph ×4 → multimodal) |
| Benchmarking / evaluation harness | ❌ none | ✅ metrics, cost & latency accounting, leaderboards, strategy recommendation |
| LLM / embedding provider freedom | OpenAI-shaped funcs you wire yourself | ✅ `provider/model` IDs across 25+ providers, swap per run |
| Structured data (SQL, SQLite, Excel, CSV/TSV, JSON/JSONL, XML, YAML) | via LibreOffice conversion only | ✅ native parsers — row-aware chunking, key-path flattening, schema-context chunks |
| Heavy system dependencies | MinerU + LibreOffice + model downloads | ✅ pure-Python stdlib-first; every parser degrades gracefully |
| UI playground for non-engineers | ❌ | ✅ Next.js playground + one-command Docker image |

**Use both together:** parse gnarly scanned PDFs with RAG-Anything/MinerU, feed the extracted
content lists into RagArena's `evaluate()` to pick the best strategy/model combo before you commit.


## Install

```bash
pip install ragarena                 # core
pip install "ragarena[all]"          # + all provider SDKs
export OPENAI_API_KEY=sk-...        # only the providers you use
```

## 60-second quickstart

```python
from ragarena import evaluate

docs = [
    {"text": "Retrieval-Augmented Generation (RAG) grounds LLM answers in your documents."},
    {"text": "Hybrid retrieval combines dense vector search with BM25 keyword search."},
    {"text": "Cross-encoder rerankers like bge-reranker significantly improve precision."},
]

report = evaluate(
    questions=["What is RAG?", "How does hybrid retrieval work?"],
    documents=docs,
    reference_answers=["RAG grounds LLMs in docs", "It fuses dense + BM25 search"],
    strategy="hybrid",                          # naive|hybrid|multi_query|hyde|rerank|
                                                # rag_fusion|compression|crag|self_rag|
                                                # decomposition|step_back|agentic|flare
    model="openai/gpt-4o-mini",
    embedding_model="openai/text-embedding-3-small",
    metrics="quality",                          # quick|quality|full|production
)

report.print_summary()
report.save("report.json")
```

```
╭─ RagArena · hybrid · openai/gpt-4o-mini
├─ embedding : openai/text-embedding-3-small
├─ samples   : 2   wall time 6.4s
├──────────────────────────────────────────────────────────
│     faithfulness : 0.92      answer relevance : 0.95
│ context precision : 0.83      context recall : 0.88
│       hit rate : 1.0                   mrr : 0.75
│   avg latency : 3.1s           total cost : $0.00214
╰──────────────────────────────────────────────────────────
```

## 🔌 Drop it into any app as your RAG layer

RagArena isn't only an offline evaluator — `answer()` uses it to actually answer questions,
with `strategy="auto"` picking the best strategy for YOUR documents automatically:

```python
from ragarena import answer

# strategy="auto" (default): evaluates candidate strategies against a few real sample
# questions the first time it sees this document set, caches the winner, reuses it after
result = answer(
    query="What is RAG?",
    documents=docs,
    auto_eval_questions=["What is RAG?", "How does hybrid retrieval work?"],
)
print(result)   # -> the answer string

# or pin a specific strategy directly, same as evaluate()
answer(query="What is RAG?", documents=docs, strategy="hybrid")
```

No eval questions available yet? Auto-generate them from your own documents:

```python
from ragarena import generate_testset, evaluate

questions, references = generate_testset(docs, n=20, model="openai/gpt-4o-mini")
report = evaluate(questions=questions, reference_answers=references, documents=docs)
```

## 🏆 Find the best strategy/model in one call

```python
from ragarena import compare

result = compare(
    questions=my_questions,
    documents=my_docs,
    reference_answers=ground_truth,
    configs=[
        {"strategy": "naive",       "model": "openai/gpt-4o-mini"},
        {"strategy": "hybrid",      "model": "openai/gpt-4o-mini"},
        {"strategy": "hyde",        "model": "openai/gpt-4o-mini"},
        {"strategy": "agentic",     "model": "groq/openai/gpt-oss-120b"},
        {"strategy": "hybrid",      "model": "anthropic/claude-3-haiku-20240307"},
    ],
)
result.print_leaderboard(sort_by="faithfulness")
print("WINNER:", result.best("faithfulness"))
```

The document index is embedded **once** and shared across all configs — comparisons are fast and cheap.

## 🎯 "Which strategy is best for MY data?"

Don't want to hand-pick configs? `recommend_strategy()` runs **every** strategy (or a
chosen subset) against your corpus and questions, then ranks them by a
quality/cost/latency-weighted composite score:

```python
from ragarena import recommend_strategy

rec = recommend_strategy(
    questions=my_questions,
    documents=my_docs,
    reference_answers=ground_truth,
    model="groq/openai/gpt-oss-20b",
    embedding_model="google/gemini-embedding-001",
    quality_weight=0.7, cost_weight=0.15, latency_weight=0.15,   # tune for your priorities
)
rec.print_summary()
print(rec.best, rec.reasoning)
print(rec.code_snippet())   # ready-to-paste answer() call using the winning strategy
```

Also available as `ragarena recommend --documents ... --questions ...` (CLI) and
`POST /api/recommend` (used by the **Recommend** tab in the playground UI).

## 📄 Bring your own data — any format, any database

```python
from ragarena import parse_file, parse_dir, from_sql

docs = parse_file("report.pdf")                 # pdf, docx, pptx, html
docs += parse_file("notes.docx")                 # csv/tsv, json/jsonl, xml, yaml/yml,
docs += parse_file("sheet.xlsx")                 # xlsx (all sheets), md, txt, images,
docs += parse_dir("./knowledge_base/")           # .sql dumps, sqlite/sqlite3/db files — mixed dirs OK

docs += from_sql("postgresql://user:pass@host/db",       # any SQLAlchemy-supported DB
                  "SELECT id, title, body FROM articles")
```

Structured data is parsed *retrieval-aware*: table rows are chunked in groups that repeat the
column headers ("col: value" pairs), JSON/YAML are flattened to full key-path facts
(`offices[0].city: Berlin`), and XML keeps tag-path + attribute context — so embeddings stay
self-contained and answers cite the right field. `pip install "ragarena[ingest,sql]"` for the
optional parser/DB-driver dependencies (SQLite works with zero extras).

## 🖥 Web dashboard

```bash
RagArena serve            # → http://localhost:4000
```

- **Overview** — run history & framework stats
- **New Evaluation** — pick strategy/models from dropdowns, paste corpus, score instantly
- **Compare** — build config matrices, get leaderboards + Chart.js visualizations
- **Runs** — drill into every sample: answer, chunks, metric reasoning
- **Catalog** — browse all 100+ models with pricing/context windows

## 🔀 Every popular provider, one syntax

Models are addressed as `provider/name`:

```python
from ragarena import completion

completion(model="openai/gpt-4o-mini", ...)          # OpenAI
completion(model="anthropic/claude-3-5-sonnet-20240620", ...)
completion(model="google/gemini-1.5-flash", ...)
completion(model="deepseek/deepseek-chat", ...)      # 97% cheaper than gpt-4o
completion(model="groq/openai/gpt-oss-20b", ...)   # sub-second inference
completion(model="ollama/llama3.1", ...)             # local & free
completion(model="bedrock/meta.llama3-1-405b-instruct-v1:0", ...)
```

<details>
<summary><b>📋 Full provider support matrix (click to expand)</b></summary>

**LLM Providers** — `provider/` prefix:

| Provider | Example models | Notes |
|---|---|---|
| `openai/` | gpt-4o, gpt-4o-mini, o1-preview | flagship quality |
| `anthropic/` | claude-3-5-sonnet, opus, haiku | best coding/agentic |
| `google/`, `vertex/` | gemini-1.5-pro (2M ctx), flash | huge context |
| `azure/` | any OpenAI model on Azure | enterprise compliance |
| `bedrock/` | claude/llama/titan on AWS | VPC deployments |
| `cohere/` | command-r-plus | native RAG features |
| `mistral/` | large, nemo, codestral | EU-hosted options |
| `xai/` | grok-beta | real-time knowledge |
| `deepseek/` | deepseek-chat, coder | extreme $/quality |
| `groq/` | gpt-oss-120b @300tok/s | fastest hosted |
| `together/`, `fireworks/`, `deepinfra/` | llama, qwen, mixtral | open-model hosts |
| `perplexity/` | sonar-online | search-grounded |
| `openrouter/` | 100+ gateway models | one key, all models |
| `nvidia_nim/`, `anyscale/`, `ai21/`, `databricks/` | … | … |
| `ollama/`, `vllm/`, `lmstudio/` | local llama/qwen/gemma/phi | free, private |

**Embedding Providers:**
`openai/text-embedding-3-*` · `cohere/embed-*` · `voyage/voyage-3(-large|-code|-law|-finance)` ·
`jina/jina-embeddings-v3` · `mistral/mistral-embed` · `google/text-embedding-004` ·
`bedrock/amazon.titan-embed-text-v2` · `huggingface/BAAI/bge-m3` (+MiniLM, E5, GTE) · `ollama/nomic-embed-text`

**Rerankers:** `cohere/rerank-v3.5` · `voyage/rerank-2` · `huggingface/BAAI/bge-reranker-v2-m3`

**Vector stores:** FAISS (built-in) · Chroma · Pinecone · Qdrant · Weaviate · Milvus · LanceDB · pgvector · Elasticsearch · Redis · OpenSearch · MongoDB

</details>

Browse everything from the CLI:

```bash
RagArena models list                       # all 100+
RagArena models list --modality embedding  # embeddings only
RagArena models providers                  # provider summary
RagArena strategies                        # the 18 strategies
```

### Already using LangChain, or your own model wrapper? Bring it as-is

`model=` and `embedding_model=` don't have to be a `provider/name` string — pass any
LangChain chat model / `Embeddings` object, or a plain Python callable, and RagArena
will use it directly:

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragarena import evaluate

evaluate(
    questions=[...], documents=[...],
    model=ChatOpenAI(model="gpt-4o-mini"),        # any LangChain BaseChatModel/Runnable
    embedding_model=OpenAIEmbeddings(),            # any LangChain Embeddings object
    judge_model="openai/gpt-4o-mini",              # mix and match freely
)

# or just a plain callable — no LangChain required
evaluate(model=lambda messages: my_llm_call(messages), ...)
```

## 🧪 The 18 built-in strategies

| Strategy | What it does | Best for |
|---|---|---|
| `naive` | dense top-k → generate | baseline / simple corpora |
| `hybrid` | dense + BM25 weighted fusion (α-tunable) | keyword-heavy domains |
| `multi_query` | LLM rewrites N queries, merges results | vague questions |
| `rag_fusion` | multi-query + Reciprocal Rank Fusion | robustness over multi_query |
| `hyde` | retrieve using an imagined answer's embedding | vocabulary mismatch |
| `rerank` | wide recall → cross-encoder refine | precision-critical |
| `compression` | LLM strips irrelevant spans pre-generation | long noisy chunks |
| `crag` | grades retrieval; rewrites query if weak | production guardrails |
| `self_rag` | model decides *if* retrieval is needed | mixed easy/hard traffic |
| `decomposition` | splits complex Q → sub-Qs → synthesis | multi-hop questions |
| `step_back` | abstract principle question first | conceptual/domain Qs |
| `agentic` | iterative search→reflect→search loop | hard research tasks |
| `flare` | flags uncertain draft claims → re-retrieves | hallucination-prone domains |
| `graph_local` | entity-precise retrieval over a knowledge graph | "who/what" factual lookups |
| `graph_global` | macro-theme retrieval across entity communities | "how/why" analytical Qs |
| `graph_hybrid` | combines local entities + global themes | general-purpose graph RAG |
| `graph_mix` | local + global fused in one synthesis pass | best-of-both retrieval |
| `multimodal` | retrieves typed chunks (text/table/image/equation) | mixed-content documents |

## 🕸️ Graph RAG (dual-level retrieval)

`graph_*` strategies layer a lightweight knowledge graph over your index — entities
are extracted per chunk, chunks that share entities form *communities*, and queries
are answered at two levels:

- **local** (`graph_local`) — match the query's entities to graph nodes and pull the
  connected chunks. Best for precise "who/what" factual lookups.
- **global** (`graph_global`) — summarise each community, rank communities by relevance
  to the query, then synthesise a cross-document answer. Best for "how/why" analysis.
- **hybrid** / **mix** (`graph_hybrid`, `graph_mix`) — combine both levels.

The graph is built lazily and cached on a shared index, so `compare()` only builds it
once. Entity extraction falls back to a deterministic keyword extractor if the LLM is
unavailable.

```python
from ragarena import compare

result = compare(
    questions=["Who builds Pinecone?", "How do retrieval systems relate?"],
    documents=my_docs,
    configs=[
        {"strategy": "graph_local",  "model": "openai/gpt-4o-mini"},
        {"strategy": "graph_global", "model": "openai/gpt-4o-mini"},
        {"strategy": "graph_hybrid", "model": "anthropic/claude-3-haiku-20240307"},
    ],
)
```

Build a graph index directly for inspection:

```python
from ragarena import VectorIndex, GraphIndex

vi = VectorIndex(embedding_model="openai/text-embedding-3-small")
vi.add_documents(my_docs)
g = GraphIndex(vi).build("openai/gpt-4o-mini")   # cache on the index
local  = g.local_search("What is Pinecone?", k=5, llm_model="openai/gpt-4o-mini")
chunks, theme = g.global_search("How do vector DBs compare?", k=5, llm_model="openai/gpt-4o-mini")
```

## 🖼️ Multimodal RAG

Tables, images and equations are kept **intact** (not sentence-split) and tagged with a
`doc_type` so retrieval and generation can treat them differently:

```python
from ragarena import MultimodalDocument, evaluate

docs = [
    MultimodalDocument(content="| model | params |", doc_type="table"),
    MultimodalDocument(content="E = mc^2",           doc_type="equation"),
    {"text": "RAG grounds LLMs in retrieved context.", "metadata": {"doc_type": "text"}},
]
evaluate(questions=["..."], documents=docs, strategy="multimodal",
         model="openai/gpt-4o-mini")
```

## 📐 Metrics

Presets: `quick` · `quality` · `full` · `production` — or cherry-pick:

```python
metrics=["context_precision", "context_recall", "hit_rate", "mrr",   # retrieval
         "faithfulness", "answer_relevance", "answer_correctness",   # generation (LLM-judge)
         "latency_s", "cost_usd", "total_tokens"]                    # operational
```

Any model can be the judge: `judge_model="anthropic/claude-3-haiku-20240307"`.

**Retrieval metrics use real embedding similarity, not keyword matching.** When an
`embedding_model` is set, `context_precision`/`context_recall`/`hit_rate`/`mrr` score
chunk relevance via cosine similarity — so a paraphrased-but-relevant chunk scores
correctly even with zero shared keywords. (Falls back to lexical overlap if no embedding
model is configured.) Tune the semantic threshold with `RAGARENA_RELEVANCE_THRESHOLD`
(default `0.55`).

**Reduce LLM-judge variance with multi-sampling.** A single judge call can be noisy —
pass `judge_samples=3` to average 3 independent judge calls (with temperature jitter)
instead of trusting one sample:

```python
evaluate(questions=[...], documents=docs, judge_samples=3)
# report includes score_stdev and all_scores per metric
```

## 🗄️ Benchmark the vector DB itself

The vector store is swappable, so you can hold the corpus, embeddings and strategy
fixed and measure what the DB choice alone costs you:

```python
from ragarena import VectorIndex, evaluate, list_backends

print(list_backends())   # which backends are installed & importable here

for backend in ["numpy", "faiss", "qdrant", "lancedb", "chroma"]:
    index = VectorIndex(embedding_model="openai/text-embedding-3-small", backend=backend)
    index.add_documents(docs)
    report = evaluate(questions=my_questions, index=index, strategy="hybrid")
    print(backend, report.summary())
```

**Nine backends, all verified against live servers.** Runnable with no server:
`numpy` (default, exact cosine), `faiss`, `chroma`, `qdrant` (in-memory), `lancedb`.
Server-backed: `qdrant`, `chroma`, `weaviate`, `elasticsearch`, `redis`, `pgvector` —
pass connection details as kwargs:

```python
VectorIndex(embedding_model=..., backend="pgvector", dsn="postgresql://user:pw@host/db")
VectorIndex(embedding_model=..., backend="weaviate", url="http://localhost:8080")
VectorIndex(embedding_model=..., backend="elasticsearch", url="http://localhost:9200")
VectorIndex(embedding_model=..., backend="redis", url="redis://localhost:6379")
VectorIndex(embedding_model=..., backend="qdrant", url="http://localhost:6333")
VectorIndex(embedding_model=..., backend="chroma", host="localhost", port=8000)
```

All nine return **identical similarity scores** on the same corpus, so a comparison
measures the store, not an accidental difference in how it was configured.

> **Benchmark Chroma before LanceDB.** They load conflicting native Arrow/Rust libraries:
> once LanceDB has written a table, creating a Chroma client in the same process aborts
> the interpreter outright (the process dies — it isn't a catchable exception). RagArena
> warns when it detects this order; the fix is to run Chroma first, or give each backend
> its own process.

> **A note on Elasticsearch defaults.** Since 8.12 an indexed `dense_vector` defaults to
> `int8_hnsw`, which quantizes vectors to 8 bits and shifts similarities by ~0.4% —
> enough to reorder documents that are genuinely close, silently scoring a *different*
> retrieval than the one you meant to measure. RagArena therefore defaults to exact
> search (`index_type="flat"`); pass `index_type="hnsw"` or `"int8_hnsw"` explicitly when
> you actually want to benchmark ANN behaviour at scale.

`register_backend()` adds your own. `RagArena backends` lists them from the CLI.

## ✅ Regression-check runs, gate CI on quality

```python
from ragarena import evaluate, diff_runs, EvaluationReport
from ragarena.testing import assert_metric, assert_no_regression

report = evaluate(questions=Q, documents=DOCS, reference_answers=REFS, strategy="hybrid")
assert_metric(report, "faithfulness", gte=0.8)      # fails like a normal assert if it drops below

report.save("latest.json")
old = EvaluationReport.load("baseline.json")
assert_no_regression(diff_runs(old, report))        # fails if any metric got worse
```

`RagArena diff --a baseline.json --b latest.json` does the same check from the CLI.

## 📦 Use inside your existing pipeline

Already have a RAG system? Score it directly:

```python
from ragarena import VectorIndex
from ragarena.engine import EvalSample, MetricContext
from ragarena.metrics import resolve_metrics

# ...run YOUR pipeline to get answer + chunks...
sample = EvalSample(question=q, reference_answer=gt,
                    generated_answer=your_answer,
                    retrieved_chunks=[{"text": c} for c in your_chunks],
                    context="...", usage={...}, latency_s=..., intermediate={})
for m in resolve_metrics("full"):
    print(m.name, m.compute(sample, MetricContext(judge_model="openai/gpt-4o-mini")).score)
```

## HTTP API

```bash
curl -X POST localhost:4000/api/evaluate -H 'Content-Type: application/json' -d '{
  "questions": ["What is RAG?"],
  "documents": [{"text": "RAG grounds LLM answers in documents."}],
  "strategy": "hybrid",
  "model": "openai/gpt-4o-mini",
  "embedding_model": "voyage/voyage-3"
}'
```

Also available: `POST /api/compare` · `GET /api/catalog` · `GET /api/runs/{id}` · `GET /health`

## Roadmap

- [ ] Async batch runner + checkpoint/resume
- [ ] Statistical significance tests (paired bootstrap) between configs
- [ ] Chroma/Pinecone/Qdrant backends for `VectorIndex`
- [ ] Graph RAG incremental updates + community-aware re-indexing
- [ ] Cloud/blob storage ingestion (S3, Azure Blob, GCS) alongside local `parse_dir()`
- [ ] Prompt-optimization loop (DSPy-style)
- [ ] Team features: API keys, budgets, RBAC
- [ ] CI mode: `ragarena ci --threshold faithfulness>=0.8` (fails PRs on regressions)

## Contributing

PRs welcome! `pip install -e ".[dev]" && pytest`. Please read `CONTRIBUTING.md`.

## License

MIT © RagArena contributors

<div align="center">
<sub>Built for developers tired of guessing. Star ⭐ if it saved you a benchmark week.</sub>
</div>
