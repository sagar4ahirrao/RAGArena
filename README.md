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
```

Also available as `ragarena recommend --documents ... --questions ...` (CLI) and
`POST /api/recommend` (used by the **Recommend** tab in the playground UI).

## 📄 Bring your own data — any format, any database

```python
from ragarena import parse_file, parse_dir, from_sql

docs = parse_file("report.pdf")                 # pdf, docx, pptx, html, csv, json,
docs += parse_file("notes.docx")                 # xlsx, md, txt, images, .sql, .sqlite
docs += parse_dir("./knowledge_base/", recursive=True)   # walk a whole directory, mixed formats

docs += from_sql("postgresql://user:pass@host/db",       # any SQLAlchemy-supported DB
                  "SELECT id, title, body FROM articles")
```

`pip install "ragarena[ingest,sql]"` for the optional parser/DB-driver dependencies.

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
