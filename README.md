<div align="center">

# ⚡ RAGEval

### Evaluate & benchmark every RAG strategy × LLM × embedding model — with one unified API

**13 strategies · 100+ models · 25+ providers · 10 metrics · built-in web dashboard**

[![PyPI](https://img.shields.io/pypi/v/rageval?color=blue&logo=pypi)](https://pypi.org/project/rageval/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen?logo=githubactions)]()

*Like [LiteLLM](https://github.com/BerriAI/litellm) unified LLM calls, RAGEval unifies **RAG evaluation**:*

```python
from rageval import evaluate

evaluate(questions=[...], documents=[...],
         strategy="hybrid",                      # any of 13 strategies
         model="openai/gpt-4o-mini",             # swap with claude/gemini/llama/...
         embedding_model="voyage/voyage-3",      # swap with openai/cohere/jina/...
         metrics="quality").print_summary()
```

</div>

---

## Why RAGEval?

Choosing a RAG stack is guesswork today. *"Is hybrid retrieval actually better than naive for
my data? Is GPT-4o worth 17× the price of GPT-4o-mini for answer faithfulness? Do Voyage-3
embeddings beat OpenAI's on my legal corpus?"*

**RAGEval turns those guesses into a leaderboard.**

| | Ragas | DeepEval | TruLens | **RAGEval** |
|---|---|---|---|---|
| Score *your existing* pipeline | ✅ | ✅ | ✅ | ✅ |
| **Run the pipelines themselves** (13 strategies) | ❌ | ❌ | ❌ | ✅ |
| **Swap LLM/embedding providers per run** (`provider/model` syntax) | partial | partial | partial | ✅ 100+ models |
| Built-in cost + latency accounting per strategy | ❌ | ❌ | partial | ✅ |
| Head-to-head leaderboard w/ shared index | ❌ | ❌ | ❌ | ✅ |
| Zero-config web dashboard | ❌ | ❌ | ❌ | ✅ |

## Install

```bash
pip install rageval                 # core
pip install "rageval[all]"          # + all provider SDKs
export OPENAI_API_KEY=sk-...        # only the providers you use
```

## 60-second quickstart

```python
from rageval import evaluate

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
╭─ RAGEval · hybrid · openai/gpt-4o-mini
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
from rageval import compare

result = compare(
    questions=my_questions,
    documents=my_docs,
    reference_answers=ground_truth,
    configs=[
        {"strategy": "naive",       "model": "openai/gpt-4o-mini"},
        {"strategy": "hybrid",      "model": "openai/gpt-4o-mini"},
        {"strategy": "hyde",        "model": "openai/gpt-4o-mini"},
        {"strategy": "agentic",     "model": "groq/llama-3.1-70b-versatile"},
        {"strategy": "hybrid",      "model": "anthropic/claude-3-haiku-20240307"},
    ],
)
result.print_leaderboard(sort_by="faithfulness")
print("WINNER:", result.best("faithfulness"))
```

The document index is embedded **once** and shared across all configs — comparisons are fast and cheap.

## 🖥 Web dashboard

```bash
rageval serve            # → http://localhost:4000
```

- **Overview** — run history & framework stats
- **New Evaluation** — pick strategy/models from dropdowns, paste corpus, score instantly
- **Compare** — build config matrices, get leaderboards + Chart.js visualizations
- **Runs** — drill into every sample: answer, chunks, metric reasoning
- **Catalog** — browse all 100+ models with pricing/context windows

## 🔀 Every popular provider, one syntax

Models are addressed as `provider/name`:

```python
from rageval import completion

completion(model="openai/gpt-4o-mini", ...)          # OpenAI
completion(model="anthropic/claude-3-5-sonnet-20240620", ...)
completion(model="google/gemini-1.5-flash", ...)
completion(model="deepseek/deepseek-chat", ...)      # 97% cheaper than gpt-4o
completion(model="groq/llama-3.1-8b-instant", ...)   # sub-second inference
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
| `groq/` | llama-3.1-70b @300tok/s | fastest hosted |
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
rageval models list                       # all 100+
rageval models list --modality embedding  # embeddings only
rageval models providers                  # provider summary
rageval strategies                        # the 13 strategies
```

## 🧪 The 13 built-in strategies

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
from rageval import VectorIndex
from rageval.engine import EvalSample, MetricContext
from rageval.metrics import resolve_metrics

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
- [ ] Prompt-optimization loop (DSPy-style)
- [ ] Team features: API keys, budgets, RBAC
- [ ] CI mode: `rageval ci --threshold faithfulness>=0.8` (fails PRs on regressions)

## Contributing

PRs welcome! `pip install -e ".[dev]" && pytest`. Please read `CONTRIBUTING.md`.

## License

MIT © RAGEval contributors

<div align="center">
<sub>Built for developers tired of guessing. Star ⭐ if it saved you a benchmark week.</sub>
</div>
