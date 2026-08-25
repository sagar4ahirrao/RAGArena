"""
Graph + multimodal RAG with RagArena.

Demonstrates:
  * graph_* strategies (dual-level entity/global retrieval) via compare()
  * building a GraphIndex directly for inspection
  * multimodal ingestion with MultimodalDocument (tables/images/equations kept intact)

Prereqs:
    export OPENAI_API_KEY=sk-...        (or any provider key you want to test)

Run:
    python examples/02_graph_multimodal.py
"""
from ragarena import (
    evaluate, compare,
    VectorIndex, GraphIndex, MultimodalDocument,
)

# ── 1. A mixed corpus: text + table + equation ───────────────────────────────────
DOCS = [
    {"text": "Pinecone is a managed vector database optimised for low-latency similarity "
             "search at scale.", "metadata": {"topic": "vector-db"}},
    {"text": "GraphRAG extracts entities and relations to build a knowledge graph, enabling "
             "questions that span many documents.", "metadata": {"topic": "graph"}},
    {"text": "A vector database indexes embeddings so nearest-neighbour search retrieves the "
             "most relevant chunks for a query.", "metadata": {"topic": "vector-db"}},
    MultimodalDocument(
        content="| strategy | latency_ms | recall@10 |\n|---|---|---|\n| naive | 40 | 0.71 |"
                "\n| hybrid | 55 | 0.88 |",
        doc_type="table"),
    MultimodalDocument(content="recall@k = |relevant ∩ retrieved_k| / |relevant|", doc_type="equation"),
]

QUESTIONS = [
    "What is Pinecone?",
    "How do vector databases relate to knowledge graphs?",
    "What does the benchmark table show about hybrid retrieval?",
]

# ── 2. Compare graph strategies on the same shared index ─────────────────────────
comparison = compare(
    questions=QUESTIONS,
    documents=DOCS,
    metrics="quality",
    configs=[
        {"strategy": "graph_local",  "model": "openai/gpt-4o-mini"},
        {"strategy": "graph_global", "model": "openai/gpt-4o-mini"},
        {"strategy": "graph_hybrid", "model": "openai/gpt-4o-mini"},
        {"strategy": "multimodal",   "model": "openai/gpt-4o-mini"},
    ],
)
comparison.print_leaderboard(sort_by="faithfulness")

# ── 3. Inspect the knowledge graph directly ─────────────────────────────────────
vi = VectorIndex(embedding_model="openai/text-embedding-3-small")
vi.add_documents(DOCS)
graph = GraphIndex(vi).build("openai/gpt-4o-mini")
print(f"\nGraph: {len(graph.entities)} chunks, {len(graph.communities)} communities")
local = graph.local_search("What is Pinecone?", k=3, llm_model="openai/gpt-4o-mini")
print("Local hits:", [c.text[:50] for c in local])

# ── 4. Multimodal question over the typed table/equation ────────────────────────
report = evaluate(
    questions=["What does the benchmark table show about hybrid retrieval?"],
    documents=DOCS,
    strategy="multimodal",
    model="openai/gpt-4o-mini",
    embedding_model="openai/text-embedding-3-small",
    metrics="quick",
)
print("\nMultimodal answer:", report.samples[0].answer)
