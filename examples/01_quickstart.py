"""
End-to-end example: find the best strategy for YOUR documents.

Prereqs:
    export OPENAI_API_KEY=sk-...        (or any provider key you want to test)

Run:
    python examples/01_quickstart.py
"""
from RagArena import evaluate, compare

# ── 1. Your knowledge base ────────────────────────────────────────────────────
DOCS = [
    {"text": "Retrieval-Augmented Generation (RAG) grounds LLM answers in your own documents "
             "instead of relying on parametric memory.", "metadata": {"topic": "basics"}},
    {"text": "Naive RAG embeds a query, retrieves top-k chunks by cosine similarity, and "
             "stuffs them into the prompt.", "metadata": {"topic": "strategies"}},
    {"text": "Hybrid retrieval combines dense vector search with sparse BM25 keyword scoring; "
             "alpha controls the weighting between the two signals.", "metadata": {"topic": "strategies"}},
    {"text": "Cross-encoder rerankers such as bge-reranker score each query-chunk pair jointly, "
             "giving much better precision than bi-encoder similarity alone.",
     "metadata": {"topic": "reranking"}},
]

QUESTIONS = [
    "What is RAG?",
    "How does hybrid retrieval work?",
    "Why use a cross-encoder reranker?",
]
REFERENCES = [
    "RAG grounds LLM answers in your documents",
    "It fuses dense vector search with BM25 keyword scores",
    "They score query and chunk jointly for better precision",
]

# ── 2. Single evaluation ──────────────────────────────────────────────────────
report = evaluate(
    questions=QUESTIONS,
    documents=DOCS,
    reference_answers=REFERENCES,
    strategy="hybrid",
    model="openai/gpt-4o-mini",
    embedding_model="openai/text-embedding-3-small",
    metrics="quality",
)
report.print_summary()
report.save("quickstart_report.json")

# ── 3. Head-to-head: which strategy + model wins? ────────────────────────────
comparison = compare(
    questions=QUESTIONS,
    documents=DOCS,
    reference_answers=REFERENCES,
    configs=[
        {"strategy": "naive",   "model": "openai/gpt-4o-mini"},
        {"strategy": "hybrid",  "model": "openai/gpt-4o-mini"},
        {"strategy": "rerank",  "model": "openai/gpt-4o-mini"},
        # swap generators freely — same corpus, shared index:
        # {"strategy": "naive", "model": "deepseek/deepseek-chat"},
        # {"strategy": "naive", "model": "anthropic/claude-3-haiku-20240307"},
    ],
)
comparison.print_leaderboard(sort_by="faithfulness")
print("\nWINNER by faithfulness:", comparison.best("faithfulness"))
