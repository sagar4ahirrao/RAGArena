"""Popular RAG/QA datasets — bundled samples + optional HuggingFace loaders.

Every loader returns::

    {
        "documents": [{"text": str, "metadata": {...}}, ...],
        "questions": [str, ...],
        "reference_answers": [str | None, ...],
    }

which plugs straight into :func:`ragarena.engine.evaluate` / :func:`compare`.

Bundled datasets need no network access and no extra dependency — they exist
so the framework can be smoke-tested and demoed offline. The HuggingFace-backed
loaders (``squad``, ``hotpotqa``, ``natural_questions``, ``triviaqa``, ``ms_marco``)
pull real benchmark data on first use via the optional ``datasets`` package
(``pip install ragarena[datasets]``) and cache a capped sample in memory.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple


def _bundled_capitals() -> Dict[str, Any]:
    docs = [
        {"text": "Paris is the capital and most populous city of France.", "metadata": {"source": "capitals"}},
        {"text": "Tokyo is the capital of Japan and one of the most populous metropolitan areas in the world.", "metadata": {"source": "capitals"}},
        {"text": "Canberra, not Sydney, is the capital city of Australia.", "metadata": {"source": "capitals"}},
        {"text": "Ottawa is the capital city of Canada, located in the province of Ontario.", "metadata": {"source": "capitals"}},
        {"text": "Cairo is the capital of Egypt and the largest city in the Arab world.", "metadata": {"source": "capitals"}},
        {"text": "Brasília is the federal capital of Brazil, purpose-built in the late 1950s.", "metadata": {"source": "capitals"}},
        {"text": "New Delhi is the capital of India, distinct from the larger city of Delhi.", "metadata": {"source": "capitals"}},
        {"text": "Berlin is the capital and largest city of Germany.", "metadata": {"source": "capitals"}},
    ]
    questions = [
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is the capital of Australia?",
        "What is the capital of Canada?",
        "What is the capital of Egypt?",
        "What is the capital of Brazil?",
        "What is the capital of India?",
        "What is the capital of Germany?",
    ]
    answers = ["Paris", "Tokyo", "Canberra", "Ottawa", "Cairo", "Brasília", "New Delhi", "Berlin"]
    return {"documents": docs, "questions": questions, "reference_answers": answers}


def _bundled_tech_faq() -> Dict[str, Any]:
    docs = [
        {"text": "RAG (Retrieval-Augmented Generation) grounds an LLM's answer in documents retrieved "
                  "from an external knowledge base at inference time, reducing hallucination.", "metadata": {"source": "faq"}},
        {"text": "A vector database stores embeddings — numeric representations of text — and supports "
                  "similarity search using metrics like cosine similarity or dot product.", "metadata": {"source": "faq"}},
        {"text": "Chunking splits long documents into smaller passages so they fit an embedding model's "
                  "context window and improve retrieval precision.", "metadata": {"source": "faq"}},
        {"text": "Reranking applies a second, more expensive model to reorder an initial set of retrieved "
                  "candidates, trading latency for higher precision.", "metadata": {"source": "faq"}},
        {"text": "HyDE (Hypothetical Document Embeddings) asks an LLM to draft a hypothetical answer first, "
                  "then embeds that answer to retrieve more semantically relevant chunks.", "metadata": {"source": "faq"}},
        {"text": "Hybrid search combines sparse lexical retrieval (e.g. BM25) with dense vector retrieval to "
                  "capture both exact keyword matches and semantic similarity.", "metadata": {"source": "faq"}},
        {"text": "Corrective RAG (CRAG) grades retrieved documents for relevance and falls back to web search "
                  "or query rewriting when the retrieved context is judged insufficient.", "metadata": {"source": "faq"}},
        {"text": "GraphRAG builds a knowledge graph from a corpus and answers global, thematic questions by "
                  "summarizing communities of related entities rather than individual passages.", "metadata": {"source": "faq"}},
    ]
    questions = [
        "What does RAG stand for and what problem does it solve?",
        "What does a vector database store and how does it search?",
        "Why do RAG pipelines chunk documents?",
        "What is reranking and what tradeoff does it make?",
        "How does HyDE improve retrieval?",
        "What is hybrid search combining?",
        "How does Corrective RAG (CRAG) decide to fall back to web search?",
        "What is GraphRAG used for?",
    ]
    answers = [
        "Retrieval-Augmented Generation; it grounds LLM answers in retrieved documents to reduce hallucination.",
        "It stores embeddings and supports similarity search (e.g. cosine similarity).",
        "To fit the embedding model's context window and improve retrieval precision.",
        "Reordering retrieved candidates with a more expensive model, trading latency for precision.",
        "By embedding a hypothetical LLM-drafted answer instead of the raw query.",
        "Sparse lexical retrieval (BM25) with dense vector retrieval.",
        "By grading retrieved documents for relevance and falling back when context is insufficient.",
        "Answering global/thematic questions by summarizing graph communities of related entities.",
    ]
    return {"documents": docs, "questions": questions, "reference_answers": answers}


def _hf_loader(hf_name: str, hf_config: Optional[str], split: str,
               question_key: str, context_key: str, answer_key: str,
               n: int, answer_is_list: bool = False) -> Callable[[int], Dict[str, Any]]:
    def _load(limit: int = n) -> Dict[str, Any]:
        try:
            from datasets import load_dataset as hf_load
        except ImportError as e:
            raise ImportError(
                f"Loading '{hf_name}' requires the optional 'datasets' package: "
                f"pip install ragarena[datasets]"
            ) from e
        split_expr = f"{split}[:{limit}]"
        ds = hf_load(hf_name, hf_config, split=split_expr) if hf_config else hf_load(hf_name, split=split_expr)

        docs, questions, answers, seen = [], [], [], set()
        for row in ds:
            ctx = row.get(context_key)
            if isinstance(ctx, dict):  # e.g. hotpot_qa context = {"title":[...], "sentences":[[...]]}
                ctx = " ".join(" ".join(s) for s in ctx.get("sentences", []))
            if not ctx or ctx in seen:
                continue
            seen.add(ctx)
            docs.append({"text": ctx, "metadata": {"source": hf_name}})
        for row in ds:
            q = row.get(question_key)
            a = row.get(answer_key)
            if answer_is_list and isinstance(a, dict):
                a = (a.get("text") or [None])[0]
            elif isinstance(a, list):
                a = a[0] if a else None
            if q:
                questions.append(q)
                answers.append(a)
        return {"documents": docs or [{"text": "(no context field found)"}],
                "questions": questions, "reference_answers": answers}
    return _load


DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "capitals": {
        "description": "8 bundled world-capital QA pairs — offline, no dependencies.",
        "loader": lambda limit=8: _bundled_capitals(),
        "size": 8, "offline": True,
    },
    "rag_faq": {
        "description": "8 bundled RAG/IR concept QA pairs — offline, no dependencies.",
        "loader": lambda limit=8: _bundled_tech_faq(),
        "size": 8, "offline": True,
    },
    "squad": {
        "description": "SQuAD v1.1 — Wikipedia reading-comprehension QA (HuggingFace 'squad').",
        "loader": _hf_loader("squad", None, "validation", "question", "context", "answers", 50, answer_is_list=True),
        "size": "10570 (val)", "offline": False,
    },
    "hotpotqa": {
        "description": "HotpotQA — multi-hop Wikipedia QA requiring reasoning across documents.",
        "loader": _hf_loader("hotpot_qa", "distractor", "validation", "question", "context", "answer", 50),
        "size": "7405 (val)", "offline": False,
    },
    "natural_questions": {
        "description": "Natural Questions (open) — real Google search questions over Wikipedia.",
        "loader": _hf_loader("google-research-datasets/nq_open", None, "validation", "question", "question", "answer", 50, answer_is_list=True),
        "size": "3610 (val)", "offline": False,
    },
    "triviaqa": {
        "description": "TriviaQA (rc.nocontext) — trivia questions with free-form answers.",
        "loader": _hf_loader("trivia_qa", "rc.nocontext", "validation", "question", "question", "answer", 50, answer_is_list=True),
        "size": "9960 (val)", "offline": False,
    },
    "ms_marco": {
        "description": "MS MARCO v2.1 — passage-ranking QA from real Bing search queries.",
        "loader": _hf_loader("microsoft/ms_marco", "v2.1", "validation", "query", "passages", "answers", 50, answer_is_list=True),
        "size": "~101k (val)", "offline": False,
    },
}


# Datasets that ship with the package and need no network/HuggingFace access.
# Derived from the registry so it can't drift out of sync with it.
BUNDLED_DATASETS = frozenset(
    name for name, meta in DATASET_REGISTRY.items() if meta.get("offline")
)


def list_datasets() -> List[Dict[str, Any]]:
    return [{"name": k, **{kk: vv for kk, vv in v.items() if kk != "loader"}}
            for k, v in DATASET_REGISTRY.items()]


def load_dataset(
    name: str, n: int = 25, use_bundled: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str], List[Optional[str]]]:
    """Load a dataset by name.

    `n` caps documents/questions for HF-backed sets. When `use_bundled` is set
    and `name` isn't itself a bundled entry, an offline bundled dataset
    ('rag_faq') is substituted — useful with no network/HF access, but it is a
    DIFFERENT corpus, so a warning is emitted rather than silently returning
    rag_faq content labelled as, say, SQuAD.
    Returns ``(documents, questions, reference_answers)``.
    """
    if use_bundled and name not in BUNDLED_DATASETS:
        import warnings
        warnings.warn(
            f"load_dataset({name!r}, use_bundled=True): '{name}' is not a bundled dataset, "
            f"returning the bundled 'rag_faq' corpus instead — results are NOT from '{name}'. "
            f"Bundled datasets: {', '.join(sorted(BUNDLED_DATASETS))}. "
            f"Pass use_bundled=False to load the real '{name}' from HuggingFace.",
            UserWarning, stacklevel=2,
        )
        name = "rag_faq"
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {', '.join(DATASET_REGISTRY)}")
    data = DATASET_REGISTRY[name]["loader"](n)
    return data["documents"], data["questions"], data["reference_answers"]
