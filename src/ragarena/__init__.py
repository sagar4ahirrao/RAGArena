"""
⚡ RagArena — evaluate & benchmark RAG strategies, LLMs and embedding models
across every popular provider with one unified API.

Quickstart::

    from ragarena import evaluate, compare

    report = evaluate(
        questions=["What is RAG?"],
        documents=[{"text": "RAG = retrieval-augmented generation..."}],
        strategy="hybrid",                       # 13 strategies available
        model="openai/gpt-4o-mini",              # any provider/model
        embedding_model="voyage/voyage-3",
        metrics="quality",
    )
    report.print_summary()
"""
from .catalog import (
    ModelInfo, ProviderConfig,
    CHAT_MODELS, EMBEDDING_MODELS, RERANK_MODELS, VECTOR_STORES,
    PROVIDERS, EMBEDDING_PROVIDERS,
    get_model, has_model, list_models, list_providers, estimate_cost,
)
from .router import completion, embedding, rerank, ModelResponse, EmbeddingResponse, Usage
from .strategies import (
    Strategy, Chunk, StrategyResult, STRATEGIES, get_strategy,
    NaiveRAG, HybridRAG, MultiQueryRAG, HyDERAG, RerankRAG, RAGFusion,
    ContextualCompressionRAG, CRAGRAG, SelfRAG, QueryDecompositionRAG,
    StepBackRAG, AgenticRAG, FLARERAG,
    GraphLocalRAG, GraphGlobalRAG, GraphHybridRAG, GraphMixRAG, MultimodalRAG,
)
from .graph import GraphIndex
from .metrics import METRICS, DEFAULT_METRIC_SETS, MetricResult
from .index import VectorIndex, TextChunker, chunk_text, MultimodalDocument
from .engine import evaluate, compare, recommend_strategy, EvaluationReport, ComparisonResult, RecommendationResult
from .ingest import parse_file, parse_dir, to_multimodal, from_sql
from .datasets import load_dataset, list_datasets, DATASET_REGISTRY

__version__ = "0.2.2"
__author__ = "RagArena contributors"

__all__ = [
    # one-liner APIs
    "completion", "embedding", "rerank", "evaluate", "compare", "recommend_strategy",
    # catalog
    "list_models", "list_providers", "get_model", "has_model", "estimate_cost",
    "CHAT_MODELS", "EMBEDDING_MODELS", "RERANK_MODELS", "VECTOR_STORES",
    "PROVIDERS", "EMBEDDING_PROVIDERS", "ModelInfo",
    # strategies
    "STRATEGIES", "get_strategy", "Strategy", "Chunk", "StrategyResult",
    "NaiveRAG", "HybridRAG", "MultiQueryRAG", "HyDERAG", "RerankRAG", "RAGFusion",
    "ContextualCompressionRAG", "CRAGRAG", "SelfRAG", "QueryDecompositionRAG",
    "StepBackRAG", "AgenticRAG", "FLARERAG",
    "GraphLocalRAG", "GraphGlobalRAG", "GraphHybridRAG", "GraphMixRAG", "MultimodalRAG",
    # graph / multimodal
    "GraphIndex", "MultimodalDocument",
    # ingestion / datasets
    "parse_file", "parse_dir", "to_multimodal", "from_sql",
    "load_dataset", "list_datasets", "DATASET_REGISTRY",
    # metrics / index / engine
    "METRICS", "DEFAULT_METRIC_SETS", "MetricResult",
    "VectorIndex", "TextChunker", "chunk_text",
    "EvaluationReport", "ComparisonResult", "RecommendationResult",
]
