"""
RagArena Model Catalog — every popular LLM, embedding, reranker & vector store.

Models are addressed as ``provider/model-name``::

    RagArena.completion(model="openai/gpt-4o-mini", messages=[...])
    RagArena.completion(model="anthropic/claude-3-5-sonnet-20240620", ...)
    RagArena.embedding(model="voyage/voyage-3", input=["hello"])

Pricing is USD per 1M tokens. ``context`` is max input context window.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Literal


Modality = Literal["chat", "embedding", "rerank"]


@dataclass(frozen=True)
class ModelInfo:
    id: str                          # "provider/model-name"
    provider: str
    model_name: str                  # raw name sent to provider
    modality: Modality = "chat"
    context_window: int = 8192
    max_output_tokens: int = 4096
    input_cost: float = 0.0          # $ / 1M input tokens
    output_cost: float = 0.0         # $ / 1M output tokens
    supports_vision: bool = False
    supports_function_calling: bool = True
    openai_compatible: bool = True   # callable via OpenAI SDK w/ base_url override
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Provider registry — how to reach each provider
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProviderConfig:
    """Connection details for a provider."""
    name: str
    base_url: str = ""               # empty → use native SDK
    api_key_env: str = ""
    sdk: str = "openai"              # openai | anthropic | google | cohere | bedrock | ollama | hf | none
    requires_key: bool = True


PROVIDERS: Dict[str, ProviderConfig] = {
    # ── Major hosted APIs ──
    "openai":       ProviderConfig("OpenAI",       "https://api.openai.com/v1",                 "OPENAI_API_KEY"),
    "azure":        ProviderConfig("Azure OpenAI", "",                                          "AZURE_API_KEY",      sdk="azure"),
    "azure_foundry":ProviderConfig("Azure AI Foundry", "https://nb-models-resource.services.ai.azure.com/models", "AZURE_FOUNDRY_KEY"),
    "bedrock":      ProviderConfig("AWS Bedrock",  "",                                          "AWS_ACCESS_KEY_ID",  sdk="bedrock"),
    "vertex":       ProviderConfig("Google Vertex","https://aiplatform.googleapis.com/v1",      "GOOGLE_API_KEY"),
    "google":       ProviderConfig("Google AI Studio", 
"https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "anthropic":    ProviderConfig("Anthropic",    "https://api.anthropic.com/v1",              "ANTHROPIC_API_KEY",  sdk="anthropic"),
    "cohere":       ProviderConfig("Cohere",       "https://api.cohere.com",                    "COHERE_API_KEY",     sdk="cohere"),
    "mistral":      ProviderConfig("Mistral AI",   "https://api.mistral.ai/v1",                 "MISTRAL_API_KEY"),
    "xai":          ProviderConfig("xAI (Grok)",   "https://api.x.ai/v1",                       "XAI_API_KEY"),
    "deepseek":     ProviderConfig("DeepSeek",     "https://api.deepseek.com/v1",               "DEEPSEEK_API_KEY"),

    # ── Fast inference / aggregators ──
    "groq":         ProviderConfig("Groq",         "https://api.groq.com/openai/v1",            "GROQ_API_KEY"),
    "together":     ProviderConfig("Together AI",  "https://api.together.xyz/v1",               "TOGETHER_API_KEY"),
    "fireworks":    ProviderConfig("Fireworks AI", "https://api.fireworks.ai/inference/v1",     "FIREWORKS_API_KEY"),
    "deepinfra":    ProviderConfig("DeepInfra",    "https://api.deepinfra.com/v1/openai",       "DEEPINFRA_API_KEY"),
    "anyscale":     ProviderConfig("Anyscale",     "https://api.endpoints.anyscale.com/v1",     "ANYSCALE_API_KEY"),
    "openrouter":   ProviderConfig("OpenRouter",   "https://openrouter.ai/api/v1",              "OPENROUTER_API_KEY"),
    "perplexity":   ProviderConfig("Perplexity",   "https://api.perplexity.ai",                 "PERPLEXITYAI_API_KEY"),
    "nvidia_nim":   ProviderConfig("NVIDIA NIM",   "https://integrate.api.nvidia.com/v1",       "NVIDIA_API_KEY"),
    "cerebras":     ProviderConfig("Cerebras",     "https://api.cerebras.ai/v1",                "CEREBRAS_API_KEY"),
    "sambanova":    ProviderConfig("SambaNova",    "https://api.sambanova.ai/v1",               "SAMBANOVA_API_KEY"),
    "databricks":   ProviderConfig("Databricks",   "",                                          "DATABRICKS_API_KEY"),
    "ai21":         ProviderConfig("AI21 Labs",    "https://api.ai21.com/studio/v1",            "AI21_API_KEY"),
    "huggingface":  ProviderConfig("HuggingFace",  "https://api-inference.huggingface.co/v1",   "HF_TOKEN"),

    # ── Local / self-hosted ──
    "ollama":       ProviderConfig("Ollama",       "http://localhost:11434/v1",                 "", requires_key=False),
    "vllm":         ProviderConfig("vLLM",         "http://localhost:8000/v1",                  "", requires_key=False),
    "lmstudio":     ProviderConfig("LM Studio",    "http://localhost:1234/v1",                  "", requires_key=False),
    "llamacpp":     ProviderConfig("llama.cpp",    "http://localhost:8080/v1",                  "", requires_key=False),
    "custom_openai":ProviderConfig("Custom OpenAI-compatible", "",                              "CUSTOM_LLM_API_KEY"),
}

# ──────────────────────────────────────────────────────────────────────────────
# Embedding-only providers
# ──────────────────────────────────────────────────────────────────────────────

EMBEDDING_PROVIDERS: Dict[str, ProviderConfig] = {
    "voyage":  ProviderConfig("Voyage AI", "https://api.voyageai.com/v1", "VOYAGE_API_KEY"),
    "jina":    ProviderConfig("Jina AI",   "https://api.jina.ai/v1",      "JINA_API_KEY"),
    "nomic":   ProviderConfig("Nomic AI",  "https://api-atlas.nomic.ai/v1", "NOMIC_API_KEY"),
}


# ──────────────────────────────────────────────────────────────────────────────
# THE MODEL CATALOG
# ──────────────────────────────────────────────────────────────────────────────

def _m(id_, provider, name, **kw) -> ModelInfo:
    return ModelInfo(id=id_, provider=provider, model_name=name, **kw)


CHAT_MODELS: List[ModelInfo] = [
    # ── OpenAI ──
    _m("openai/gpt-4o", "openai", "gpt-4o", context_window=128000, max_output_tokens=16384,
       input_cost=2.50, output_cost=10.00, supports_vision=True, description="Flagship multimodal model"),
    _m("openai/gpt-4o-mini", "openai", "gpt-4o-mini", context_window=128000, max_output_tokens=16384,
       input_cost=0.15, output_cost=0.60, supports_vision=True, description="Fast, cheap multimodal workhorse"),
    _m("openai/gpt-4-turbo", "openai", "gpt-4-turbo", context_window=128000,
       input_cost=10.00, output_cost=30.00, supports_vision=True),
    _m("openai/gpt-4", "openai", "gpt-4", context_window=8192, input_cost=30.00, output_cost=60.00),
    _m("openai/gpt-3.5-turbo", "openai", "gpt-3.5-turbo", context_window=16385,
       input_cost=0.50, output_cost=1.50, description="Budget legacy model"),
    _m("openai/o1-preview", "openai", "o1-preview", context_window=128000, input_cost=15.00, output_cost=60.00,
       description="Reasoning model"),
    _m("openai/o1-mini", "openai", "o1-mini", context_window=128000, input_cost=3.00, output_cost=12.00),

    # ── Anthropic ──
    _m("anthropic/claude-3-5-sonnet-20240620", "anthropic", "claude-3-5-sonnet-20240620",
       context_window=200000, input_cost=3.00, output_cost=15.00, supports_vision=True,
       description="Best coding model class"),
    _m("anthropic/claude-3-opus-20240229", "anthropic", "claude-3-opus-20240229",
       context_window=200000, input_cost=15.00, output_cost=75.00, supports_vision=True,
       description="Most capable Claude"),
    _m("anthropic/claude-3-sonnet-20240229", "anthropic", "claude-3-sonnet-20240229",
       context_window=200000, input_cost=3.00, output_cost=15.00, supports_vision=True),
    _m("anthropic/claude-3-haiku-20240307", "anthropic", "claude-3-haiku-20240307",
       context_window=200000, input_cost=0.25, output_cost=1.25, supports_vision=True,
       description="Fastest Claude"),
    _m("bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0", "bedrock", "anthropic.claude-3-5-sonnet-20240620-v1:0",
       context_window=200000, input_cost=3.00, output_cost=15.00),
    _m("bedrock/anthropic.claude-3-haiku-20240307-v1:0", "bedrock", "anthropic.claude-3-haiku-20240307-v1:0",
       context_window=200000, input_cost=0.25, output_cost=1.25),
    _m("vertex/claude-3-5-sonnet@20240620", "vertex", "claude-3-5-sonnet@20240620",
       context_window=200000, input_cost=3.00, output_cost=15.00),

    # ── Google Gemini ──
    _m("google/gemini-1.5-pro", "google", "gemini-1.5-pro", context_window=2097152, max_output_tokens=8192,
       input_cost=1.25, output_cost=5.00, supports_vision=True, description="2M-token context"),
    _m("google/gemini-1.5-flash", "google", "gemini-1.5-flash", context_window=1048576,
        supports_vision=True, input_cost=0.075, output_cost=0.30),
    _m("google/gemini-1.5-pro", "google", "gemini-1.5-pro", context_window=2097152,
        supports_vision=True, input_cost=1.25, output_cost=5.00),
    _m("google/gemini-2.5-flash", "google", "gemini-2.5-flash", context_window=1048576,
        supports_vision=True, input_cost=0.30, output_cost=2.50, description="Fast Gemini 2.5"),
    _m("google/gemini-2.5-pro", "google", "gemini-2.5-pro", context_window=1048576,
        supports_vision=True, input_cost=1.25, output_cost=10.00, description="Strong Gemini 2.5"),
    _m("google/gemma-2-27b-it", "google", "gemma-2-27b-it", context_window=8192,
        description="Open-weight Gemma"),
    _m("vertex/gemini-1.5-pro", "vertex", "gemini-1.5-pro", context_window=2097152,
       input_cost=1.25, output_cost=5.00, supports_vision=True),

    # ── Groq-hosted models (verified live via /v1/models — Groq deprecates fast,
    #    llama-3.1-70b-versatile/8b-instant/mixtral-8x7b were retired) ──
    _m("groq/openai/gpt-oss-120b", "groq", "openai/gpt-oss-120b", context_window=131072,
        input_cost=0.15, output_cost=0.60, description="GPT-OSS 120B on Groq — best quality/speed tradeoff"),
    _m("groq/openai/gpt-oss-20b", "groq", "openai/gpt-oss-20b", context_window=131072,
        input_cost=0.10, output_cost=0.20, description="GPT-OSS 20B on Groq — fast + cheap"),
    _m("groq/qwen/qwen3.6-27b", "groq", "qwen/qwen3.6-27b", context_window=131072,
        input_cost=0.20, output_cost=0.20, description="Qwen 3.6 27B on Groq"),
    _m("groq/compound", "groq", "compound", context_window=131072,
        input_cost=0.0, output_cost=0.0, description="Groq compound router (agentic, tool-using)"),
    _m("groq/compound-mini", "groq", "compound-mini", context_window=131072,
        input_cost=0.0, output_cost=0.0, description="Groq compound router — smaller/faster"),

    # ── OpenRouter (400+ models behind one key) ──
    _m("openrouter/openai/gpt-4o-mini", "openrouter", "openai/gpt-4o-mini", context_window=128000,
        input_cost=0.15, output_cost=0.60, description="GPT-4o mini via OpenRouter"),
    _m("openrouter/anthropic/claude-3.5-sonnet", "openrouter", "anthropic/claude-3.5-sonnet",
        context_window=200000, input_cost=3.00, output_cost=15.00,
        description="Claude 3.5 Sonnet via OpenRouter"),
    _m("openrouter/meta-llama/llama-3.1-405b-instruct", "openrouter",
        "meta-llama/llama-3.1-405b-instruct", context_window=131072,
        input_cost=2.70, output_cost=2.70, description="Llama 3.1 405B via OpenRouter"),
    _m("openrouter/meta-llama/llama-3.1-70b-instruct", "openrouter",
        "meta-llama/llama-3.1-70b-instruct", context_window=131072,
        input_cost=0.59, output_cost=0.79, description="Llama 3.1 70B via OpenRouter"),
    _m("openrouter/meta-llama/llama-3.1-8b-instruct", "openrouter",
        "meta-llama/llama-3.1-8b-instruct", context_window=131072,
        input_cost=0.06, output_cost=0.06, description="Llama 3.1 8B — cheap & fast via OpenRouter"),
    _m("openrouter/deepseek/deepseek-chat", "openrouter", "deepseek/deepseek-chat",
        context_window=65536, input_cost=0.14, output_cost=0.28,
        description="DeepSeek V3 chat via OpenRouter"),
    _m("openrouter/google/gemini-flash-1.5", "openrouter", "google/gemini-flash-1.5",
        context_window=1048576, input_cost=0.075, output_cost=0.30,
        description="Gemini Flash 1.5 (1M ctx) via OpenRouter"),
    _m("openrouter/qwen/qwen-2.5-72b-instruct", "openrouter", "qwen/qwen-2.5-72b-instruct",
        context_window=32768, input_cost=0.35, output_cost=0.40,
        description="Qwen 2.5 72B via OpenRouter"),
    _m("openrouter/mistralai/mistral-7b-instruct", "openrouter", "mistralai/mistral-7b-instruct",
        context_window=32768, input_cost=0.05, output_cost=0.05,
        description="Mistral 7B — budget option via OpenRouter"),

    # ── Azure OpenAI / Azure AI Foundry ──
    _m("azure/nc-tools-v3.1", "azure", "nc-tools-v3.1", context_window=128000,
        input_cost=0.0, output_cost=0.0, description="Azure-hosted model (nb-llm-resource)"),
    _m("azure/DeepSeek-V3.2", "azure", "DeepSeek-V3.2", context_window=128000,
        input_cost=0.0, output_cost=0.0, description="Azure AI Foundry model"),
    _m("azure/Llama-4-Maverick-17B-128E-Instruct-FP8", "azure",
        "Llama-4-Maverick-17B-128E-Instruct-FP8", context_window=128000,
        input_cost=0.0, output_cost=0.0, description="Azure AI Foundry model"),
    _m("azure_foundry/Llama-4-Maverick-17B-128E-Instruct-FP8", "azure_foundry",
        "Llama-4-Maverick-17B-128E-Instruct-FP8", context_window=128000,
        input_cost=0.0, output_cost=0.0, description="Azure AI Foundry (models endpoint)"),
    _m("azure_foundry/DeepSeek-V3.2", "azure_foundry", "DeepSeek-V3.2", context_window=128000,
        input_cost=0.0, output_cost=0.0, description="Azure AI Foundry (models endpoint)"),
    _m("azure/gpt-5.5", "azure", "gpt-5.5", context_window=128000,
        input_cost=0.0, output_cost=0.0, description="Azure OpenAI deployment (nb-nc-llm resource)"),
    _m("together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", "together", "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
       context_window=130000, input_cost=3.50, output_cost=3.50, description="Biggest open model"),
    _m("together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "together", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
       context_window=131072, input_cost=0.88, output_cost=0.88),
    _m("together/meta-llama/Llama-3-8b-chat-hf", "together", "meta-llama/Llama-3-8b-chat-hf",
       context_window=8192, input_cost=0.20, output_cost=0.20),
    _m("fireworks/accounts/fireworks/models/llama-v3p1-70b-instruct", "fireworks",
       "accounts/fireworks/models/llama-v3p1-70b-instruct", context_window=131072,
       input_cost=0.90, output_cost=0.90),
    _m("deepinfra/meta-llama/Meta-Llama-3.1-70B-Instruct", "deepinfra", "meta-llama/Meta-Llama-3.1-70B-Instruct",
       context_window=131072, input_cost=0.35, output_cost=0.40),
    _m("bedrock/meta.llama3-1-405b-instruct-v1:0", "bedrock", "meta.llama3-1-405b-instruct-v1:0",
       context_window=128000, input_cost=5.32, output_cost=16.00),
    _m("ollama/llama3.1", "ollama", "llama3.1", description="Local Llama 3.1 (free)"),
    _m("ollama/llama3", "ollama", "llama3", description="Local Llama 3 (free)"),

    # ── Mistral ──
    _m("mistral/mistral-large-latest", "mistral", "mistral-large-latest", context_window=128000,
       input_cost=2.00, output_cost=6.00, description="Frontier Mistral"),
    _m("mistral/mistral-small-latest", "mistral", "mistral-small-latest", context_window=32768,
       input_cost=0.20, output_cost=0.60),
    _m("mistral/open-mistral-nemo", "mistral", "open-mistral-nemo", context_window=128000,
       input_cost=0.15, output_cost=0.15, description="Apache 2.0 open weights"),
    _m("mistral/codestral-latest", "mistral", "codestral-latest", context_window=32768,
       input_cost=0.20, output_cost=0.60, description="Code-specialized"),
    _m("together/mistralai/Mixtral-8x22B-Instruct-v0.1", "together", "mistralai/Mixtral-8x22B-Instruct-v0.1",
       context_window=65536, input_cost=1.20, output_cost=1.20),

    # ── DeepSeek ──
    _m("deepseek/deepseek-chat", "deepseek", "deepseek-chat", context_window=65536,
       input_cost=0.14, output_cost=0.28, description="DeepSeek V3 — incredible price/perf"),
    _m("deepseek/deepseek-coder", "deepseek", "deepseek-coder", context_window=65536,
       input_cost=0.14, output_cost=0.28, description="Code-specialized"),

    # ── xAI ──
    _m("xai/grok-beta", "xai", "grok-beta", context_window=131072,
       input_cost=5.00, output_cost=15.00, description="xAI flagship"),

    # ── Qwen / other open families ──
    _m("together/Qwen/Qwen2.5-72B-Instruct", "together", "Qwen/Qwen2.5-72B-Instruct",
       context_window=131072, input_cost=1.20, output_cost=1.20),
    _m("together/Qwen/Qwen2.5-7B-Instruct", "together", "Qwen/Qwen2.5-7B-Instruct",
       context_window=131072, input_cost=0.20, output_cost=0.20),
    _m("ollama/qwen2.5", "ollama", "qwen2.5", description="Local Qwen 2.5 (free)"),
    _m("ollama/phi3", "ollama", "phi3", description="Microsoft Phi-3 mini (free)"),
    _m("ollama/gemma2", "ollama", "gemma2", description="Local Gemma 2 (free)"),
    _m("ollama/mistral", "ollama", "mistral", description="Local Mistral 7B (free)"),
    _m("ollama/nous-hermes2", "ollama", "nous-hermes2", description="Nous Hermes 2 (free)"),

    # ── Cohere ──
    _m("cohere/command-r-plus", "cohere", "command-r-plus", context_window=128000,
       input_cost=2.50, output_cost=10.00, description="Enterprise RAG-tuned"),
    _m("cohere/command-r", "cohere", "command-r", context_window=128000,
       input_cost=0.15, output_cost=0.60, description="RAG-native features"),

    # ── Perplexity (search-grounded) ──
    _m("perplexity/llama-3.1-sonar-small-128k-online", "perplexity", "llama-3.1-sonar-small-128k-online",
       context_window=127072, input_cost=0.20, output_cost=0.20, description="Search-grounded answers"),
    _m("perplexity/llama-3.1-sonar-large-128k-online", "perplexity", "llama-3.1-sonar-large-128k-online",
       context_window=127072, input_cost=1.00, output_cost=1.00),

    # ── Amazon Titan / Nova on Bedrock ──
    _m("bedrock/amazon.titan-text-premier-v1:0", "bedrock", "amazon.titan-text-premier-v1:0",
       context_window=32000, input_cost=0.50, output_cost=1.50),
]

EMBEDDING_MODELS: List[ModelInfo] = [
    # ── OpenAI ──
    _m("openai/text-embedding-3-small", "openai", "text-embedding-3-small", modality="embedding",
       input_cost=0.02, description="Best cost/perf default (1536-d)"),
    _m("openai/text-embedding-3-large", "openai", "text-embedding-3-large", modality="embedding",
       input_cost=0.13, description="Highest quality OpenAI (3072-d)"),
    _m("openai/text-embedding-ada-002", "openai", "text-embedding-ada-002", modality="embedding",
       input_cost=0.10, description="Legacy (1536-d)"),

    # ── Cohere ──
    _m("cohere/embed-english-v3.0", "cohere", "embed-english-v3.0", modality="embedding",
       input_cost=0.10, description="English SOTA w/ int8 compression (1024-d)"),
    _m("cohere/embed-multilingual-v3.0", "cohere", "embed-multilingual-v3.0", modality="embedding",
       input_cost=0.10, description="100+ languages (1024-d)"),

    # ── Voyage AI ──
    _m("voyage/voyage-3", "voyage", "voyage-3", modality="embedding",
       input_cost=0.06, description="Strong general-purpose (1024-d)"),
    _m("voyage/voyage-3-large", "voyage", "voyage-3-large", modality="embedding",
       input_cost=0.18, description="Voyage highest quality"),
    _m("voyage/voyage-code-2", "voyage", "voyage-code-2", modality="embedding",
       input_cost=0.12, description="Code retrieval specialist"),
    _m("voyage/voyage-law-2", "voyage", "voyage-law-2", modality="embedding",
       input_cost=0.12, description="Legal domain"),
    _m("voyage/voyage-finance-2", "voyage", "voyage-finance-2", modality="embedding",
       input_cost=0.12, description="Finance domain"),

    # ── Jina ──
    _m("jina/jina-embeddings-v3", "jina", "jina-embeddings-v3", modality="embedding",
       input_cost=0.02, description="SOTA multilingual task-tuned (1024-d)"),
    _m("jina/jina-clip-v2", "jina", "jina-clip-v2", modality="embedding",
       input_cost=0.02, description="Text+image cross-modal"),

    # ── Mistral ──
    _m("mistral/mistral-embed", "mistral", "mistral-embed", modality="embedding",
       input_cost=0.10, description="1024-d multilingual"),

    # ── Google ──
    _m("google/text-embedding-004", "google", "text-embedding-004", modality="embedding",
        input_cost=0.0, description="Deprecated Gemini embeddings"),
    _m("google/gemini-embedding-001", "google", "gemini-embedding-001", modality="embedding",
        input_cost=0.0, description="Free Gemini embeddings (3072-d)"),

    # ── Bedrock Titan ──
    _m("bedrock/amazon.titan-embed-text-v2:0", "bedrock", "amazon.titan-embed-text-v2:0",
       modality="embedding", input_cost=0.02, description="Matryoshka dims 256/512/1024"),

    # ── Local (free) ──
    _m("huggingface/sentence-transformers/all-MiniLM-L6-v2", "huggingface", "sentence-transformers/all-MiniLM-L6-v2",
       modality="embedding", description="384-d classic baseline, runs anywhere"),
    _m("huggingface/BAAI/bge-large-en-v1.5", "huggingface", "BAAI/bge-large-en-v1.5",
       modality="embedding", description="1024-d strong English"),
    _m("huggingface/BAAI/bge-m3", "huggingface", "BAAI/bge-m3",
       modality="embedding", description="Multilingual + long-context (1024-d)"),
    _m("huggingface/intfloat/e5-large-v2", "huggingface", "intfloat/e5-large-v2",
       modality="embedding", description="1024-d general purpose"),
    _m("huggingface/Alibaba-NLP/gte-large-en-v1.5", "huggingface", "Alibaba-NLP/gte-large-en-v1.5",
       modality="embedding", description="1024-d MTEB leader-tier"),
    _m("ollama/nomic-embed-text", "ollama", "nomic-embed-text", modality="embedding",
       description="768-d local embedding (free)"),
    _m("ollama/mxbai-embed-large", "ollama", "mxbai-embed-large", modality="embedding",
       description="1024-d local (free)"),
]

RERANK_MODELS: List[ModelInfo] = [
    _m("cohere/rerank-v3.5", "cohere", "rerank-v3.5", modality="rerank",
       input_cost=2.00, description="Industry-standard hosted reranker"),
    _m("voyage/rerank-2", "voyage", "rerank-2", modality="rerank",
       input_cost=0.05, description="High accuracy reranker"),
    _m("huggingface/BAAI/bge-reranker-v2-m3", "huggingface", "BAAI/bge-reranker-v2-m3",
       modality="rerank", description="Free local cross-encoder"),
    _m("huggingface/cross-encoder/ms-marco-MiniLM-L-6-v2", "huggingface", "cross-encoder/ms-marco-MiniLM-L-6-v2",
       modality="rerank", description="Free lightweight reranker"),
]

VECTOR_STORES: Dict[str, str] = {
    "faiss":     "In-memory/local index — zero infra, fastest for experiments",
    "chroma":    "Embedded OSS DB with persistence — great dev experience",
    "pinecone":  "Fully-managed serverless cloud — zero ops at scale",
    "qdrant":    "Rust-based, self-host or cloud — rich filtering",
    "weaviate":  "OSS with built-in hybrid search + modules",
    "milvus":    "Distributed OSS for billion-scale",
    "lancedb":   "Serverless embedded columnar store",
    "pgvector":  "Postgres extension — SQL joins + vectors",
    "elasticsearch": "Hybrid BM25+kNN battle-tested search",
    "redis":     "Sub-ms vector search in cache",
    "opensearch":"AWS managed hybrid search",
    "mongodb":   "Atlas Vector Search",
}


# ──────────────────────────────────────────────────────────────────────────────
# Lookup helpers
# ──────────────────────────────────────────────────────────────────────────────

_ALL_MODELS: Dict[str, ModelInfo] = {}
for _lst in (CHAT_MODELS, EMBEDDING_MODELS, RERANK_MODELS):
    for _mod in _lst:
        _ALL_MODELS[_mod.id] = _mod


class UnknownModelError(KeyError):
    def __init__(self, model_id: str):
        super().__init__(
            f"Unknown model '{model_id}'. "
            f"Browse the catalog: `ragarena models list`, or call ragarena.list_models()."
        )


def get_model(model_id: str) -> ModelInfo:
    """Look up a model by 'provider/name'. Raises :class:`UnknownModelError`."""
    if model_id not in _ALL_MODELS:
        raise UnknownModelError(model_id)
    return _ALL_MODELS[model_id]


def has_model(model_id: str) -> bool:
    return model_id in _ALL_MODELS


def parse_model_id(model_id: str) -> tuple[str, str]:
    """Split 'provider/model' → ('provider', 'model')."""
    if "/" not in model_id:
        raise ValueError(
            f"Model must be 'provider/model' format, got '{model_id}'. Example: openai/gpt-4o-mini"
        )
    provider, name = model_id.split("/", 1)
    return provider, name


def list_models(
    provider: Optional[str] = None,
    modality: Optional[Modality] = None,
) -> List[ModelInfo]:
    out = list(_ALL_MODELS.values())
    if provider:
        out = [m for m in out if m.provider == provider]
    if modality:
        out = [m for m in out if m.modality == modality]
    return sorted(out, key=lambda m: m.id)


def list_providers() -> List[dict]:
    seen = sorted({m.provider for m in _ALL_MODELS.values()})
    return [
        {
            "provider": p,
            "display_name": PROVIDERS.get(p, EMBEDDING_PROVIDERS.get(p)).name
                            if (p in PROVIDERS or p in EMBEDDING_PROVIDERS) else p,
            "models": len([m for m in _ALL_MODELS.values() if m.provider == p]),
        }
        for p in seen
    ]


def estimate_cost(model_id: str, prompt_tokens: int, completion_tokens: int = 0) -> float:
    m = get_model(model_id)
    if m.modality == "chat":
        return prompt_tokens * m.input_cost / 1e6 + completion_tokens * m.output_cost / 1e6
    return prompt_tokens * m.input_cost / 1e6
