"""
Unified model router — one API, every provider.

Works::

    from ragarena import completion, embedding

    resp = completion(model="openai/gpt-4o-mini", messages=[{"role":"user","content":"hi"}])
    vecs = embedding(model="voyage/voyage-3", input=["hello world"])

completion() is backed by LiteLLM, which unifies auth/request-shaping across
100+ providers; ragarena still resolves API keys itself (for the
MissingAPIKeyError UX) and computes cost from its own catalog pricing.
embedding()/rerank() keep direct SDK integrations (voyage/cohere/local HF).
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

try:
    from dotenv import load_dotenv
    # override=True: a project-local .env should win over stray same-named vars
    # already set in the shell (e.g. unrelated tools exporting GEMINI_API_KEY).
    load_dotenv(override=True)
except ImportError:  # pragma: no cover — optional convenience only
    pass

from .catalog import PROVIDERS, EMBEDDING_PROVIDERS, get_model, parse_model_id


# ──────────────────────────────────────────────────────────────────────────────
# Response types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 8),
        }


@dataclass
class ModelResponse:
    text: str
    model: str
    usage: Usage
    latency_s: float
    finish_reason: Optional[str] = None
    raw: Optional[Any] = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "latency_s": round(self.latency_s, 4),
            "finish_reason": self.finish_reason,
        }


@dataclass
class EmbeddingResponse:
    vectors: List[List[float]]
    model: str
    usage: Usage
    latency_s: float

    def to_dict(self) -> dict:
        return {"model": self.model, "dim": len(self.vectors[0]) if self.vectors else 0,
                "count": len(self.vectors), "usage": self.usage.to_dict(),
                "latency_s": round(self.latency_s, 4)}


# ──────────────────────────────────────────────────────────────────────────────
# Client resolution
# ──────────────────────────────────────────────────────────────────────────────

_client_cache: Dict[tuple, OpenAI] = {}

_KEY_ENV_ALIASES = {
    "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "xai": ["XAI_API_KEY", "GROK_API_KEY"],
    "huggingface": ["HF_TOKEN", "HUGGINGFACE_API_KEY"],
    "azure": ["AZURE_API_KEY", "AZURE_OPENAI_API_KEY"],
    "vertex": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
}


class MissingAPIKeyError(RuntimeError):
    """Raised when a provider needs credentials and none were found in the environment."""

    def __init__(self, provider: str):
        cfg = PROVIDERS.get(provider) or EMBEDDING_PROVIDERS.get(provider)
        env_names = ([cfg.api_key_env] if cfg and cfg.api_key_env else []) + _KEY_ENV_ALIASES.get(provider, [])
        env_names = [e for e in dict.fromkeys(env_names) if e]  # dedupe, drop empties
        hint = (f"set one of: {', '.join(env_names)}" if env_names
                else "check ragarena.list_providers() for the expected credential")
        super().__init__(
            f"No API key found for provider '{provider}' — {hint} "
            f"(as an environment variable, in a .env file, or passed as api_key=...)."
        )


def _resolve_api_key(provider: str, api_key: Optional[str]) -> Optional[str]:
    if api_key:
        return api_key
    cfg = PROVIDERS.get(provider)
    env = cfg.api_key_env if cfg else ""
    for candidate in [env] + _KEY_ENV_ALIASES.get(provider, []):
        if candidate and os.getenv(candidate):
            return os.getenv(candidate)
    return None


def _get_openai_compatible_client(provider: str, api_key: Optional[str], base_url_override: Optional[str] = None) -> OpenAI:
    cfg = PROVIDERS[provider]
    resolved_key = _resolve_api_key(provider, api_key)
    if not resolved_key and cfg.requires_key:
        raise MissingAPIKeyError(provider)
    key = resolved_key or "sk-no-key-required"   # ollama/vllm/etc. don't need a real key
    cache_key = (provider, key, base_url_override)
    if cache_key not in _client_cache:
        _client_cache[cache_key] = OpenAI(
            api_key=key,
            base_url=base_url_override or cfg.base_url or None,
            timeout=120.0,
        )
    return _client_cache[cache_key]


# ──────────────────────────────────────────────────────────────────────────────
# Public router API — completion()
# ──────────────────────────────────────────────────────────────────────────────

def completion(
    model: str,
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    timeout: float = 120.0,
    **kwargs: Any,
) -> ModelResponse:
    """
    Call any supported LLM. ``model`` is ``provider/name``.

    Examples::

        completion(model="openai/gpt-4o-mini", messages=[...])
        completion(model="anthropic/claude-3-haiku-20240307", messages=[...])
        completion(model="groq/openai/gpt-oss-20b", messages=[...])
        completion(model="ollama/llama3.1", messages=[...])   # local, free
    """
    provider, model_name = parse_model_id(model)
    info = get_model(model)

    # Azure AI Foundry's "models/chat/completions" REST surface is bespoke
    # enough (Bearer token, no fixed deployment) that it's kept as a direct
    # REST call rather than mapped through LiteLLM.
    if provider == "azure_foundry":
        return _azure_foundry_completion(model, model_name, messages, api_key, api_base, temperature, max_tokens)

    key = _resolve_api_key(provider, api_key)
    cfg = PROVIDERS.get(provider)
    if not key and (cfg is None or cfg.requires_key):
        raise MissingAPIKeyError(provider)

    litellm_model, extra = _litellm_params(provider, model_name, api_base, key)
    import litellm
    litellm.suppress_debug_info = True

    token_limit = max_tokens or info.max_output_tokens
    call_kwargs: Dict[str, Any] = {
        "model": litellm_model, "messages": messages, "temperature": temperature,
        "max_tokens": token_limit, "timeout": timeout, **extra,
    }

    t0 = time.perf_counter()
    resp = None
    last_err: Optional[Exception] = None
    rate_limit_retries = 0
    # newer "reasoning" model deployments (o1/o3/gpt-5-style, etc.) often
    # reject standard params like max_tokens/temperature — retry stripping
    # whichever single param the provider names, up to a few attempts.
    for _ in range(4 + 3):  # + headroom for rate-limit retries (don't burn param-fix attempts)
        try:
            resp = litellm.completion(**call_kwargs)
            break
        except Exception as e:
            last_err = e
            msg = str(e)
            is_rate_limit = "RateLimitError" in type(e).__name__ or "rate_limit" in msg.lower()
            if is_rate_limit and rate_limit_retries < 3:
                rate_limit_retries += 1
                wait_s = 1.5 * rate_limit_retries  # default backoff
                m = re.search(r"try again in ([\d.]+)s", msg)
                if m:
                    wait_s = min(float(m.group(1)) + 0.5, 30.0)  # provider's own suggested wait, capped
                time.sleep(wait_s)
            elif "max_tokens" in msg and "max_tokens" in call_kwargs:
                call_kwargs["max_completion_tokens"] = call_kwargs.pop("max_tokens")
            elif "temperature" in msg and "temperature" in call_kwargs:
                call_kwargs.pop("temperature")
            else:
                raise RuntimeError(f"[{model}] request failed: {e}") from e
    if resp is None:
        raise RuntimeError(f"[{model}] request failed: {last_err}") from last_err
    latency = time.perf_counter() - t0

    choice = resp.choices[0]
    usage_raw = getattr(resp, "usage", None)
    usage = Usage(
        prompt_tokens=getattr(usage_raw, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage_raw, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage_raw, "total_tokens", 0) or 0,
    )
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

    from .catalog import estimate_cost
    usage.cost_usd = estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)

    return ModelResponse(
        text=choice.message.content or "",
        model=model,
        usage=usage,
        latency_s=latency,
        finish_reason=choice.finish_reason,
    )


# Provider slugs where our catalog's name differs from LiteLLM's own prefix
# convention (https://docs.litellm.ai/docs/providers). Anything not listed
# here is passed through as "<provider>/<model_name>" unchanged — LiteLLM
# uses the same "provider/model" shape we do for most providers.
_LITELLM_PREFIX = {
    "google": "gemini",
    "together": "together_ai",
    "fireworks": "fireworks_ai",
    "nvidia_nim": "nvidia_nim",
    "vertex": "vertex_ai",
}
# Self-hosted / custom-endpoint providers: reached via LiteLLM's generic
# OpenAI-compatible route, pointed at our own base_url/env override.
_SELF_HOSTED = {"ollama", "vllm", "lmstudio", "llamacpp", "custom_openai"}


def _litellm_params(provider: str, model_name: str, api_base: Optional[str],
                    key: Optional[str]) -> tuple[str, dict]:
    """Map a ragarena (provider, model_name) pair to a LiteLLM model string
    plus any extra kwargs (api_key/api_base/api_version) it needs."""
    cfg = PROVIDERS.get(provider)
    extra: Dict[str, Any] = {}
    if key:
        extra["api_key"] = key

    if provider == "azure":
        endpoint = (api_base or os.getenv("AZURE_OPENAI_ENDPOINT")
                    or os.getenv("AZURE_ENDPOINT") or os.getenv("AZURE_API_BASE"))
        if not endpoint:
            raise RuntimeError("Azure requires AZURE_OPENAI_ENDPOINT (e.g. https://<resource>.services.ai.azure.com)")
        extra["api_base"] = endpoint
        extra["api_version"] = (os.getenv("AZURE_OPENAI_API_VERSION")
                                or os.getenv("AZURE_API_VERSION") or "2024-05-01-preview")
        return f"azure/{model_name}", extra

    if provider in _SELF_HOSTED:
        extra["api_base"] = api_base or (cfg.base_url if cfg else None)
        return f"openai/{model_name}", extra

    if api_base:
        extra["api_base"] = api_base
    litellm_provider = _LITELLM_PREFIX.get(provider, provider)
    return f"{litellm_provider}/{model_name}", extra


def _azure_foundry_completion(model_id, model_name, messages, api_key, api_base, temperature, max_tokens) -> ModelResponse:
    """Azure AI Foundry 'models/chat/completions' REST endpoint (Bearer token)."""
    import httpx

    base = api_base or PROVIDERS["azure_foundry"].base_url
    api_version = os.getenv("AZURE_FOUNDRY_API_VERSION") or "2024-05-01-preview"
    key = _resolve_api_key("azure_foundry", api_key)
    if not key:
        raise RuntimeError("Azure AI Foundry requires AZURE_FOUNDRY_KEY")
    if not base:
        raise RuntimeError("Azure AI Foundry requires a base URL")
    from .catalog import estimate_cost

    url = f"{base.rstrip('/')}/chat/completions?api-version={api_version}"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": model_name, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens or 2048}
    t0 = time.perf_counter()
    resp = httpx.post(url, headers=headers, json=payload, timeout=120)
    latency = time.perf_counter() - t0
    if resp.status_code != 200:
        raise RuntimeError(f"[{model_id}] Azure Foundry request failed: {resp.status_code} - {resp.text[:300]}")
    data = resp.json()
    choice = data["choices"][0]
    text = choice["message"]["content"] or ""
    u = data.get("usage", {})
    pt = u.get("prompt_tokens", 0) or 0
    ct = u.get("completion_tokens", 0) or 0
    usage = Usage(pt, ct, pt + ct)
    usage.cost_usd = estimate_cost(model_id, pt, ct)
    return ModelResponse(text=text, model=model_id, usage=usage, latency_s=latency,
                         finish_reason=choice.get("finish_reason"))


# ──────────────────────────────────────────────────────────────────────────────
# Public router API — embedding()
# ──────────────────────────────────────────────────────────────────────────────

def embedding(
    model: str,
    input: List[str],
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    batch_size: int = 96,
) -> EmbeddingResponse:
    """Embed texts with any supported embedding model ('provider/name')."""
    provider, model_name = parse_model_id(model)
    get_model(model)  # validate

    all_vectors: List[List[float]] = []
    total_prompt_tokens = 0
    t0 = time.perf_counter()

    if provider in ("huggingface",):
        vectors = _hf_local_embed(model_name, input)
        all_vectors.extend(vectors)
        total_prompt_tokens = sum(len(t.split()) for t in input)

    elif provider == "ollama":
        client = _get_openai_compatible_client("ollama", None)
        for i in range(0, len(input), batch_size):
            batch = input[i:i + batch_size]
            r = client.embeddings.create(model=model_name, input=batch)
            all_vectors.extend(d.embedding for d in r.data)

    else:
        client_cfg = EMBEDDING_PROVIDERS.get(provider) or PROVIDERS.get(provider)
        if client_cfg is None:
            raise ValueError(f"No embedding route for provider '{provider}'")

        if provider == "voyage":
            import voyageai
            vc = voyageai.Client(api_key=_resolve_api_key("voyage", api_key))
            r = vc.embed(input, model=model_name, input_type="document")
            all_vectors.extend(r.embeddings)
            total_prompt_tokens = r.total_tokens
        elif provider == "cohere":
            import cohere
            cc = cohere.Client(_resolve_api_key("cohere", api_key))
            out_type = "search_document"
            for i in range(0, len(input), batch_size):
                batch = input[i:i + batch_size]
                r = cc.embed(texts=batch, model=model_name,
                             input_type=out_type, embedding_types=["float"])
                all_vectors.extend(r.embeddings.float_[0])
                total_prompt_tokens += sum(len(t.split()) for t in batch)
        else:
            client = _get_openai_compatible_client(provider, api_key, api_base)
            for i in range(0, len(input), batch_size):
                batch = input[i:i + batch_size]
                r = client.embeddings.create(model=model_name, input=batch)
                all_vectors.extend(d.embedding for d in r.data)
                total_prompt_tokens += sum(max(1, len(t) // 4) for t in batch)

    latency = time.perf_counter() - t0
    usage = Usage(prompt_tokens=total_prompt_tokens, completion_tokens=0,
                  total_tokens=total_prompt_tokens)
    from .catalog import estimate_cost
    usage.cost_usd = estimate_cost(model, total_prompt_tokens)

    return EmbeddingResponse(vectors=all_vectors, model=model, usage=usage, latency_s=latency)


_hf_model_cache: dict = {}

def _hf_local_embed(model_path: str, texts: List[str]) -> List[List[float]]:
    """Local sentence-transformers inference."""
    if model_path not in _hf_model_cache:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                f"'{model_path}' needs local inference: pip install sentence-transformers torch"
            ) from e
        _hf_model_cache[model_path] = SentenceTransformer(model_path)
    st = _hf_model_cache[model_path]
    embs = st.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return [e.tolist() for e in embs]


def rerank(
    model: str,
    query: str,
    documents: List[str],
    top_n: Optional[int] = None,
    api_key: Optional[str] = None,
) -> List[dict]:
    """Rerank documents against a query. Returns [{'index', 'relevance_score'}]."""
    provider, model_name = parse_model_id(model)
    top_n = top_n or len(documents)

    if provider == "huggingface":
        from sentence_transformers import CrossEncoder
        key = f"xenc::{model}"
        if key not in _hf_model_cache:
            _hf_model_cache[key] = CrossEncoder(model_name)
        scores = _hf_model_cache[key].predict([(query, d) for d in documents])
        ranked = sorted(
            ({"index": i, "relevance_score": float(s)} for i, s in enumerate(scores)),
            key=lambda x: x["relevance_score"], reverse=True,
        )
        return ranked[:top_n]

    if provider == "cohere":
        import cohere
        cc = cohere.Client(_resolve_api_key("cohere", api_key))
        r = cc.rerank(query=query, documents=documents, model=model_name, top_n=top_n)
        return [{"index": x.index, "relevance_score": x.relevance_score} for x in r.results]

    if provider == "voyage":
        import voyageai
        vc = voyageai.Client(api_key=_resolve_api_key("voyage", api_key))
        r = vc.rerank(documents, query, model=model_name, top_k=top_n)
        return [{"index": x.index, "relevance_score": x.relevance_score} for x in r.results]

    raise ValueError(f"No rerank route for '{provider}'. Supported: huggingface, cohere, voyage")
