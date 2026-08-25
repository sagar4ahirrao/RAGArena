"""
Unified model router — one API, every provider.

Works::

    from ragarena import completion, embedding

    resp = completion(model="openai/gpt-4o-mini", messages=[{"role":"user","content":"hi"}])
    vecs = embedding(model="voyage/voyage-3", input=["hello world"])

Any OpenAI-compatible provider is called through the OpenAI SDK with a
base-url override; Anthropic/Cohere/Google/Bedrock use their native SDKs.
"""
from __future__ import annotations

import os
import time
import base64
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

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


def _resolve_api_key(provider: str, api_key: Optional[str]) -> Optional[str]:
    if api_key:
        return api_key
    cfg = PROVIDERS.get(provider)
    env = cfg.api_key_env if cfg else ""
    aliases = {
        "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "xai": ["XAI_API_KEY", "GROK_API_KEY"],
        "huggingface": ["HF_TOKEN", "HUGGINGFACE_API_KEY"],
        "azure": ["AZURE_API_KEY", "AZURE_OPENAI_API_KEY"],
        "vertex": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    }
    for candidate in [env] + aliases.get(provider, []):
        if candidate and os.getenv(candidate):
            return os.getenv(candidate)
    return None


def _get_openai_compatible_client(provider: str, api_key: Optional[str], base_url_override: Optional[str] = None) -> OpenAI:
    cfg = PROVIDERS[provider]
    resolved_key = _resolve_api_key(provider, api_key)
    key = resolved_key or "sk-no-key-required"   # ollama/vllm need a dummy
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
        completion(model="groq/llama-3.1-8b-instant", messages=[...])
        completion(model="ollama/llama3.1", messages=[...])   # local, free
    """
    provider, model_name = parse_model_id(model)
    info = get_model(model)

    # ── Native SDK providers ────────────────────────────────────────────────
    if provider == "anthropic":
        return _anthropic_completion(model, model_name, messages, api_key, temperature, max_tokens or info.max_output_tokens)
    if provider in ("cohere",) and kwargs.get("native"):
        return _cohere_completion(model, model_name, messages, api_key, temperature, max_tokens)
    if provider == "bedrock":
        return _bedrock_completion(model, model_name, messages, temperature, max_tokens)

    # ── Everything else is OpenAI-compatible (incl. gemini via AI studio) ──
    client = _get_openai_compatible_client(provider, api_key, api_base)
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    except Exception as e:
        raise RuntimeError(f"[{model}] request failed: {e}") from e
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


def _anthropic_completion(model_id, model_name, messages, api_key, temperature, max_tokens) -> ModelResponse:
    import anthropic
    client = anthropic.Anthropic(api_key=_resolve_api_key("anthropic", api_key))
    system = "\n".join(m["content"] for m in messages if m["role"] == "system") or ""
    chat = [m for m in messages if m["role"] != "system"]

    t0 = time.perf_counter()
    resp = client.messages.create(
        model=model_name, system=system, messages=chat,
        temperature=temperature, max_tokens=max_tokens,
    )
    latency = time.perf_counter() - t0

    text = "".join(block.text for block in resp.content if hasattr(block, "text"))
    usage = Usage(resp.usage.input_tokens, resp.usage.output_tokens,
                  resp.usage.input_tokens + resp.usage.output_tokens)
    from .catalog import estimate_cost
    usage.cost_usd = estimate_cost(model_id, usage.prompt_tokens, usage.completion_tokens)
    return ModelResponse(text=text, model=model_id, usage=usage, latency_s=latency)


def _cohere_completion(model_id, model_name, messages, api_key, temperature, max_tokens) -> ModelResponse:
    import cohere
    client = cohere.ClientV2(_resolve_api_key("cohere", api_key))
    t0 = time.perf_counter()
    resp = client.chat(model=model_name, messages=messages, temperature=temperature)
    latency = time.perf_counter() - t0
    text = resp.message.content[0].text if resp.message and resp.message.content else ""
    u = resp.usage or {}
    pt = getattr(getattr(u, "tokens", None), "input_tokens", 0) or 0
    ct = getattr(getattr(u, "tokens", None), "output_tokens", 0) or 0
    usage = Usage(pt, ct, pt + ct)
    from .catalog import estimate_cost
    usage.cost_usd = estimate_cost(model_id, pt, ct)
    return ModelResponse(text=text, model=model_id, usage=usage, latency_s=latency)


def _bedrock_completion(model_id, model_name, messages, temperature, max_tokens) -> ModelResponse:
    try:
        import boto3
    except ImportError as e:
        raise RuntimeError("AWS Bedrock requires boto3: pip install boto3") from e
    import re

    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    client = boto3.client("bedrock-runtime", region_name=region)
    service = model_name.split(".")[0]

    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    chat = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
            for m in messages if m["role"] != "system"]

    if service == "anthropic":
        body = json.dumps({"anthropic_version": "bedrock-2023-05-31",
                           "max_tokens": max_tokens or 4096,
                           "system": system, "messages": chat,
                           "temperature": temperature})
    elif service == "meta":
        prompt_text = "\n\n".join(m["content"] for m in messages)
        body = json.dumps({"prompt": prompt_text, "max_gen_len": max_tokens or 2048,
                           "temperature": temperature})
    else:  # titan / amazon
        prompt_text = "\n\n".join(m["content"] for m in messages)
        body = json.dumps({"inputText": prompt_text,
                           "textGenerationConfig": {"maxTokenCount": max_tokens or 4096,
                                                    "temperature": temperature}})

    t0 = time.perf_counter()
    resp = client.invoke_model(modelId=model_name, body=body)
    latency = time.perf_counter() - t0
    payload = json.loads(resp["body"].read())

    if service == "anthropic":
        text = "".join(b.get("text", "") for b in payload.get("content", []))
        pt, ct = payload.get("usage", {}).get("input_tokens", 0), payload.get("usage", {}).get("output_tokens", 0)
    elif service == "meta":
        text = payload.get("generation", "")
        pt = len(payload.get("prompt_token_count", 0) and [0] * payload["prompt_token_count"])
        ct = len([0] * payload.get("generation_token_count", 0))
    else:
        results = payload.get("results", [{}])
        text = results[0].get("outputText", "")
        pt = results[0].get("inputTextTokenCount", 0)
        ct = results[0].get("generationTokenCount", 0)

    usage = Usage(pt, ct, pt + ct)
    from .catalog import estimate_cost
    usage.cost_usd = estimate_cost(model_id, pt, ct)
    return ModelResponse(text=text, model=model_id, usage=usage, latency_s=latency)


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
            _hf_model_cache[key] = CrossEncoder(model)
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
