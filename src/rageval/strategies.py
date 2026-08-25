"""
RAG strategy library — 13 battle-tested retrieval+generation pipelines.

Every strategy implements the same interface::

    result = strategy.run(query, index, llm_model, embedding_model, **cfg)

so they are interchangeable inside evaluations.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .router import completion, embedding, rerank, ModelResponse, Usage


# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    def to_dict(self) -> dict:
        return {"text": self.text, "metadata": self.metadata,
                "score": round(self.score, 4)}


@dataclass
class StrategyResult:
    answer: str
    chunks: List[Chunk]
    context: str
    usage: Usage
    latency_s: float
    intermediate: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "chunks": [c.to_dict() for c in self.chunks],
            "usage": self.usage.to_dict(),
            "latency_s": round(self.latency_s, 4),
            "intermediate": self.intermediate,
        }


DEFAULT_SYSTEM = (
    "You are a precise assistant. Answer ONLY from the provided context. "
    "If the context is insufficient, say 'I don't have enough information.'"
)


def _rag_prompt(query: str, context: str) -> str:
    return f"""Context:
{context}

Question: {query}

Answer:"""


def _dedupe(chunks: List[Chunk]) -> List[Chunk]:
    seen, out = set(), []
    for c in chunks:
        if c.text[:200] not in seen:
            seen.add(c.text[:200])
            out.append(c)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Base
# ──────────────────────────────────────────────────────────────────────────────

class Strategy(ABC):
    """Base class. Subclasses implement :meth:`run`."""

    name: str = "base"
    description: str = ""

    def __init__(self, **config):
        self.config = config

    @abstractmethod
    def run(
        self,
        query: str,
        index: "VectorIndex",
        llm_model: str,
        embedding_model: str,
    ) -> StrategyResult: ...

    def __repr__(self):
        return f"<Strategy {self.name!r}>"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Naive RAG
# ──────────────────────────────────────────────────────────────────────────────

class NaiveRAG(Strategy):
    name = "naive"
    description = "Dense top-k retrieval → single LLM call"

    def run(self, query, index, llm_model, embedding_model):
        t0 = time.perf_counter()
        chunks = index.search(query, k=self.config.get("k", 5), embed_model=embedding_model)
        context = "\n\n---\n\n".join(c.text for c in chunks)
        resp = completion(
            model=llm_model, temperature=self.config.get("temperature", 0),
            messages=[{"role": "system", "content": self.config.get("system", DEFAULT_SYSTEM)},
                      {"role": "user", "content": _rag_prompt(query, context)}],
        )
        resp.usage.cost_usd += sum(u.cost_usd for u in index.last_embed_usage)
        return StrategyResult(resp.text, chunks, context, resp.usage,
                              time.perf_counter() - t0)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Hybrid RAG (dense + BM25)
# ──────────────────────────────────────────────────────────────────────────────

class HybridRAG(Strategy):
    name = "hybrid"
    description = "Dense + sparse(BM25) weighted fusion"

    def run(self, query, index, llm_model, embedding_model):
        from rank_bm25 import BM25Okapi
        import re
        t0 = time.perf_counter()
        alpha = self.config.get("alpha", 0.7)
        k = self.config.get("k", 5)

        dense = index.search_with_scores(query, k=min(k * 4, len(index)), embed_model=embedding_model)
        if not dense:
            raise RuntimeError("index empty")

        tok = lambda s: re.findall(r"\w+", s.lower())
        corpus = [tok(c.text) for c, _ in dense]
        bm25 = BM25Okapi(corpus)
        sparse_scores = bm25.get_scores(tok(query))
        smax = max(sparse_scores) or 1e-9

        fused = []
        for i, (chunk, dscore) in enumerate(dense):
            sscore = sparse_scores[i] / smax
            fused.append((chunk, alpha * dscore + (1 - alpha) * sscore))
        fused.sort(key=lambda x: x[1], reverse=True)
        chunks = [c for c, _ in fused[:k]]

        context = "\n\n---\n\n".join(c.text for c in chunks)
        resp = completion(model=llm_model, temperature=0,
                          messages=[{"role": "system", "content": DEFAULT_SYSTEM},
                                    {"role": "user", "content": _rag_prompt(query, context)}])
        resp.usage.cost_usd += sum(u.cost_usd for u in index.last_embed_usage)
        return StrategyResult(resp.text, chunks, context, resp.usage,
                              time.perf_counter() - t0,
                              {"alpha": alpha})


# ──────────────────────────────────────────────────────────────────────────────
# 3. Multi-Query RAG
# ──────────────────────────────────────────────────────────────────────────────

class MultiQueryRAG(Strategy):
    name = "multi_query"
    description = "LLM generates N query variants; union of retrievals"

    def run(self, query, index, llm_model, embedding_model):
        t0 = time.perf_counter()
        n = self.config.get("n_queries", 4)

        gen = completion(model=llm_model, messages=[
            {"role": "system", "content": "Generate search queries."},
            {"role": "user", "content":
                f"Write {n} alternative search queries for the question below. "
                f"One per line, no numbering.\n\nQuestion: {query}\n\nQueries:"}])
        variants = [v.strip("-• ").strip() for v in gen.text.splitlines() if v.strip()][:n]
        variants.insert(0, query)

        pool: List[Chunk] = []
        for v in variants:
            pool.extend(index.search(v, k=self.config.get("k", 5), embed_model=embedding_model))

        # frequency-weighted keep
        freq: Dict[str, int] = {}
        for c in pool:
            freq[c.text[:200]] = freq.get(c.text[:200], 0) + 1
        unique = sorted(_dedupe(pool), key=lambda c: freq[c.text[:200]], reverse=True)
        chunks = unique[: self.config.get("final_k", 8)]

        context = "\n\n---\n\n".join(c.text for c in chunks)
        final = completion(model=llm_model, temperature=0,
                           messages=[{"role": "system", "content": DEFAULT_SYSTEM},
                                     {"role": "user", "content": _rag_prompt(query, context)}])
        total_usage = Usage(prompt_tokens=gen.usage.prompt_tokens + final.usage.prompt_tokens,
                            completion_tokens=gen.usage.completion_tokens + final.usage.completion_tokens,
                            total_tokens=gen.usage.total_tokens + final.usage.total_tokens,
                            cost_usd=gen.usage.cost_usd + final.usage.cost_usd)
        total_usage.cost_usd += sum(u.cost_usd for u in index.last_embed_usage)
        return StrategyResult(final.text, chunks, context, total_usage,
                              time.perf_counter() - t0, {"queries": variants})


# ──────────────────────────────────────────────────────────────────────────────
# 4. HyDE
# ──────────────────────────────────────────────────────────────────────────────

class HyDERAG(Strategy):
    name = "hyde"
    description = "Retrieve using an LLM-imagined answer's embedding"

    def run(self, query, index, llm_model, embedding_model):
        t0 = time.perf_counter()
        hyp = completion(model=llm_model, messages=[
            {"role": "system", "content": "You write plausible passages."},
            {"role": "user", "content":
                f"Write one paragraph that would appear in the perfect source document "
                f"answering this question. Be specific and technical.\n\n{query}"}])

        vec = embedding(embedding_model, input=[hyp.text])
        chunks = index.search_by_vector(vec.vectors[0], k=self.config.get("k", 5))

        context = "\n\n---\n\n".join(c.text for c in chunks)
        final = completion(model=llm_model, temperature=0,
                           messages=[{"role": "system", "content": DEFAULT_SYSTEM},
                                     {"role": "user", "content": _rag_prompt(query, context)}])
        total_usage = Usage(hyp.usage.prompt_tokens + final.usage.prompt_tokens,
                            hyp.usage.completion_tokens + final.usage.completion_tokens,
                            hyp.usage.total_tokens + final.usage.total_tokens,
                            hyp.usage.cost_usd + final.usage.cost_usd)
        total_usage.cost_usd += vec.usage.cost_usd
        return StrategyResult(final.text, chunks, context, total_usage,
                              time.perf_counter() - t0)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Rerank RAG
# ──────────────────────────────────────────────────────────────────────────────

class RerankRAG(Strategy):
    name = "rerank"
    description = "Recall wide with vectors, refine with a cross-encoder"

    def run(self, query, index, llm_model, embedding_model):
        t0 = time.perf_counter()
        candidates = index.search(query, k=self.config.get("candidates_k", 20),
                                  embed_model=embedding_model)
        rr = rerank(model=self.config.get("rerank_model",
                                          "huggingface/BAAI/bge-reranker-v2-m3"),
                    query=query,
                    documents=[c.text for c in candidates],
                    top_n=self.config.get("k", 5))
        chunks = [candidates[r["index"]] for r in rr]
        for c, s in zip(chunks, [r["relevance_score"] for r in rr]):
            c.score = s

        context = "\n\n---\n\n".join(c.text for c in chunks)
        resp = completion(model=llm_model, temperature=0,
                          messages=[{"role": "system", "content": DEFAULT_SYSTEM},
                                    {"role": "user", "content": _rag_prompt(query, context)}])
        resp.usage.cost_usd += sum(u.cost_usd for u in index.last_embed_usage)
        return StrategyResult(resp.text, chunks, context, resp.usage,
                              time.perf_counter() - t0)


# ──────────────────────────────────────────────────────────────────────────────
# 6. RAG-Fusion (RRF over multi-query)
# ──────────────────────────────────────────────────────────────────────────────

class RAGFusion(Strategy):
    name = "rag_fusion"
    description = "Multi-query + Reciprocal Rank Fusion aggregation"

    def run(self, query, index, llm_model, embedding_model):
        t0 = time.perf_counter()
        n = self.config.get("n_queries", 4)
        K = self.config.get("rrf_k", 60)

        gen = completion(model=llm_model, messages=[
            {"role": "user", "content":
                f"Generate {n} diverse search queries for: {query}\nOne per line."}])
        variants = [v.strip("-• ").strip() for v in gen.text.splitlines() if v.strip()][:n]
        variants.insert(0, query)

        rrf: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}
        for v in variants:
            hits = index.search(v, k=self.config.get("k", 10), embed_model=embedding_model)
            for rank, c in enumerate(hits, start=1):
                key = c.text[:200]
                rrf[key] = rrf.get(key, 0) + 1.0 / (K + rank)
                chunk_map.setdefault(key, c)

        ranked = sorted(chunk_map.items(), key=lambda kv: kv[1], reverse=True)
        chunks = [chunk_map[k] for k, _ in ranked[: self.config.get("final_k", 6)]]

        context = "\n\n---\n\n".join(c.text for c in chunks)
        final = completion(model=llm_model, temperature=0,
                           messages=[{"role": "system", "content": DEFAULT_SYSTEM},
                                     {"role": "user", "content": _rag_prompt(query, context)}])
        total_usage = Usage(gen.usage.prompt_tokens + final.usage.prompt_tokens,
                            gen.usage.completion_tokens + final.usage.completion_tokens,
                            gen.usage.total_tokens + final.usage.total_tokens,
                            gen.usage.cost_usd + final.usage.cost_usd)
        total_usage.cost_usd += sum(u.cost_usd for u in index.last_embed_usage)
        return StrategyResult(final.text, chunks, context, total_usage,
                              time.perf_counter() - t0, {"rrf_top": dict(list(ranked.items())[:3]) if ranked else {}})


# ──────────────────────────────────────────────────────────────────────────────
# 7. Contextual Compression RAG
# ──────────────────────────────────────────────────────────────────────────────

class ContextualCompressionRAG(Strategy):
    name = "compression"
    description = "LLM extracts only relevant spans before generation"

    def run(self, query, index, llm_model, embedding_model):
        t0 = time.perf_counter()
        raw = index.search(query, k=self.config.get("k", 5), embed_model=embedding_model)

        extract_usage = Usage()
        kept: List[Chunk] = []
        for c in raw:
            ex = completion(model=llm_model, messages=[
                {"role": "user", "content":
                    f"Extract every sentence relevant to the question from the passage. "
                    f"If nothing is relevant reply exactly NONE.\n\nPassage:\n{c.text}\n\nQuestion: {query}\n\nRelevant sentences:"}])
            extract_usage.prompt_tokens += ex.usage.prompt_tokens
            extract_usage.completion_tokens += ex.usage.completion_tokens
            extract_usage.total_tokens += ex.usage.total_tokens
            extract_usage.cost_usd += ex.usage.cost_usd
            if ex.text.strip().upper() != "NONE" and ex.text.strip():
                kept.append(Chunk(text=ex.text.strip(), metadata=c.metadata, score=c.score))

        if not kept:
            kept = raw

        context = "\n\n---\n\n".join(c.text for c in kept)
        final = completion(model=llm_model, temperature=0,
                           messages=[{"role": "system", "content": DEFAULT_SYSTEM},
                                     {"role": "user", "content": _rag_prompt(query, context)}])
        total_usage = Usage(extract_usage.prompt_tokens + final.usage.prompt_tokens,
                            extract_usage.completion_tokens + final.usage.completion_tokens,
                            extract_usage.total_tokens + final.usage.total_tokens,
                            extract_usage.cost_usd + final.usage.cost_usd)
        total_usage.cost_usd += sum(u.cost_usd for u in index.last_embed_usage)
        return StrategyResult(final.text, kept, context, total_usage,
                              time.perf_counter() - t0)


# ──────────────────────────────────────────────────────────────────────────────
# 8. Corrective RAG (CRAG)
# ──────────────────────────────────────────────────────────────────────────────

class CRAGRAG(Strategy):
    name = "crag"
    description = "Grade retrieval quality → rewrite query & retry if poor"

    def run(self, query, index, llm_model, embedding_model):
        t0 = time.perf_counter()
        threshold = self.config.get("threshold", 0.5)
        grade_usage = Usage()

        chunks = index.search(query, k=self.config.get("k", 5), embed_model=embedding_model)
        grades = []
        for c in chunks:
            g = completion(model=llm_model, temperature=0, messages=[
                {"role": "user", "content":
                    f"Does this passage help answer the question? Reply only CORRECT / INCORRECT.\n\n"
                    f"Question: {query}\n\nPassage:\n{c.text[:1500]}\n\nVerdict:"}])
            grade_usage.prompt_tokens += g.usage.prompt_tokens
            grade_usage.completion_tokens += g.usage.completion_tokens
            grade_usage.cost_usd += g.usage.cost_usd
            verdict = g.text.strip().upper().startswith("CORRECT")
            grades.append(verdict)
            c.score = 1.0 if verdict else 0.0

        action = "use"
        if grades and sum(grades) / len(grades) < threshold:
            action = "correct"
            rw = completion(model=llm_model, messages=[
                {"role": "user", "content":
                    f"The initial search failed. Rewrite as a better keyword-style search query.\n\n"
                    f"Original: {query}\n\nRewritten query:"}])
            grade_usage.prompt_tokens += rw.usage.prompt_tokens
            grade_usage.completion_tokens += rw.usage.completion_tokens
            grade_usage.cost_usd += rw.usage.cost_usd
            more = index.search(rw.text.strip(), k=self.config.get("k", 5), embed_model=embedding_model)
            merged = _dedupe([c for c, ok in zip(chunks, grades) if ok] + more)
            chunks = merged[: self.config.get("k", 5)]

        good = [c for c in chunks if c.score >= 0.99] or chunks
        context = "\n\n---\n\n".join(c.text for c in good)
        final = completion(model=llm_model, temperature=0,
                           messages=[{"role": "system", "content": DEFAULT_SYSTEM},
                                     {"role": "user", "content": _rag_prompt(query, context)}])
        total_usage = Usage(grade_usage.prompt_tokens + final.usage.prompt_tokens,
                            grade_usage.completion_tokens + final.usage.completion_tokens,
                            grade_usage.total_tokens + final.usage.total_tokens,
                            grade_usage.cost_usd + final.usage.cost_usd)
        total_usage.cost_usd += sum(u.cost_usd for u in index.last_embed_usage)
        return StrategyResult(final.text, good, context, total_usage,
                              time.perf_counter() - t0, {"correction_action": action,
                                                         "grades": grades})


# ──────────────────────────────────────────────────────────────────────────────
# 9. Self-RAG style adaptive
# ──────────────────────────────────────────────────────────────────────────────

class SelfRAG(Strategy):
    name = "self_rag"
    description = "Model decides whether retrieval is needed at all"

    def run(self, query, index, llm_model, embedding_model):
        t0 = time.perf_counter()
        decision = completion(model=llm_model, temperature=0, messages=[
            {"role": "user", "content":
                f"Should external knowledge be retrieved for this question? "
                f"Reply RETRIEVE or NO_RETRIEVE.\n\nQuestion: {query}"}])

        retrieve = "NO_RETRIEVE" not in decision.text.upper()
        chunks: List[Chunk] = []
        extra_usage = Usage(prompt_tokens=decision.usage.prompt_tokens,
                            completion_tokens=decision.usage.completion_tokens,
                            cost_usd=decision.usage.cost_usd)

        if retrieve:
            chunks = index.search(query, k=self.config.get("k", 5), embed_model=embedding_model)
            extra_usage.total_tokens += decision.usage.total_tokens

        context = "\n\n---\n\n".join(c.text for c in chunks) if chunks \
            else "(no retrieved context)"
        resp = completion(model=llm_model, temperature=0,
                          messages=[{"role": "system", "content": DEFAULT_SYSTEM},
                                    {"role": "user", "content":
                                        f"Context (may be empty):\n{context}\n\nQuestion: {query}\n\nAnswer:"}])
        extra_usage.prompt_tokens += resp.usage.prompt_tokens
        extra_usage.completion_tokens += resp.usage.completion_tokens
        extra_usage.total_tokens += resp.usage.total_tokens
        extra_usage.cost_usd += resp.usage.cost_usd
        extra_usage.cost_usd += sum(u.cost_usd for u in index.last_embed_usage)
        return StrategyResult(resp.text, chunks, context, extra_usage,
                              time.perf_counter() - t0, {"retrieved": retrieve})


# ──────────────────────────────────────────────────────────────────────────────
# 10. Query Decomposition
# ──────────────────────────────────────────────────────────────────────────────

class QueryDecompositionRAG(Strategy):
    name = "decomposition"
    description = "Split complex question into sub-questions, merge answers"

    def run(self, query, index, llm_model, embedding_model):
        t0 = time.perf_counter()
        dec = completion(model=llm_model, messages=[
            {"role": "user", "content":
                f"Break this into 2-4 independent sub-questions, one per line:\n\n{query}"}])
        subs = [s.strip("-• ").strip() for s in dec.text.splitlines() if s.strip()]

        sub_answers = []
        agg_usage = Usage(dec.usage.prompt_tokens, dec.usage.completion_tokens,
                          dec.usage.total_tokens, dec.usage.cost_usd)
        all_chunks: List[Chunk] = []

        for sub in subs:
            cs = index.search(sub, k=self.config.get("k", 3), embed_model=embedding_model)
            all_chunks.extend(cs)
            ctx = "\n\n".join(c.text for c in cs)
            a = completion(model=llm_model, temperature=0, messages=[
                {"role": "user", "content": _rag_prompt(sub, ctx)}])
            sub_answers.append(f"Q: {sub}\nA: {a.text}")
            agg_usage.prompt_tokens += a.usage.prompt_tokens
            agg_usage.completion_tokens += a.usage.completion_tokens
            agg_usage.cost_usd += a.usage.cost_usd

        synth_ctx = "\n\n".join(sub_answers)
        final = completion(model=llm_model, temperature=0, messages=[
            {"role": "user", "content":
                f"Original question: {query}\n\nSub-question findings:\n{synth_ctx}\n\n"
                f"Synthesize one complete answer:"}])
        agg_usage.prompt_tokens += final.usage.prompt_tokens
        agg_usage.completion_tokens += final.usage.completion_tokens
        agg_usage.cost_usd += final.usage.cost_usd
        agg_usage.cost_usd += sum(u.cost_usd for u in index.last_embed_usage)
        return StrategyResult(final.text, _dedupe(all_chunks)[:12],
                              "\n\n".join(c.text for c in _dedupe(all_chunks)),
                              agg_usage, time.perf_counter() - t0, {"sub_questions": subs})


# ──────────────────────────────────────────────────────────────────────────────
# 11. Step-back Prompting
# ──────────────────────────────────────────────────────────────────────────────

class StepBackRAG(Strategy):
    name = "step_back"
    description = "Ask a broader principle question first, then answer specifics"

    def run(self, query, index, llm_model, embedding_model):
        t0 = time.perf_counter()
        sb = completion(model=llm_model, messages=[
            {"role": "user", "content":
                f"Rewrite this specific question as a general/conceptual one about underlying principles.\n\n"
                f"Specific: {query}\n\nGeneral:"}])

        spec_chunks = index.search(query, k=self.config.get("k", 3), embed_model=embedding_model)
        broad_chunks = index.search(sb.text.strip(), k=self.config.get("k", 3),
                                    embed_model=embedding_model)
        chunks = _dedupe(spec_chunks + broad_chunks)[: self.config.get("final_k", 6)]

        context = "\n\n---\n\n".join(c.text for c in chunks)
        resp = completion(model=llm_model, temperature=0, messages=[
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content":
                f"General principle: {sb.text.strip()}\n\nContext:\n{context}\n\nSpecific question: {query}\n\nAnswer:"}])
        total_usage = Usage(sb.usage.prompt_tokens + resp.usage.prompt_tokens,
                            sb.usage.completion_tokens + resp.usage.completion_tokens,
                            sb.usage.total_tokens + resp.usage.total_tokens,
                            sb.usage.cost_usd + resp.usage.cost_usd)
        total_usage.cost_usd += sum(u.cost_usd for u in index.last_embed_usage)
        return StrategyResult(resp.text, chunks, context, total_usage,
                              time.perf_counter() - t0, {"step_back_question": sb.text.strip()})


# ──────────────────────────────────────────────────────────────────────────────
# 12. Agentic RAG (iterative)
# ──────────────────────────────────────────────────────────────────────────────

class AgenticRAG(Strategy):
    name = "agentic"
    description = "Agent loops: search → reflect → search again until satisfied"

    def run(self, query, index, llm_model, embedding_model):
        t0 = time.perf_counter()
        max_iters = self.config.get("max_iterations", 3)
        collected: List[Chunk] = []
        trace: List[dict] = []
        usage_acc = Usage()
        answer_text = ""

        current_search = query
        for it in range(max_iters):
            hits = index.search(current_search, k=self.config.get("k", 3),
                                embed_model=embedding_model)
            collected.extend(hits)
            ctx = "\n\n---\n\n".join(c.text for c in _dedupe(collected))

            step = completion(model=llm_model, temperature=0, messages=[
                {"role": "system", "content":
                    "You are a research agent. If the context fully answers the question, "
                    "reply FINAL_ANSWER followed by your answer. Otherwise reply "
                    "SEARCH: <new focused search query>."},
                {"role": "user", "content":
                    f"Original question: {query}\n\nAccumulated context:\n{ctx[:6000]}"}])

            usage_acc.prompt_tokens += step.usage.prompt_tokens
            usage_acc.completion_tokens += step.usage.completion_tokens
            usage_acc.total_tokens += step.usage.total_tokens
            usage_acc.cost_usd += step.usage.cost_usd
            trace.append({"iteration": it + 1, "search": current_search,
                          "action": step.text[:120]})

            if "FINAL_ANSWER" in step.text.upper():
                answer_text = step.text.split("FINAL_ANSWER", 1)[1].strip().lstrip(":").strip()
                break
            if "SEARCH:" in step.text.upper():
                current_search = step.text.split(":", 1)[1].strip()
            else:
                answer_text = step.text
                break
        else:
            ctx = "\n\n---\n\n".join(c.text for c in _dedupe(collected))
            fin = completion(model=llm_model, temperature=0, messages=[
                {"role": "user", "content": _rag_prompt(query, ctx)}])
            answer_text = fin.text
            usage_acc.prompt_tokens += fin.usage.prompt_tokens
            usage_acc.completion_tokens += fin.usage.completion_tokens
            usage_acc.cost_usd += fin.usage.cost_usd

        usage_acc.cost_usd += sum(u.cost_usd for u in index.last_embed_usage)
        final_chunks = _dedupe(collected)[: self.config.get("final_k", 8)]
        return StrategyResult(answer_text, final_chunks,
                              "\n\n---\n\n".join(c.text for c in final_chunks),
                              usage_acc, time.perf_counter() - t0, {"agent_trace": trace})


# ──────────────────────────────────────────────────────────────────────────────
# 13. FLARE (simplified active retrieval between sentences)
# ──────────────────────────────────────────────────────────────────────────────

class FLARERAG(Strategy):
    name = "flare"
    description = "Generate draft → low-confidence sentences trigger re-retrieval"

    def run(self, query, index, llm_model, embedding_model):
        t0 = time.perf_counter()
        draft_ctx = "\n\n".join(
            c.text for c in index.search(query, k=self.config.get("initial_k", 3),
                                         embed_model=embedding_model))
        draft = completion(model=llm_model, temperature=0, messages=[
            {"role": "user", "content": _rag_prompt(query, draft_ctx)}])

        # confidence proxy: ask model to flag uncertain claims
        audit = completion(model=llm_model, temperature=0, messages=[
            {"role": "user", "content":
                f"List any claims in this draft needing better sources. One per line. "
                f"If none, reply SOLID.\n\nDraft:\n{draft.text}"}])

        extra_chunks: List[Chunk] = []
        if "SOLID" not in audit.text.upper():
            weak = [l.strip("-• ").strip() for l in audit.text.splitlines() if l.strip()]
            for w in weak[:2]:
                extra_chunks.extend(index.search(w, k=2, embed_model=embedding_model))

        chunks = _dedupe(
            index.search(query, k=self.config.get("initial_k", 3), embed_model=embedding_model)
            + extra_chunks)[: self.config.get("final_k", 6)]

        context = "\n\n---\n\n".join(c.text for c in chunks)
        final = completion(model=llm_model, temperature=0, messages=[
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content":
                f"Improve this draft using ALL context.\n\nDraft:\n{draft.text}\n\nContext:\n{context}\n\nFinal answer:"}])

        total_usage = Usage(draft.usage.prompt_tokens + audit.usage.prompt_tokens + final.usage.prompt_tokens,
                            draft.usage.completion_tokens + audit.usage.completion_tokens + final.usage.completion_tokens,
                            0, draft.usage.cost_usd + audit.usage.cost_usd + final.usage.cost_usd)
        total_usage.total_tokens = total_usage.prompt_tokens + total_usage.completion_tokens
        total_usage.cost_usd += sum(u.cost_usd for u in index.last_embed_usage)
        return StrategyResult(final.text, chunks, context, total_usage,
                              time.perf_counter() - t0,
                              {"draft_uncertainty_flags": audit.text[:400]})


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

STRATEGIES: Dict[str, type] = {
    cls.name: cls for cls in [
        NaiveRAG, HybridRAG, MultiQueryRAG, HyDERAG, RerankRAG, RAGFusion,
        ContextualCompressionRAG, CRAGRAG, SelfRAG, QueryDecompositionRAG,
        StepBackRAG, AgenticRAG, FLARERAG,
    ]
}


def get_strategy(name: str, **config) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Available: {sorted(STRATEGIES)}")
    return STRATEGIES[name](**config)
