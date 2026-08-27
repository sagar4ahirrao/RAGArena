"""
Evaluation metrics — retrieval, generation, and operational dimensions.

Metric families:
  * Retrieval:   context_precision, context_recall, hit_rate, mrr
  * Generation:  faithfulness, answer_relevance, answer_correctness
  * Operational: latency, cost, token_usage

LLM-judged metrics run through the same unified router as everything else,
so ANY provider/model can act as judge.
"""
from __future__ import annotations

import re
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MetricResult:
    name: str
    score: float                      # normalized 0-1 (higher = better) except raw metrics
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        precision = 8 if self.name == "cost_usd" else 4
        return {"name": self.name,
                "score": round(self.score, precision),
                "details": {k: (v if not isinstance(v, float) else round(v, 6))
                            for k, v in self.details.items()}}


class BaseMetric(ABC):
    """A metric receives the full evaluation sample and returns a score."""

    name: str = "base"
    higher_is_better: bool = True
    requires_reference: bool = False       # needs ground-truth answer?
    requires_llm_judge: bool = False

    @abstractmethod
    def compute(self, sample: "EvalSample", ctx: "MetricContext") -> MetricResult: ...


@dataclass
class EvalSample:
    question: str
    reference_answer: Optional[str]
    generated_answer: str
    retrieved_chunks: List[dict]           # [{text, metadata, score}]
    context: str
    usage: Dict[str, Any]                  # token usage dict from strategy
    latency_s: float
    intermediate: Dict[str, Any]


@dataclass
class MetricContext:
    """Shared services a metric may use."""
    judge_model: str                       # 'provider/model' for LLM-as-judge
    temperature: float = 0.0
    embedding_model: Optional[str] = None  # when set, retrieval metrics use real
                                            # embedding cosine similarity instead of
                                            # the lexical keyword-overlap fallback
    judge_samples: int = 1                 # >1: average N independent judge calls
                                            # to reduce single-sample LLM-judge variance
    _cache: Dict[str, Any] = field(default_factory=dict)  # per-sample memo, avoids
                                            # re-embedding the same query+chunks once
                                            # per retrieval metric (precision/hit/mrr)


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval metrics
# ──────────────────────────────────────────────────────────────────────────────

def _keyword_overlap(text_a: str, text_b: str) -> float:
    stop = set("the a an is are was were be to of in on for with as at by from "
               "and or not it its this that these those what which who whom whose "
               "how why when where does do did can could will would".split())
    wa = {w for w in re.findall(r"\w+", text_a.lower()) if w not in stop and len(w) > 2}
    wb = {w for w in re.findall(r"\w+", text_b.lower()) if w not in stop and len(w) > 2}
    if not wa:
        return 0.0
    return len(wa & wb) / len(wa)


# Cosine-similarity threshold above which a chunk counts as "relevant" for the
# binary precision/hit-rate/MRR decisions. Embeddings vary by model/provider,
# so this is a reasonable default for modern general-purpose embedding models,
# not a universal constant — tune via RAGARENA_RELEVANCE_THRESHOLD if needed.
import os as _os
_SEMANTIC_THRESHOLD = float(_os.environ.get("RAGARENA_RELEVANCE_THRESHOLD", "0.55"))
_LEXICAL_THRESHOLD = 0.15


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _relevance_scores(query: str, texts: List[str], ctx: MetricContext) -> tuple[List[float], float, str]:
    """Return (per-text relevance score, binary-relevance threshold, method used).

    Uses real embedding cosine similarity when ``ctx.embedding_model`` is set
    (the normal path — engine.py always sets it to the same model used for
    retrieval), and falls back to the lexical keyword-overlap heuristic
    otherwise or if the embedding call itself fails (e.g. transient API error)
    so a metrics computation never crashes an entire evaluation run.
    """
    if not texts:
        return [], _SEMANTIC_THRESHOLD, "none"
    cache_key = f"rel::{query}::{'|'.join(texts)[:500]}"
    cached = ctx._cache.get(cache_key)
    if cached is not None:
        return cached
    if ctx.embedding_model:
        try:
            from .router import embedding as _embed
            resp = _embed(ctx.embedding_model, input=[query] + texts)
            qvec, tvecs = resp.vectors[0], resp.vectors[1:]
            result = ([max(0.0, _cosine(qvec, tv)) for tv in tvecs], _SEMANTIC_THRESHOLD, "embedding")
            ctx._cache[cache_key] = result
            return result
        except Exception:
            pass  # fall through to lexical
    result = ([_keyword_overlap(query, t) for t in texts], _LEXICAL_THRESHOLD, "keyword")
    ctx._cache[cache_key] = result
    return result


class ContextPrecision(BaseMetric):
    """Fraction of retrieved chunks actually relevant to the question."""
    name = "context_precision"

    def compute(self, s: EvalSample, ctx: MetricContext) -> MetricResult:
        texts = [c["text"] for c in s.retrieved_chunks]
        scores, threshold, method = _relevance_scores(s.question, texts, ctx)
        relevant = [1.0 if sc >= threshold else 0.0 for sc in scores]
        prec_at_k = []
        hits = 0
        for i, rel in enumerate(relevant, start=1):
            hits += rel
            prec_at_k.append(hits / i)
        score = sum(p * r for p, r in zip(prec_at_k, relevant)) / max(sum(relevant), 1e-9)
        return MetricResult(self.name, score,
                            {"relevant_chunks": int(sum(relevant)),
                             "total_chunks": len(relevant), "method": method})


class ContextRecall(BaseMetric):
    """How much of the ground-truth answer's content is covered by retrieved context."""
    name = "context_recall"
    requires_reference = True

    def compute(self, s: EvalSample, ctx: MetricContext) -> MetricResult:
        if not s.reference_answer:
            return MetricResult(self.name, 0.0, {"error": "reference required"})
        scores, _, method = _relevance_scores(s.reference_answer, [s.context], ctx)
        covered = scores[0] if scores else 0.0
        return MetricResult(self.name, covered, {"overlap": covered, "method": method})


class HitRate(BaseMetric):
    """Did any retrieved chunk contain query-relevant signal?"""
    name = "hit_rate"

    def compute(self, s: EvalSample, ctx: MetricContext) -> MetricResult:
        texts = [c["text"] for c in s.retrieved_chunks]
        scores, threshold, method = _relevance_scores(s.question, texts, ctx)
        best_i, best_v = max(enumerate(scores), key=lambda x: x[1], default=(-1, 0.0))
        hit = 1.0 if best_v >= threshold else 0.0
        return MetricResult(self.name, hit,
                            {"best_chunk_index": best_i, "best_score": round(best_v, 4), "method": method})


class MRR(BaseMetric):
    """Mean reciprocal rank of first relevant chunk."""
    name = "mrr"

    def compute(self, s: EvalSample, ctx: MetricContext) -> MetricResult:
        texts = [c["text"] for c in s.retrieved_chunks]
        scores, threshold, method = _relevance_scores(s.question, texts, ctx)
        for i, sc in enumerate(scores, start=1):
            if sc >= threshold:
                return MetricResult(self.name, 1.0 / i, {"first_relevant_rank": i, "method": method})
        return MetricResult(self.name, 0.0, {"first_relevant_rank": None, "method": method})


# ──────────────────────────────────────────────────────────────────────────────
# LLM-judged generation metrics
# ──────────────────────────────────────────────────────────────────────────────

_JUDGE_INSTRUCTIONS = {
    "faithfulness": (
        "Faithfulness measures whether every claim in the ANSWER is supported by the CONTEXT.",
        "Score 1.0 if all claims are supported by context; 0.0 if none are. "
        "Partial support → proportional score between 0 and 1."),
    "answer_relevance": (
        "Answer relevance measures how well the ANSWER addresses the QUESTION.",
        "Score 1.0 if fully and directly answers; 0.0 if irrelevant or evasive."),
    "answer_correctness": (
        "Correctness compares the GENERATED answer to the REFERENCE answer.",
        "Score 1.0 if factually equivalent to reference; 0.0 if contradicts or is empty."),
}


def _llm_judge_score_once(instruction: str, rubric: str, prompt: str,
                          ctx: MetricContext, temperature: float) -> tuple[float, dict]:
    import json

    from .router import completion
    resp = completion(model=ctx.judge_model, temperature=temperature, messages=[
        {"role": "system", "content":
            f"You are a strict evaluation judge. {instruction}\n{rubric}\n"
            f'Respond ONLY with JSON: {{"score": <float 0-1>, "reasoning": "<one sentence>"}}'},
        {"role": "user", "content": prompt}])
    raw = resp.text.strip()
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0)) if match else json.loads(raw)
        score = max(0.0, min(1.0, float(data.get("score", 0))))
        return score, {"reasoning": data.get("reasoning", ""), "judge_cost_usd": resp.usage.cost_usd}
    except Exception:
        # fallback: first float found
        nums = re.findall(r"0?\.\d+|1\.0+|0|1", raw)
        score = float(nums[0]) if nums else 0.0
        return max(0.0, min(1.0, score)), {"parse_fallback": raw[:200],
                                           "judge_cost_usd": resp.usage.cost_usd}


def _llm_judge_score(instruction: str, rubric: str, prompt: str,
                     ctx: MetricContext) -> tuple[float, dict]:
    """Score once by default. When ``ctx.judge_samples > 1``, calls the judge
    multiple times (temperature nudged up slightly on repeats so they aren't
    identical calls) and averages — a cheap way to reduce the real run-to-run
    variance a single LLM-judge call has, at the cost of judge_samples-1 extra
    LLM calls. Opt in via ``evaluate(..., judge_samples=3)`` when you need a
    more reliable score, e.g. for a benchmark you'll publish or compare on.
    """
    n = max(1, ctx.judge_samples)
    if n == 1:
        return _llm_judge_score_once(instruction, rubric, prompt, ctx, ctx.temperature)

    scores: List[float] = []
    total_cost = 0.0
    reasonings: List[str] = []
    for i in range(n):
        temp = ctx.temperature if i == 0 else min(1.0, ctx.temperature + 0.3)
        sc, details = _llm_judge_score_once(instruction, rubric, prompt, ctx, temp)
        scores.append(sc)
        total_cost += details.get("judge_cost_usd", 0.0)
        if details.get("reasoning"):
            reasonings.append(details["reasoning"])
    mean_score = statistics.fmean(scores)
    stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    return mean_score, {
        "reasoning": reasonings[0] if reasonings else "",
        "judge_cost_usd": total_cost,
        "judge_samples": n,
        "score_stdev": round(stdev, 4),
        "all_scores": scores,
    }


class Faithfulness(BaseMetric):
    name = "faithfulness"
    requires_llm_judge = True

    def compute(self, s: EvalSample, ctx: MetricContext) -> MetricResult:
        inst, rubric = _JUDGE_INSTRUCTIONS["faithfulness"]
        prompt = (f"CONTEXT:\n{s.context[:8000]}\n\nANSWER:\n{s.generated_answer}")
        score, details = _llm_judge_score(inst, rubric, prompt, ctx)
        return MetricResult(self.name, score, details)


class AnswerRelevance(BaseMetric):
    name = "answer_relevance"
    requires_llm_judge = True

    def compute(self, s: EvalSample, ctx: MetricContext) -> MetricResult:
        inst, rubric = _JUDGE_INSTRUCTIONS["answer_relevance"]
        prompt = f"QUESTION:\n{s.question}\n\nANSWER:\n{s.generated_answer}"
        score, details = _llm_judge_score(inst, rubric, prompt, ctx)
        return MetricResult(self.name, score, details)


class AnswerCorrectness(BaseMetric):
    name = "answer_correctness"
    requires_llm_judge = True
    requires_reference = True

    def compute(self, s: EvalSample, ctx: MetricContext) -> MetricResult:
        if not s.reference_answer:
            return MetricResult(self.name, 0.0, {"error": "reference required"})
        inst, rubric = _JUDGE_INSTRUCTIONS["answer_correctness"]
        prompt = (f"QUESTION:\n{s.question}\n\nREFERENCE:\n{s.reference_answer}\n\n"
                  f"GENERATED:\n{s.generated_answer}")
        score, details = _llm_judge_score(inst, rubric, prompt, ctx)
        return MetricResult(self.name, score, details)


# ──────────────────────────────────────────────────────────────────────────────
# Operational metrics (raw values, no normalization needed by user)
# ──────────────────────────────────────────────────────────────────────────────

class Latency(BaseMetric):
    name = "latency_s"
    higher_is_better = False

    def compute(self, s: EvalSample, ctx: MetricContext) -> MetricResult:
        return MetricResult(self.name, s.latency_s, {"unit": "seconds"})


class Cost(BaseMetric):
    name = "cost_usd"
    higher_is_better = False

    def compute(self, s: EvalSample, ctx: MetricContext) -> MetricResult:
        cost = float(s.usage.get("cost_usd", 0.0))
        return MetricResult(self.name, cost, {"unit": "usd"})


class TokenUsage(BaseMetric):
    name = "total_tokens"
    higher_is_better = False

    def compute(self, s: EvalSample, ctx: MetricContext) -> MetricResult:
        u = s.usage
        return MetricResult(
            self.name, float(u.get("total_tokens", 0)),
            {"prompt_tokens": u.get("prompt_tokens", 0),
             "completion_tokens": u.get("completion_tokens", 0),
             "context_chars": len(s.context)})


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

METRICS: Dict[str, type] = {cls.name: cls for cls in [
    ContextPrecision, ContextRecall, HitRate, MRR,
    Faithfulness, AnswerRelevance, AnswerCorrectness,
    Latency, Cost, TokenUsage,
]}


DEFAULT_METRIC_SETS = {
    "quick":      ["hit_rate", "latency_s"],
    "quality":    ["context_precision", "context_recall", "hit_rate", "mrr",
                   "faithfulness", "answer_relevance"],
    "full":       ["context_precision", "context_recall", "hit_rate", "mrr",
                   "faithfulness", "answer_relevance", "answer_correctness",
                   "latency_s", "cost_usd", "total_tokens"],
    "production": ["faithfulness", "answer_relevance", "latency_s", "cost_usd",
                   "total_tokens"],
}


def resolve_metrics(names: List[str] | str) -> List[BaseMetric]:
    if isinstance(names, str):
        names = DEFAULT_METRIC_SETS.get(names, [names])
    out = []
    for n in names:
        if n not in METRICS:
            raise ValueError(f"Unknown metric '{n}'. Available: {sorted(METRICS)} "
                             f"(or preset sets: {sorted(DEFAULT_METRIC_SETS)})")
        out.append(METRICS[n]())
    return out
