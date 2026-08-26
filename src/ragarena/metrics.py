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


class ContextPrecision(BaseMetric):
    """Fraction of retrieved chunks actually relevant to the question."""
    name = "context_precision"

    def compute(self, s: EvalSample, ctx: MetricContext) -> MetricResult:
        scores = [_keyword_overlap(s.question, c["text"]) for c in s.retrieved_chunks]
        relevant = [1.0 if sc >= 0.15 else 0.0 for sc in scores]
        prec_at_k = []
        hits = 0
        for i, rel in enumerate(relevant, start=1):
            hits += rel
            prec_at_k.append(hits / i)
        score = sum(p * r for p, r in zip(prec_at_k, relevant)) / max(sum(relevant), 1e-9)
        return MetricResult(self.name, score,
                            {"relevant_chunks": int(sum(relevant)),
                             "total_chunks": len(relevant)})


class ContextRecall(BaseMetric):
    """How much of the ground-truth answer's content is covered by retrieved context."""
    name = "context_recall"
    requires_reference = True

    def compute(self, s: EvalSample, ctx: MetricContext) -> MetricResult:
        if not s.reference_answer:
            return MetricResult(self.name, 0.0, {"error": "reference required"})
        covered = _keyword_overlap(s.reference_answer, s.context)
        return MetricResult(self.name, covered, {"overlap": covered})


class HitRate(BaseMetric):
    """Did any retrieved chunk contain query-relevant signal?"""
    name = "hit_rate"

    def compute(self, s: EvalSample, ctx: MetricContext) -> MetricResult:
        best = max(((_keyword_overlap(s.question, c["text"]), i)
                    for i, c in enumerate(s.retrieved_chunks)), default=(0.0, -1))
        hit = 1.0 if best[0] >= 0.15 else 0.0
        return MetricResult(self.name, hit,
                            {"best_chunk_index": best[1], "best_overlap": best[0]})


class MRR(BaseMetric):
    """Mean reciprocal rank of first relevant chunk."""
    name = "mrr"

    def compute(self, s: EvalSample, ctx: MetricContext) -> MetricResult:
        for i, c in enumerate(s.retrieved_chunks, start=1):
            if _keyword_overlap(s.question, c["text"]) >= 0.15:
                return MetricResult(self.name, 1.0 / i, {"first_relevant_rank": i})
        return MetricResult(self.name, 0.0, {"first_relevant_rank": None})


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


def _llm_judge_score(instruction: str, rubric: str, prompt: str,
                     ctx: MetricContext) -> tuple[float, dict]:
    import json

    from .router import completion
    resp = completion(model=ctx.judge_model, temperature=ctx.temperature, messages=[
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
