"""
Evaluation engine — `evaluate()` and `compare()` one-liners.

Ergonomic one-liners::

    from ragarena import evaluate

    result = evaluate(
        questions=["What is RAG?"],
        documents=[{"text": "RAG = retrieval-augmented generation ..."}],
        strategy="hybrid",
        model="openai/gpt-4o-mini",                    # generator
        embedding_model="openai/text-embedding-3-small",
        judge_model="openai/gpt-4o-mini",              # LLM-as-judge
        metrics="quality",                             # preset or list
    )
    print(result.summary())
"""
from __future__ import annotations

import json
import sys
import time
import uuid
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from .catalog import get_model, UnknownModelError
from .index import VectorIndex, TextChunker
from .metrics import EvalSample, MetricContext, MetricResult, resolve_metrics, METRICS
from .router import completion, Usage, MissingAPIKeyError
from .strategies import Chunk, StrategyResult, get_strategy, STRATEGIES


def _print(*args, **kwargs) -> None:
    """print() that never crashes on non-UTF-8 consoles (e.g. Windows cp1252)."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        safe = [a.encode(enc, errors="replace").decode(enc) if isinstance(a, str) else a for a in args]
        print(*safe, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SampleResult:
    question: str
    reference_answer: Optional[str]
    answer: str
    chunks: List[dict]
    metrics: Dict[str, dict]           # name -> MetricResult.to_dict()
    latency_s: float
    usage: Dict[str, Any]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class EvaluationReport:
    run_id: str
    strategy: str
    model: str
    embedding_model: str
    judge_model: str
    samples: List[SampleResult] = field(default_factory=list)
    aggregate: Dict[str, float] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    wall_time_s: float = 0.0
    created_at: str = ""

    def summary(self) -> Dict[str, float]:
        """Mean score per metric across all samples."""
        agg: Dict[str, List[float]] = {}
        for s in self.samples:
            for mname, m in s.metrics.items():
                if mname in ("latency_s", "cost_usd", "total_tokens"):
                    continue          # operational metrics summarized separately
                agg.setdefault(mname, []).append(m["score"])
        out = {k: round(statistics.fmean(v), 4) for k, v in agg.items()}
        lat = [s.latency_s for s in self.samples if s.latency_s is not None]
        if lat:
            out["p50_latency_s"] = round(statistics.median(lat), 3)
            out["avg_latency_s"] = round(statistics.fmean(lat), 3)
            out["p95_latency_s"] = round(sorted(lat)[int(len(lat) * 0.95) - 1 if len(lat) > 1 else 0], 3)
        out["total_cost_usd"] = round(self.total_cost_usd, 6)
        out["total_tokens"] = self.total_tokens
        return out

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id, "strategy": self.strategy,
            "model": self.model, "embedding_model": self.embedding_model,
            "judge_model": self.judge_model,
            "aggregate": self.summary(),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_tokens": self.total_tokens,
            "wall_time_s": round(self.wall_time_s, 2),
            "created_at": self.created_at,
            "samples": [s.to_dict() for s in self.samples],
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self) -> None:
        r = self.summary()
        _print(f"\n╭─ RagArena · {self.strategy} · {self.model}")
        _print(f"├─ embedding : {self.embedding_model}")
        _print(f"├─ samples   : {len(self.samples)}   wall time {self.wall_time_s:.1f}s")
        _print("├" + "─" * 58)
        for k, v in r.items():
            label = k.replace("_", " ").rjust(18)
            val = f"${v:.6f}" if k == "total_cost_usd" else f"{v}"
            _print(f"│  {label} : {val}")
        _print("╰" + "─" * 58)


# ──────────────────────────────────────────────────────────────────────────────
# Core single-run evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    questions: List[str],
    documents: Optional[List[Dict[str, Any]]] = None,
    index: Optional[VectorIndex] = None,
    reference_answers: Optional[List[Optional[str]]] = None,
    strategy: str = "naive",
    strategy_config: Optional[Dict[str, Any]] = None,
    model: str = "openai/gpt-4o-mini",
    embedding_model: str = "openai/text-embedding-3-small",
    judge_model: str = "openai/gpt-4o-mini",
    metrics: Union[str, List[str]] = "production",
    max_concurrency: int = 1,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> EvaluationReport:
    """
    Run a full RAG evaluation over a set of questions.

    Args:
        questions: queries to evaluate.
        documents: raw docs ``{"text": ..., "metadata": {...}}`` (chunked+embedded automatically).
        index: prebuilt VectorIndex (mutually exclusive with documents).
        reference_answers: ground truth per question (enables recall/correctness).
        strategy: any of 13 strategies (see RagArena.list_strategies()).
        model: generator LLM as 'provider/name'.
        embedding_model: retriever embeddings as 'provider/name'.
        judge_model: LLM-as-judge for faithfulness/relevance metrics.
        metrics: preset name ('quick'|'quality'|'full'|'production') or explicit list.

    Returns:
        EvaluationReport with per-sample results and aggregates.
    """
    if index is None:
        if not documents:
            raise ValueError("Provide either `documents` or a prebuilt `index`.")
        chunker = (TextChunker(chunk_size or 1000, chunk_overlap or 150)
                   if (chunk_size or chunk_overlap) else None)
        index = VectorIndex(embedding_model=embedding_model, chunker=chunker)
        index.add_documents(documents)

    strat = get_strategy(strategy, **(strategy_config or {}))
    metric_impls = resolve_metrics(metrics)
    needs_ref = any(m.requires_reference for m in metric_impls)

    refs = reference_answers or [None] * len(questions)
    report = EvaluationReport(
        run_id=uuid.uuid4().hex[:12],
        strategy=strategy, model=model,
        embedding_model=index.embedding_model, judge_model=judge_model,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    t_wall = time.perf_counter()

    for q, ref in zip(questions, refs):
        try:
            sr: StrategyResult = strat.run(q, index, model, index.embedding_model)
        except (MissingAPIKeyError, UnknownModelError):
            raise   # unrecoverable config error — fail fast instead of repeating per question
        except Exception as e:
            sample = SampleResult(
                question=q, reference_answer=ref, answer=f"<ERROR> {e}",
                chunks=[], metrics={}, latency_s=0.0, usage={})
            report.samples.append(sample)
            continue

        sample_data = EvalSample(
            question=q, reference_answer=ref, generated_answer=sr.answer,
            retrieved_chunks=[c.to_dict() for c in sr.chunks],
            context=sr.context,
            usage=sr.usage.to_dict(), latency_s=sr.latency_s,
            intermediate=sr.intermediate,
        )
        mctx = MetricContext(judge_model=judge_model)

        metric_out: Dict[str, dict] = {}
        for impl in metric_impls:
            if impl.requires_reference and not ref:
                continue
            try:
                res: MetricResult = impl.compute(sample_data, mctx)
            except Exception as e:
                res = MetricResult(impl.name, 0.0, {"error": str(e)})
            metric_out[impl.name] = res.to_dict()

        usage_d = sr.usage.to_dict()
        report.total_cost_usd += usage_d.get("cost_usd", 0.0) + sum(
            v.get("details", {}).get("judge_cost_usd", 0.0)
            for v in metric_out.values() if isinstance(v, dict))
        report.total_tokens += usage_d.get("total_tokens", 0)
        report.samples.append(SampleResult(
            question=q, reference_answer=ref, answer=sr.answer,
            chunks=[c.to_dict() for c in sr.chunks],
            metrics=metric_out, latency_s=sr.latency_s, usage=usage_d))

    report.wall_time_s = time.perf_counter() - t_wall
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Strategy / model comparison
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ComparisonResult:
    runs: Dict[str, EvaluationReport] = field(default_factory=dict)
    matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)   # config name -> failure reason

    def best(self, metric: str = "faithfulness") -> Optional[str]:
        scores = {name: vals.get(metric) for name, vals in self.matrix.items()}
        scored = {k: v for k, v in scores.items() if v is not None}
        return max(scored, key=scored.get) if scored else None     # type: ignore[arg-type]

    def leaderboard(self, sort_by: str = "faithfulness") -> List[dict]:
        rows = []
        for name, vals in self.matrix.items():
            row = {"config": name, **{k: round(v, 4) if isinstance(v, float) else v
                                      for k, v in vals.items()}}
            rows.append(row)
        rows.sort(key=lambda r: -r.get(sort_by, 0))
        return rows

    def print_leaderboard(self, sort_by: str = "faithfulness") -> None:
        rows = self.leaderboard(sort_by)
        if not rows:
            _print("(no results)")
            return
        cols = list(rows[0].keys())
        widths = [max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols]
        _print("\n|" + "|".join(c.ljust(w) for c, w in zip(cols, widths)) + "|")
        _print("|" + "|".join("-" * w for w in widths) + "|")
        for r in rows:
            _print("|" + "|".join(str(r.get(c, "")).ljust(w) for c, w in zip(cols, widths)) + "|")

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "matrix": self.matrix,
                "runs": {k: v.to_dict() for k, v in self.runs.items()},
                "errors": self.errors,
            }, f, indent=2)


def compare(
    questions: List[str],
    documents: List[Dict[str, Any]],
    configs: List[Dict[str, Any]],
    reference_answers: Optional[List[Optional[str]]] = None,
    metrics: Union[str, List[str]] = "quality",
    embedding_model: str = "openai/text-embedding-3-small",
) -> ComparisonResult:
    """
    Head-to-head benchmarking across strategies and/or models.

    Each config is a dict::

        compare(questions, docs, configs=[
            {"strategy": "naive",  "model": "openai/gpt-4o-mini"},
            {"strategy": "hybrid", "model": "openai/gpt-4o-mini"},
            {"strategy": "hybrid", "model": "anthropic/claude-3-haiku-20240307"},
        ])

    Returns ComparisonResult with a full metric matrix + leaderboard.
    """
    # Build the index ONCE and share it (embeddings are expensive!)
    shared_index = VectorIndex(embedding_model=embedding_model)
    shared_index.add_documents(documents)

    result = ComparisonResult()
    for cfg in configs:
        name = f"{cfg.get('strategy', 'naive')}·{cfg.get('model', 'openai/gpt-4o-mini')}"
        _print(f"▶ evaluating {name} …")
        try:
            report = evaluate(
                questions=questions,
                index=shared_index,                      # reuse!
                reference_answers=reference_answers,
                strategy=cfg.get("strategy", "naive"),
                strategy_config=cfg.get("strategy_config"),
                model=cfg.get("model", "openai/gpt-4o-mini"),
                judge_model=cfg.get("judge_model", cfg.get("model", "openai/gpt-4o-mini")),
                metrics=metrics,
            )
        except Exception as e:
            # one bad config (missing key, unknown model, ...) shouldn't discard
            # results already computed for the other configs in this comparison
            _print(f"  ✗ {name} failed: {e}")
            result.errors[name] = str(e)
            continue
        result.runs[name] = report
        result.matrix[name] = report.summary()

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Strategy recommendation — "which RAG strategy is best for MY data?"
# ──────────────────────────────────────────────────────────────────────────────

QUALITY_METRIC_NAMES = (
    "faithfulness", "answer_relevance", "answer_correctness",
    "context_precision", "context_recall", "hit_rate", "mrr",
)


@dataclass
class RecommendationResult:
    leaderboard: List[Dict[str, Any]]      # sorted best -> worst, one row per strategy
    best: Optional[str]
    reasoning: str
    comparison: ComparisonResult

    def to_dict(self) -> dict:
        return {
            "leaderboard": self.leaderboard,
            "best": self.best,
            "reasoning": self.reasoning,
            "matrix": self.comparison.matrix,
            "errors": self.comparison.errors,
        }

    def print_summary(self) -> None:
        _print(f"\n🏆 best strategy for this dataset: {self.best}")
        _print(f"   {self.reasoning}\n")
        _print(f"{'rank':<5}{'strategy':<16}{'score':<8}{'cost($)':<12}{'latency(s)'}")
        for i, row in enumerate(self.leaderboard, 1):
            _print(f"{i:<5}{row['strategy']:<16}{row['composite_score']:<8}"
                   f"{row['total_cost_usd']:<12}{row['avg_latency_s']}")


def recommend_strategy(
    questions: List[str],
    documents: List[Dict[str, Any]],
    reference_answers: Optional[List[Optional[str]]] = None,
    strategies: Optional[List[str]] = None,
    model: str = "openai/gpt-4o-mini",
    embedding_model: str = "openai/text-embedding-3-small",
    judge_model: Optional[str] = None,
    metrics: Union[str, List[str]] = "quality",
    quality_weight: float = 0.7,
    cost_weight: float = 0.15,
    latency_weight: float = 0.15,
) -> RecommendationResult:
    """
    Run every RAG strategy (or a chosen subset) over the SAME corpus + questions
    and recommend the best-scoring one for this specific dataset.

    Ranks by a composite score: mean quality-metric score, penalized by
    normalized cost and latency (weights configurable — bump `quality_weight`
    toward 1.0 to only care about answer quality, or raise `cost_weight` /
    `latency_weight` for a budget/real-time-constrained deployment).

    Example::

        rec = recommend_strategy(questions, documents, reference_answers=refs,
                                  model="groq/openai/gpt-oss-20b",
                                  embedding_model="google/gemini-embedding-001")
        print(rec.best, rec.reasoning)
        for row in rec.leaderboard:
            print(row["strategy"], row["composite_score"])
    """
    names = strategies or sorted(STRATEGIES)
    configs = [{"strategy": s, "model": model,
                "judge_model": judge_model or model} for s in names]

    cmp_res = compare(
        questions=questions, documents=documents, configs=configs,
        reference_answers=reference_answers, metrics=metrics,
        embedding_model=embedding_model,
    )

    rows = []
    for cfg_name, agg in cmp_res.matrix.items():
        strategy_name = cfg_name.split("·", 1)[0]
        quality_scores = [agg[m] for m in QUALITY_METRIC_NAMES if m in agg]
        quality = statistics.fmean(quality_scores) if quality_scores else 0.0
        rows.append({
            "strategy": strategy_name,
            "quality_score": round(quality, 4),
            "total_cost_usd": agg.get("total_cost_usd", 0.0),
            "avg_latency_s": agg.get("avg_latency_s", 0.0),
        })

    max_cost = max((r["total_cost_usd"] for r in rows), default=0.0) or 1e-9
    max_latency = max((r["avg_latency_s"] for r in rows), default=0.0) or 1e-9
    for r in rows:
        cost_penalty = r["total_cost_usd"] / max_cost
        latency_penalty = r["avg_latency_s"] / max_latency
        r["composite_score"] = round(
            quality_weight * r["quality_score"]
            - cost_weight * cost_penalty
            - latency_weight * latency_penalty,
            4,
        )

    rows.sort(key=lambda r: r["composite_score"], reverse=True)
    best = rows[0]["strategy"] if rows else None
    reasoning = (
        f"highest composite score ({rows[0]['composite_score']}) — "
        f"quality={rows[0]['quality_score']}, cost=${rows[0]['total_cost_usd']:.6f}, "
        f"latency={rows[0]['avg_latency_s']}s across {len(questions)} question(s)"
    ) if rows else "no strategies evaluated"

    return RecommendationResult(leaderboard=rows, best=best, reasoning=reasoning, comparison=cmp_res)
