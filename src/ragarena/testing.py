"""
CI-friendly assertions over an EvaluationReport — gate a merge on RAG
quality the same way you'd gate on a unit test:

    from ragarena import evaluate
    from ragarena.testing import assert_metric

    def test_hybrid_strategy_faithfulness():
        report = evaluate(questions=Q, documents=DOCS, reference_answers=REFS,
                           strategy="hybrid", metrics=["faithfulness"])
        assert_metric(report, "faithfulness", gte=0.8)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .engine import EvaluationReport


class MetricAssertionError(AssertionError):
    """Raised by assert_metric() when a report's metric score fails its bound."""


def assert_metric(
    report: "EvaluationReport",
    metric: str,
    gte: Optional[float] = None,
    lte: Optional[float] = None,
    eq: Optional[float] = None,
) -> float:
    """Assert a report's aggregate score for `metric` satisfies the given
    bound(s) — pass any combination of `gte`/`lte`/`eq`. Raises
    MetricAssertionError (a plain AssertionError subclass, so pytest reports
    it like any other failed assert) with the actual score and the run_id on
    failure, so a CI log tells you which run and by how much it missed.

    Returns the actual score on success, for chaining/logging.
    """
    summary = report.summary()
    if metric not in summary:
        raise MetricAssertionError(
            f"metric '{metric}' not found in report {report.run_id} "
            f"(available: {sorted(summary)}) — was it included in the `metrics=` list?"
        )
    score = summary[metric]

    failures = []
    if gte is not None and score < gte:
        failures.append(f"expected >= {gte}, got {score}")
    if lte is not None and score > lte:
        failures.append(f"expected <= {lte}, got {score}")
    if eq is not None and score != eq:
        failures.append(f"expected == {eq}, got {score}")

    if failures:
        raise MetricAssertionError(
            f"[{report.run_id}] {metric} failed: {'; '.join(failures)} "
            f"(strategy={report.strategy}, model={report.model})"
        )
    return score


def assert_no_regression(diff, threshold: float = 0.0) -> None:
    """Assert a RunDiff (from ``diff_runs()``) has no metric regressions
    beyond `threshold`. Raises MetricAssertionError listing every regressed
    metric with its before/after values if any are found."""
    regressed = diff.regressions(threshold)
    if regressed:
        lines = [f"{name}: {diff.deltas[name]['before']} -> {diff.deltas[name]['after']}"
                for name in regressed]
        raise MetricAssertionError(
            f"[{diff.run_id_a} -> {diff.run_id_b}] regression(s) detected: " + "; ".join(lines)
        )
