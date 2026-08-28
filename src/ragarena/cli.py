"""
RagArena CLI.

    RagArena serve                          → web dashboard at localhost:4000
    RagArena models list [--provider X]     → browse 100+ model catalog
    RagArena ask --query "..." --documents ... → get an answer (strategy='auto' picks the best one)
    RagArena testgen --documents ...        → generate a synthetic Q&A test set
    RagArena run --strategy hybrid ...      → CLI evaluation
    RagArena diff --a run1.json --b run2.json → regression check between two runs
    RagArena compare --configs c1.yaml ...  → head-to-head benchmark
    RagArena recommend --documents ... --questions ... → best strategy for YOUR data
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import yaml


def _print_models_table(models):
    from .catalog import PROVIDERS, EMBEDDING_PROVIDERS
    name_of = {**{k: v.name for k, v in PROVIDERS.items()},
               **{k: v.name for k, v in EMBEDDING_PROVIDERS.items()}}
    print(f"\n{'MODEL ID':<62} {'TYPE':<10} {'CTX':>7}  {'$/1M in':>8}  {'$/1M out':>9}  NOTES")
    print("─" * 118)
    for m in models:
        ctx = f"{m.context_window // 1000}k" if m.modality == "chat" else "—"
        pin = f"${m.input_cost}" if m.input_cost else "free"
        pout = f"${m.output_cost}" if m.output_cost else "—"
        print(f"{m.id:<62} {m.modality:<10} {ctx:>7}  {pin:>8}  {pout:>9}  {m.description[:40]}")
    print()


def cmd_models(args):
    from .catalog import list_models, list_providers
    if args.action == "providers":
        for p in list_providers():
            print(f"  {p['provider']:<14} ({p['display_name']}) — {p['models']} models")
        return
    models = list_models(provider=args.provider, modality=args.modality)
    _print_models_table(models)


def cmd_backends(_args):
    from .backends import list_backends
    print("\nVector-store backends (swap with VectorIndex(backend=...)):\n")
    for b in list_backends():
        mark = "ok " if b["available"] else "-- "
        note = "" if b["available"] else f"  ({b['reason'][:60]})"
        print(f"  {mark} {b['backend']:<12}{note}")
    print()


def cmd_strategies(_args):
    from .strategies import STRATEGIES
    print("\nAvailable RAG strategies:\n")
    for cls in STRATEGIES.values():
        s = cls()
        print(f"  {s.name:<16} {s.description}")
    print()


def cmd_run(args):
    from .engine import evaluate

    docs = [{"text": t} for t in args.documents]
    questions = [l.strip() for l in args.questions.split(",") if l.strip()]
    refs = [r.strip() if r.strip() else None
            for r in (args.references or "").split(",")]

    report = evaluate(
        questions=questions, documents=docs,
        reference_answers=refs if any(refs) else None,
        strategy=args.strategy,
        strategy_config=json.loads(args.config) if args.config else {},
        model=args.model,
        embedding_model=args.embedding,
        judge_model=args.judge,
        metrics=args.metrics,
        judge_samples=args.judge_samples,
    )
    report.print_summary()
    if args.save:
        report.save(args.save)
        print(f"saved → {args.save}")


def cmd_compare(args):
    from .engine import compare

    cfgs = []
    srcs = list(args.configs)
    if args.inline:
        cfgs.append(json.loads(args.inline))
    if not srcs:
        sys.exit("provide --configs files or --inline '{...}'")
    for path in srcs:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        cfgs.extend(loaded["configs"] if isinstance(loaded, dict) else loaded)

    docs = [{"text": t} for t in args.documents]
    questions = [l.strip() for l in args.questions.split(",") if l.strip()]

    result = compare(
        questions=questions, documents=docs, configs=cfgs,
        metrics=args.metrics, embedding_model=args.embedding,
    )
    result.print_leaderboard(sort_by=args.sort)
    if args.save:
        result.save(args.save)
        print(f"\nsaved → {args.save}")


def cmd_recommend(args):
    from .engine import recommend_strategy

    docs = [{"text": t} for t in args.documents]
    questions = [l.strip() for l in args.questions.split(",") if l.strip()]
    refs = [r.strip() if r.strip() else None
            for r in (args.references or "").split(",")]

    rec = recommend_strategy(
        questions=questions, documents=docs,
        reference_answers=refs if any(refs) else None,
        strategies=args.strategies.split(",") if args.strategies else None,
        model=args.model, embedding_model=args.embedding, judge_model=args.judge,
        metrics=args.metrics,
        quality_weight=args.quality_weight, cost_weight=args.cost_weight,
        latency_weight=args.latency_weight,
    )
    rec.print_summary()
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(rec.to_dict(), f, indent=2)
        print(f"saved -> {args.save}")


def cmd_ask(args):
    from .engine import answer

    docs = [{"text": t} for t in args.documents]
    eval_qs = [q.strip() for q in (args.auto_questions or "").split(",") if q.strip()] or None

    result = answer(
        query=args.query, documents=docs, strategy=args.strategy,
        model=args.model, embedding_model=args.embedding,
        auto_eval_questions=eval_qs, return_details=True,
    )
    print(f"\nstrategy used : {result['strategy']}")
    print(f"answer        : {result['answer']}")
    print(f"latency       : {result['latency_s']:.2f}s")


def cmd_testgen(args):
    from .testgen import generate_testset_detailed

    docs = [{"text": t} for t in args.documents]
    cases = generate_testset_detailed(
        docs, n=args.n, model=args.model,
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
        seed=args.seed,
    )
    for i, c in enumerate(cases, 1):
        print(f"\n[{i}] ({c.question_type}) {c.question}\n    -> {c.reference_answer}")
    print(f"\ngenerated {len(cases)} question(s)")
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in cases], f, indent=2)
        print(f"saved -> {args.save}")


def cmd_diff(args):
    from .engine import EvaluationReport, diff_runs

    report_a = EvaluationReport.load(args.a)
    report_b = EvaluationReport.load(args.b)
    d = diff_runs(report_a, report_b)
    d.print_diff()
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(d.to_dict(), f, indent=2)
        print(f"saved -> {args.save}")


def cmd_serve(args):
    from .api.server import start_server
    url = f"http://localhost:{args.port}"
    print(f"\n  ⚡ RagArena dashboard → {url}\n")
    start_server(host=args.host, port=args.port)


def cmd_ui(args):
    """Launch the FastAPI backend + the Next.js playground together."""
    web_dir = Path(__file__).resolve().parent.parent.parent / "web"
    if not web_dir.exists():
        print("error: web/ frontend not found — run `ragarena serve` for the API-only dashboard")
        sys.exit(1)

    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "ragarena.api.server:app",
         "--host", args.host, "--port", str(args.port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    frontend = subprocess.Popen(
        ["npm", "--prefix", str(web_dir), "run", "dev", "--", "--port", str(args.web_port)],
        cwd=str(web_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )

    def _stop(*_):
        for p in (backend, frontend):
            try:
                p.terminate()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    print(f"\n  ⚡ RagArena API      → http://localhost:{args.port}")
    print(f"  ⚡ RagArena Playground → http://localhost:{args.web_port}\n  (Ctrl+C to stop)\n")
    try:
        backend.wait()
    finally:
        _stop()


def build_parser():
    p = argparse.ArgumentParser(prog="ragarena",
                                description="⚡ RAG strategy & model evaluation framework")
    sub = p.add_subparsers(dest="command", required=True)

    # serve
    sp = sub.add_parser("serve", help="start web dashboard + API")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=4000)
    sp.set_defaults(fn=cmd_serve)

    # ui — Next.js playground + API
    up = sub.add_parser("ui", help="launch Next.js playground + API (recommended)")
    up.add_argument("--host", default="0.0.0.0")
    up.add_argument("--port", type=int, default=4000, help="backend API port")
    up.add_argument("--web-port", type=int, default=3000, help="frontend port")
    up.set_defaults(fn=cmd_ui)

    # models
    mp = sub.add_parser("models", help="browse model catalog")
    msub = mp.add_subparsers(dest="action")
    ml = msub.add_parser("list")
    ml.add_argument("--provider", default=None)
    ml.add_argument("--modality", choices=["chat", "embedding", "rerank"], default=None)
    ml.set_defaults(fn=cmd_models)
    mpr = msub.add_parser("providers")
    mpr.set_defaults(fn=cmd_models)

    # strategies
    stp = sub.add_parser("strategies", help="list RAG strategies")
    stp.set_defaults(fn=cmd_strategies)

    # backends
    bp = sub.add_parser("backends", help="list vector-store backends and their availability")
    bp.set_defaults(fn=cmd_backends)

    # run
    rp = sub.add_parser("run", help="run one evaluation")
    rp.add_argument("--documents", nargs="+", required=True)
    rp.add_argument("--questions", required=True, help="comma-separated")
    rp.add_argument("--references", help="comma-separated ground truth")
    rp.add_argument("--strategy", default="naive")
    rp.add_argument("--config", help='JSON dict e.g. \'{"k":8}\'')
    rp.add_argument("--model", default="openai/gpt-4o-mini")
    rp.add_argument("--embedding", default="openai/text-embedding-3-small")
    rp.add_argument("--judge", default="openai/gpt-4o-mini")
    rp.add_argument("--metrics", default="production")
    rp.add_argument("--judge-samples", type=int, default=1, dest="judge_samples",
                    help="average N independent judge calls to reduce LLM-judge variance (default 1)")
    rp.add_argument("--save")
    rp.set_defaults(fn=cmd_run)

    # ask — plug-and-play answering (not just evaluation)
    ap = sub.add_parser("ask", help="get an answer from your documents (strategy='auto' picks the best one)")
    ap.add_argument("--query", required=True)
    ap.add_argument("--documents", nargs="+", required=True)
    ap.add_argument("--strategy", default="auto", help="strategy name, or 'auto' to pick the best one")
    ap.add_argument("--auto-questions", dest="auto_questions",
                    help="comma-separated sample questions used to pick the strategy when --strategy=auto")
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--embedding", default="openai/text-embedding-3-small")
    ap.set_defaults(fn=cmd_ask)

    # testgen
    tp = sub.add_parser("testgen", help="generate a synthetic Q&A test set from documents")
    tp.add_argument("--documents", nargs="+", required=True)
    tp.add_argument("--n", type=int, default=20, help="number of questions to generate")
    tp.add_argument("--model", default="openai/gpt-4o-mini")
    tp.add_argument("--chunk-size", type=int, default=1000, dest="chunk_size")
    tp.add_argument("--chunk-overlap", type=int, default=150, dest="chunk_overlap")
    tp.add_argument("--seed", type=int, default=None)
    tp.add_argument("--save", help="save generated cases to a JSON file")
    tp.set_defaults(fn=cmd_testgen)

    # diff
    dp = sub.add_parser("diff", help="compare two saved evaluation runs (regression check)")
    dp.add_argument("--a", required=True, help="path to first run's saved JSON")
    dp.add_argument("--b", required=True, help="path to second run's saved JSON")
    dp.add_argument("--save", help="save the diff to a JSON file")
    dp.set_defaults(fn=cmd_diff)

    # compare
    cp = sub.add_parser("compare", help="benchmark multiple configs")
    cp.add_argument("--documents", nargs="+", required=True)
    cp.add_argument("--questions", required=True)
    cp.add_argument("--configs", nargs="*", help="YAML/JSON config files")
    cp.add_argument("--inline", help='single JSON config {"strategy":"hybrid","model":"..."}')
    cp.add_argument("--metrics", default="quality")
    cp.add_argument("--embedding", default="openai/text-embedding-3-small")
    cp.add_argument("--sort", default="faithfulness")
    cp.add_argument("--save")
    cp.set_defaults(fn=cmd_compare)

    # recommend
    rec = sub.add_parser("recommend", help="run every strategy on your data & recommend the best one")
    rec.add_argument("--documents", nargs="+", required=True)
    rec.add_argument("--questions", required=True, help="comma-separated")
    rec.add_argument("--references", help="comma-separated ground truth")
    rec.add_argument("--strategies", help="comma-separated subset (default: all 18)")
    rec.add_argument("--model", default="openai/gpt-4o-mini")
    rec.add_argument("--embedding", default="openai/text-embedding-3-small")
    rec.add_argument("--judge", default="openai/gpt-4o-mini")
    rec.add_argument("--metrics", default="quality")
    rec.add_argument("--quality-weight", type=float, default=0.7, dest="quality_weight")
    rec.add_argument("--cost-weight", type=float, default=0.15, dest="cost_weight")
    rec.add_argument("--latency-weight", type=float, default=0.15, dest="latency_weight")
    rec.add_argument("--save")
    rec.set_defaults(fn=cmd_recommend)

    return p


def main():
    # Make Unicode box-drawing / arrows in output safe on non-UTF-8 consoles
    # (e.g. Windows cmd/PowerShell default to cp1252/cp437 and crash on '→', '·', '▶').
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    args = build_parser().parse_args()
    try:
        args.fn(args)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
