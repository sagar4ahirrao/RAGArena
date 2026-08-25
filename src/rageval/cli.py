"""
RAGEval CLI.

    rageval serve                          → web dashboard at localhost:4000
    rageval models list [--provider X]     → browse 100+ model catalog
    rageval run --strategy hybrid ...      → CLI evaluation
    rageval compare --configs c1.yaml ...  → head-to-head benchmark
"""
from __future__ import annotations

import argparse
import json
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


def cmd_serve(args):
    from .api.server import start_server
    url = f"http://localhost:{args.port}"
    print(f"\n  ⚡ RAGEval dashboard → {url}\n")
    start_server(host=args.host, port=args.port)


def build_parser():
    p = argparse.ArgumentParser(prog="rageval",
                                description="⚡ RAG strategy & model evaluation framework")
    sub = p.add_subparsers(dest="command", required=True)

    # serve
    sp = sub.add_parser("serve", help="start web dashboard + API")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=4000)
    sp.set_defaults(fn=cmd_serve)

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
    rp.add_argument("--save")
    rp.set_defaults(fn=cmd_run)

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

    return p


def main():
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
