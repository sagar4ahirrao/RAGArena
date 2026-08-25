"""Provider + dataset coverage test for RagArena.

Validates that ALL strategies run end-to-end against free providers:
  - generation:  Azure AI Foundry (Llama-4-Maverick / DeepSeek-V3.2)
  - embeddings:  local HuggingFace sentence-transformers (no API needed)

Run:  TEST_MODEL=azure_foundry/Llama-4-Maverick-17B-128E-Instruct-FP8 \
            AZURE_FOUNDRY_KEY=...  HF_TOKEN=...  python examples/03_provider_test.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ragarena import evaluate, STRATEGIES

EMB = "huggingface/sentence-transformers/all-MiniLM-L6-v2"
GEN = os.getenv("TEST_MODEL", "azure_foundry/Llama-4-Maverick-17B-128E-Instruct-FP8")

# ── Variety of datasets (domains + a multimodal one) ──────────────────────────
DATASETS = [
    {
        "name": "tech-rag",
        "docs": [
            {"text": "RagArena is an open-source RAG evaluation framework that benchmarks "
                     "retrieval-augmented generation strategies, LLMs, and embedding models "
                     "against each other on your own corpus.", "metadata": {"src": "a"}},
            {"text": "Graph RAG builds an entity-relationship graph over the corpus and supports "
                     "local retrieval (entity-linked chunks) and global retrieval (community "
                     "summaries) for hierarchical reasoning.", "metadata": {"src": "b"}},
        ],
        "questions": ["What is RagArena?", "How does graph retrieval work?"],
        "refs": [
            "RagArena is an open-source RAG evaluation framework.",
            "Graph retrieval uses an entity graph for local and global retrieval.",
        ],
    },
    {
        "name": "medical",
        "docs": [
            {"text": "Type 2 diabetes is managed with metformin as first-line therapy, alongside "
                     "diet and exercise. HbA1c targets are typically below 7%.", "metadata": {"src": "m1"}},
            {"text": "Hypoglycemia is treated with fast-acting glucose; glucagon is used when the "
                     "patient is unconscious.", "metadata": {"src": "m2"}},
        ],
        "questions": ["What is first-line therapy for type 2 diabetes?",
                      "How is hypoglycemia treated?"],
        "refs": ["Metformin is first-line therapy for type 2 diabetes.",
                 "Hypoglycemia is treated with fast-acting glucose or glucagon."],
    },
    {
        "name": "legal",
        "docs": [
            {"text": "A Non-Disclosure Agreement (NDA) prohibits the receiving party from "
                     "disclosing confidential information for a period of three years.", "metadata": {"src": "l1"}},
            {"text": "Indemnification clauses shift liability from one party to another for "
                     "third-party claims arising from the agreement.", "metadata": {"src": "l2"}},
        ],
        "questions": ["What does an NDA prohibit?", "What does an indemnification clause do?"],
        "refs": ["An NDA prohibits disclosing confidential information for three years.",
                 "Indemnification shifts liability for third-party claims."],
    },
    {
        "name": "multimodal",   # exercises the multimodal strategy path
        "docs": [
            {"text": "Quarterly revenue grew 12% QoQ. See the table for segment breakdown.",
             "metadata": {"src": "q1"},
             "tables": [["Segment", "Revenue", "YoY"],
                        ["Cloud", "4.2B", "+18%"], ["Hardware", "1.1B", "-3%"]],
             "images": ["chart_revenue.png"]},
            {"text": "Operating margin expanded to 31% driven by AI workload demand.",
             "metadata": {"src": "q2"}},
        ],
        "questions": ["What drove the revenue growth?", "What was the operating margin?"],
        "refs": ["Cloud segment growth of 18% drove revenue.",
                 "Operating margin expanded to 31%."],
    },
]


def run_all():
    total = len(DATASETS) * len(STRATEGIES)
    done = 0
    fails = []
    t_start = time.perf_counter()
    print(f"Generation model : {GEN}")
    print(f"Embedding model  : {EMB}")
    print(f"Strategies       : {len(STRATEGIES)} -> {', '.join(STRATEGIES)}")
    print(f"Datasets         : {len(DATASETS)} ({', '.join(d['name'] for d in DATASETS)})")
    print("=" * 78)

    for ds in DATASETS:
        print(f"\n### Dataset: {ds['name']}  ({len(ds['questions'])} questions)")
        col_w = max(len(s) for s in STRATEGIES)
        for strat in STRATEGIES:
            try:
                rep = evaluate(
                    questions=ds["questions"], documents=ds["docs"],
                    strategy=strat, model=GEN, embedding_model=EMB,
                    reference_answers=ds["refs"], metrics="quick",
                    max_concurrency=1,
                )
                ans = rep.samples[0].answer
                ok = bool(ans) and not ans.startswith("<ERROR>")
                status = "OK " if ok else "EMPTY"
                hit = rep.samples[0].metrics.get("hit_rate", {}).get("score")
                print(f"  {strat:<{col_w}} {status}  hit_rate={hit}  "
                      f"cost=${rep.total_cost_usd:.4f}  t={rep.wall_time_s:5.1f}s")
                if not ok:
                    fails.append(f"{ds['name']}/{strat}: {ans[:60]}")
            except Exception as e:
                fails.append(f"{ds['name']}/{strat}: EXC {e}")
                print(f"  {strat:<{col_w}} FAIL  {type(e).__name__}: {str(e)[:80]}")
            done += 1
        print(f"  progress: {done}/{total}")

    print("\n" + "=" * 78)
    print(f"DONE in {time.perf_counter()-t_start:.1f}s | "
          f"{total-len(fails)}/{total} strategy-dataset runs produced an answer")
    if fails:
        print(f"\n{len(fails)} issue(s):")
        for f in fails:
            print("  -", f)


# Optional: confirm LLM-as-judge metrics also work (one dataset, 3 strategies)
def run_judge():
    print("\n--- LLM-judge sanity (faithfulness/answer_relevance) ---")
    ds = DATASETS[0]
    for strat in ("naive", "hyde", "graph_hybrid"):
        try:
            rep = evaluate(
                questions=ds["questions"], documents=ds["docs"],
                strategy=strat, model=GEN, embedding_model=EMB,
                reference_answers=ds["refs"], metrics="production",
                judge_model=GEN, max_concurrency=1,
            )
            m = rep.samples[0].metrics
            print(f"  {strat:<12} faith={m.get('faithfulness',{}).get('score')} "
                  f"rel={m.get('answer_relevance',{}).get('score')}")
        except Exception as e:
            print(f"  {strat:<12} FAIL {type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    run_all()
    run_judge()
