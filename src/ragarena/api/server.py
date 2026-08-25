"""
FastAPI serving layer + built-in web dashboard.

Start everything with::

    RagArena serve                    # http://localhost:4000
    RagArena serve --port 8080

Endpoints:
    GET  /                     → web dashboard
    GET  /health               → liveness
    GET  /api/catalog          → all providers/models w/ pricing
    GET  /api/strategies       → strategy registry
    GET  /api/metrics          → metric registry
    POST /api/evaluate         → run single evaluation (blocking)
    POST /api/compare          → run multi-config comparison (blocking)
    GET  /api/runs             → stored runs
    GET  /api/runs/{run_id}    → one run detail
"""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from ..catalog import (
    CHAT_MODELS, EMBEDDING_MODELS, RERANK_MODELS, VECTOR_STORES,
    PROVIDERS, EMBEDDING_PROVIDERS, list_models,
)
from ..engine import EvaluationReport, ComparisonResult, evaluate, compare
from ..metrics import METRICS, DEFAULT_METRIC_SETS
from ..strategies import STRATEGIES

_DASHBOARD = Path(__file__).with_name("dashboard.html")

# in-memory run store (swap for Redis/Postgres in production deployments)
_RUNS: Dict[str, dict] = {}
_JOBS: Dict[str, dict] = {}

app = FastAPI(
    title="RagArena API",
    description="Unified evaluation API for RAG strategies, LLMs and embedding models.",
    version="0.1.0",
)


# ──────────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    questions: List[str] = Field(..., min_length=1)
    documents: List[Dict[str, Any]] = Field(..., min_length=1)
    reference_answers: Optional[List[Optional[str]]] = None
    strategy: str = "naive"
    strategy_config: Dict[str, Any] = Field(default_factory=dict)
    model: str = "openai/gpt-4o-mini"
    embedding_model: str = "openai/text-embedding-3-small"
    judge_model: str = "openai/gpt-4o-mini"
    metrics: Any = "production"


class CompareRequest(BaseModel):
    questions: List[str] = Field(..., min_length=1)
    documents: List[Dict[str, Any]] = Field(..., min_length=1)
    configs: List[Dict[str, Any]] = Field(..., min_length=2)
    reference_answers: Optional[List[Optional[str]]] = None
    metrics: Any = "quality"
    embedding_model: str = "openai/text-embedding-3-small"


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    if _DASHBOARD.exists():
        return _DASHBOARD.read_text(encoding="utf-8")
    return "<h1>RagArena</h1><p>dashboard.html missing from install.</p>"


@app.get("/health")
def health():
    return {"status": "ok", "runs_stored": len(_RUNS)}


# ──────────────────────────────────────────────────────────────────────────────
# Catalog endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/catalog")
def catalog(provider: Optional[str] = None, modality: Optional[str] = None):
    models = [m.to_dict() for m in list_models(provider=provider, modality=modality)]  # type: ignore[arg-type]
    return {
        "counts": {
            "chat": len(CHAT_MODELS), "embedding": len(EMBEDDING_MODELS),
            "rerank": len(RERANK_MODELS), "total": models.__len__(),
            "providers": len({m["provider"] for m in models}),
            "vector_stores": len(VECTOR_STORES),
        },
        "models": models,
        "vector_stores": VECTOR_STORES,
        "provider_endpoints": {k: v.base_url or "(native sdk)" for k, v in PROVIDERS.items()},
    }


@app.get("/api/strategies")
def strategies():
    return [
        {"name": s.name, "description": s.description}
        for s in (cls() for cls in STRATEGIES.values())
    ]


@app.get("/api/metrics")
def metrics():
    return {
        "presets": DEFAULT_METRIC_SETS,
        "metrics": [
            {"name": name,
             "requires_reference": cls.requires_reference,
             "llm_judged": cls.requires_llm_judge}
            for name, cls in METRICS.items()
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Run endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/api/evaluate")
def run_evaluate(req: EvaluateRequest):
    try:
        report = evaluate(
            questions=req.questions, documents=req.documents,
            reference_answers=req.reference_answers,
            strategy=req.strategy, strategy_config=req.strategy_config,
            model=req.model, embedding_model=req.embedding_model,
            judge_model=req.judge_model, metrics=req.metrics,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    data = report.to_dict()
    _RUNS[data["run_id"]] = data
    return data


@app.post("/api/compare")
def run_compare(req: CompareRequest):
    try:
        cmp_res = compare(
            questions=req.questions, documents=req.documents,
            configs=req.configs, reference_answers=req.reference_answers,
            metrics=req.metrics, embedding_model=req.embedding_model,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    run_id = uuid.uuid4().hex[:12]
    data = {
        "run_id": run_id,
        "kind": "comparison",
        "matrix": cmp_res.matrix,
        "runs": {k: v.to_dict() for k, v in cmp_res.runs.items()},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _RUNS[run_id] = data
    return data


@app.get("/api/runs")
def list_runs():
    return [{"run_id": rid, "kind": d.get("kind", "single"),
             "created_at": d.get("created_at"),
             "strategy": d.get("strategy") or f"{len(d.get('matrix', {}))} configs",
             "aggregate": d.get("aggregate") or d.get("matrix")}
            for rid, d in sorted(_RUNS.items(), key=lambda kv: kv[0])]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    if run_id not in _RUNS:
        raise HTTPException(404, "run not found")
    return _RUNS[run_id]


def start_server(host: str = "0.0.0.0", port: int = 4000, reload: bool = False):
    import uvicorn
    uvicorn.run("RagArena.api.server:app", host=host, port=port,
                reload=False, log_level="info")


if __name__ == "__main__":
    start_server()
