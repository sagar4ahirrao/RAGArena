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

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..catalog import (
    CHAT_MODELS, EMBEDDING_MODELS, RERANK_MODELS, VECTOR_STORES,
    PROVIDERS, EMBEDDING_PROVIDERS, list_models,
)
from ..engine import EvaluationReport, ComparisonResult, RecommendationResult, evaluate, compare, recommend_strategy
from ..metrics import METRICS, DEFAULT_METRIC_SETS
from ..strategies import STRATEGIES
from ..datasets import load_dataset, list_datasets
from ..ingest import parse_file

_DASHBOARD = Path(__file__).with_name("dashboard.html")
_UI_DIST = Path(__file__).with_name("ui_dist")     # pre-built Next.js static export (professional playground UI)

# in-memory run store (swap for Redis/Postgres in production deployments)
_RUNS: Dict[str, dict] = {}
_JOBS: Dict[str, dict] = {}

app = FastAPI(
    title="RagArena API",
    description="Unified evaluation API for RAG strategies, LLMs and embedding models.",
    version="0.6.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    judge_samples: int = 1


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
    ui_index = _UI_DIST / "index.html"
    if ui_index.exists():
        return ui_index.read_text(encoding="utf-8")
    if _DASHBOARD.exists():
        return _DASHBOARD.read_text(encoding="utf-8")
    return "<h1>RagArena</h1><p>UI assets missing from install.</p>"


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
    run_id = uuid.uuid4().hex[:12]
    _JOB = {"status": "running", "created_at": time.time()}

    def _worker():
        try:
            report = evaluate(
                questions=req.questions, documents=req.documents,
                reference_answers=req.reference_answers,
                strategy=req.strategy, strategy_config=req.strategy_config,
                model=req.model, embedding_model=req.embedding_model,
                judge_model=req.judge_model, metrics=req.metrics,
                chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap,
                judge_samples=req.judge_samples,
            )
            data = report.to_dict()
            _RUNS[run_id] = data
            _JOB["status"] = "done"
        except Exception as e:  # pragma: no cover
            _JOB["status"] = "error"
            _JOB["error"] = str(e)

    threading.Thread(target=_worker, daemon=True).start()
    _JOBS[run_id] = _JOB
    return {"run_id": run_id, "status": "running"}


@app.post("/api/compare")
def run_compare(req: CompareRequest):
    run_id = uuid.uuid4().hex[:12]
    _JOB = {"status": "running", "created_at": time.time()}

    def _worker():
        try:
            cmp_res = compare(
                questions=req.questions, documents=req.documents,
                configs=req.configs, reference_answers=req.reference_answers,
                metrics=req.metrics, embedding_model=req.embedding_model,
            )
            data = {
                "run_id": run_id,
                "kind": "comparison",
                "matrix": cmp_res.matrix,
                "runs": {k: v.to_dict() for k, v in cmp_res.runs.items()},
                "errors": cmp_res.errors,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            _RUNS[run_id] = data
            _JOB["status"] = "done"
        except Exception as e:  # pragma: no cover
            _JOB["status"] = "error"
            _JOB["error"] = str(e)

    threading.Thread(target=_worker, daemon=True).start()
    _JOBS[run_id] = _JOB
    return {"run_id": run_id, "status": "running"}


class RecommendRequest(BaseModel):
    questions: List[str] = Field(..., min_length=1)
    documents: List[Dict[str, Any]] = Field(..., min_length=1)
    reference_answers: Optional[List[Optional[str]]] = None
    strategies: Optional[List[str]] = None
    model: str = "openai/gpt-4o-mini"
    embedding_model: str = "openai/text-embedding-3-small"
    judge_model: Optional[str] = None
    metrics: Any = "quality"
    quality_weight: float = 0.7
    cost_weight: float = 0.15
    latency_weight: float = 0.15


@app.post("/api/recommend")
def run_recommend(req: RecommendRequest):
    """Run every strategy (or a chosen subset) on the SAME corpus and recommend the best one."""
    run_id = uuid.uuid4().hex[:12]
    _JOB = {"status": "running", "created_at": time.time()}

    def _worker():
        try:
            rec = recommend_strategy(
                questions=req.questions, documents=req.documents,
                reference_answers=req.reference_answers, strategies=req.strategies,
                model=req.model, embedding_model=req.embedding_model,
                judge_model=req.judge_model, metrics=req.metrics,
                quality_weight=req.quality_weight, cost_weight=req.cost_weight,
                latency_weight=req.latency_weight,
            )
            data = {
                "run_id": run_id, "kind": "recommendation",
                **rec.to_dict(),
                "runs": {k: v.to_dict() for k, v in rec.comparison.runs.items()},
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            _RUNS[run_id] = data
            _JOB["status"] = "done"
        except Exception as e:  # pragma: no cover
            _JOB["status"] = "error"
            _JOB["error"] = str(e)

    threading.Thread(target=_worker, daemon=True).start()
    _JOBS[run_id] = _JOB
    return {"run_id": run_id, "status": "running"}


@app.get("/api/env-status")
def env_status():
    """Which providers currently have a usable API key/credential in this environment."""
    from ..router import _resolve_api_key
    seen: Dict[str, bool] = {}
    for provider in {**PROVIDERS, **EMBEDDING_PROVIDERS}:
        try:
            seen[provider] = bool(_resolve_api_key(provider, None))
        except Exception:
            seen[provider] = False
    return {"providers": seen, "configured_count": sum(seen.values())}


@app.get("/api/datasets")
def datasets():
    return {"datasets": list_datasets()}


@app.get("/api/datasets/{name}")
def dataset_detail(name: str, n: int = 20, bundled: bool = False):
    try:
        docs, qs, refs = load_dataset(name, n=n, use_bundled=bundled)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "name": name,
        "n_documents": len(docs),
        "n_questions": len(qs),
        "questions": qs,
        "reference_answers": refs,
        "documents": docs,
        "sample_document": docs[0]["text"][:600] if docs else "",
    }


@app.post("/api/ingest")
def ingest(payload: Dict[str, Any]):
    """Parse raw text content (e.g. pasted/uploaded) into documents."""
    text = payload.get("text", "")
    if not text.strip():
        raise HTTPException(status_code=400, detail="empty text")
    docs = [{"text": text, "metadata": {"source": "paste", "type": "txt"}}]
    return {"documents": docs, "n_documents": len(docs)}


@app.post("/api/upload")
async def upload(files: List["UploadFile"] = File(...)):  # type: ignore[name-defined]
    """Parse uploaded files (pdf/docx/pptx/html/csv/json/xlsx/md/txt) into documents."""
    from tempfile import NamedTemporaryFile
    from ..ingest import parse_file  # local import to avoid circulars
    all_docs: List[Dict[str, Any]] = []
    for f in files:
        suffix = os.path.splitext(f.filename or "x.txt")[1] or ".txt"
        with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await f.read())
            tmp_path = tmp.name
        try:
            parsed = parse_file(tmp_path)
            for d in parsed:
                d["metadata"] = {**d.get("metadata", {}), "source": f.filename, "path": f.filename}
            all_docs.extend(parsed)
        except Exception as e:
            all_docs.append({"text": "", "metadata": {"source": f.filename, "error": str(e)[:120]}})
        finally:
            os.unlink(tmp_path)
    all_docs = [d for d in all_docs if d.get("text") or d.get("tables") or d.get("images")]
    return {"documents": all_docs, "n_documents": len(all_docs)}


@app.get("/api/options")
def options():
    chat = [m.to_dict() for m in CHAT_MODELS if m.provider in
            ("azure_foundry", "groq", "google", "openai", "anthropic", "mistral",
             "together", "fireworks", "deepinfra", "openrouter")]
    emb = [m.to_dict() for m in EMBEDDING_MODELS]
    return {
        "chat_models": chat,
        "embedding_models": emb,
        "strategies": [
            {"name": s.name, "description": s.description}
            for s in (cls() for cls in STRATEGIES.values())
        ],
        "metrics_presets": list(DEFAULT_METRIC_SETS.keys()),
        "datasets": list_datasets(),
        "chunk_sizes": [256, 512, 768, 1000, 1500, 2000],
        "chunk_overlaps": [0, 64, 128, 200, 300],
    }


@app.get("/api/runs")
def list_runs():
    return [{"run_id": rid, "kind": d.get("kind", "single"),
             "created_at": d.get("created_at"),
             "strategy": d.get("strategy") or f"{len(d.get('matrix', {}))} configs",
             "aggregate": d.get("aggregate") or d.get("matrix")}
            for rid, d in sorted(_RUNS.items(), key=lambda kv: kv[0])]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    job = _JOBS.get(run_id)
    if job and job["status"] != "done":
        return {"run_id": run_id, "status": job["status"],
                "error": job.get("error"), "kind": "pending"}
    if run_id not in _RUNS:
        raise HTTPException(404, "run not found")
    return {"status": "done", **_RUNS[run_id]}


if (_UI_DIST / "_next").exists():
    app.mount("/_next", StaticFiles(directory=str(_UI_DIST / "_next")), name="ui-assets")

    @app.get("/{path:path}", include_in_schema=False)
    def ui_page(path: str):
        """Serve any file from the Next.js static export: page HTML, the .txt RSC
        prefetch payload Next's client router requests for each nav link, or a
        bare asset. Falls back to {path}/index.html for a directory-style route."""
        safe = (_UI_DIST / path).resolve()
        if _UI_DIST.resolve() not in safe.parents and safe != _UI_DIST.resolve():
            raise HTTPException(404, "page not found")
        if safe.is_file():
            return FileResponse(safe)
        index_html = _UI_DIST / path / "index.html"
        if index_html.exists():
            return FileResponse(index_html)
        raise HTTPException(404, "page not found")


def start_server(host: str = "0.0.0.0", port: int = 4000, reload: bool = False):
    import uvicorn
    uvicorn.run("ragarena.api.server:app", host=host, port=port,
                reload=False, log_level="info")


if __name__ == "__main__":
    start_server()
