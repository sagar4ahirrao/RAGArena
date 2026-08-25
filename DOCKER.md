# RagArena — Docker Image 🐳

The official RagArena image bundles the **FastAPI evaluation engine** and the pre-built
**Next.js playground UI** in a single container on a single port (`4000`) — no Python or
Node install required, and no Node runtime needed at container runtime either (the
playground is compiled to static files at image-build time and served by FastAPI).

```
sagar4ahirrao/ragarena:latest
```

Published automatically on every `v*` GitHub release tag, to both Docker Hub and GHCR
(`ghcr.io/sagar4ahirrao/ragarena`), based on `python:3.11-slim` with the `retrieval`,
`ingest`, `datasets` and `providers` extras pre-installed.

---

## 1. Quick start (Docker Compose — recommended)

```bash
# 1. create a .env file with the provider keys you need (see table below)
cat > .env <<'EOF'
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIza...
OPENAI_API_KEY=sk-...
EOF

# 2. launch
docker compose up -d

# 3. open the playground
open http://localhost:4000
```

---

## 2. Quick start (plain Docker)

```bash
docker run -d --name ragarena \
  -p 4000:4000 \
  -e GROQ_API_KEY=$GROQ_API_KEY \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v ragarena_cache:/root/.cache \
  sagar4ahirrao/ragarena:latest
```

---

## 3. Environment variables

| Variable | Purpose | Example |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI chat + embeddings | `sk-...` |
| `GEMINI_API_KEY` | Google Gemini chat + embeddings | `AIza...` |
| `GROQ_API_KEY` | Groq chat (fast/cheap) | `gsk_...` |
| `ANTHROPIC_API_KEY` | Claude chat | `sk-ant-...` |
| `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` | Azure OpenAI deployments | — |
| `AZURE_FOUNDRY_KEY` | Azure AI Foundry (Llama-4, DeepSeek) | — |
| `COHERE_API_KEY` | Cohere rerank/embed | `...` |
| `VOYAGE_API_KEY` | Voyage embeddings | `...` |
| `HF_TOKEN` | HuggingFace datasets + gated models | `hf_...` |
| `OPENROUTER_API_KEY` | OpenRouter (100+ models via one key) | `sk-or-...` |

Check `GET /api/env-status` on a running container to see which providers it actually
detects. Local embeddings (`huggingface/sentence-transformers/all-MiniLM-L6-v2`) work
with no key at all after the model downloads once.

---

## 4. What's inside the image

| Path | Component |
| --- | --- |
| `ragarena.api.server:app` (uvicorn) | FastAPI + evaluation engine, async jobs, `/api/*`, and the playground UI at `/` |
| `src/ragarena/api/ui_dist/` | Next.js static export (built at image-build time, no Node at runtime) |

Key endpoints:

- `GET  /api/options` / `/api/env-status` — models, strategies, metrics, datasets, configured providers
- `POST /api/evaluate` / `POST /api/compare` / `POST /api/recommend` — run, benchmark, or auto-pick the best strategy for your data (each returns a `run_id`, polled via `GET /api/runs/{id}`)
- `GET  /api/datasets` / `GET /api/datasets/{name}` — bundled + HuggingFace benchmark datasets
- `POST /api/ingest` / `POST /api/upload` — parse pasted text or upload pdf/docx/pptx/html/csv/json/xlsx/image/txt/md

---

## 5. Persisting data

The container caches HuggingFace models/datasets under `/root/.cache`:

```bash
-v ragarena_cache:/root/.cache
```

(already configured in `docker-compose.yml`).

---

## 6. Building the image locally

```bash
docker build -t ragarena:dev .
docker run -d -p 4000:4000 ragarena:dev
```

---

## 7. Publishing (maintainers)

```bash
# bump version in pyproject.toml + src/ragarena/__init__.py, then:
git tag v0.2.2
git push origin v0.2.2          # triggers .github/workflows/docker.yml
```

The workflow pushes to GHCR automatically (uses the built-in `GITHUB_TOKEN`) and to
Docker Hub if `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` are set under
**Repository → Settings → Secrets and variables → Actions**.

---

## 8. Troubleshooting

| Symptom | Fix |
| --- | --- |
| Evaluations hang | check `docker logs ragarena` — usually a missing/invalid provider key |
| Local embeddings slow on first run | first launch downloads the model; mount the cache volume |
| Out of disk building the image | the `local` extra (torch) is large — only install it if you need local embeddings |

---

## 9. License

MIT — same as the RagArena source. The container bundles third-party model weights
governed by their respective licenses (e.g. `all-MiniLM-L6-v2` is Apache-2.0).
