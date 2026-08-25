# Contributing to RagArena

Thanks for helping make RAG evaluation less guesswork! 🎯

## Dev setup

```bash
git clone https://github.com/RagArena/RagArena && cd RagArena
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[all,dev]"
pre-commit install                                    # if hooks configured
pytest                                                # run tests
```

## Project layout

```
src/RagArena/
├── catalog.py      # model registry: providers, pricing, context windows
├── router.py       # unified completion()/embedding()/rerank() across providers
├── strategies.py   # 13 RAG pipelines (all implement Strategy.run())
├── metrics.py      # retrieval + LLM-judge + operational metrics
├── index.py        # VectorIndex: chunking, embedding, search
├── engine.py       # evaluate() / compare() orchestration
├── api/            # FastAPI server + dashboard.html SPA
└── cli.py          # RagArena serve | models | run | compare
```

## Adding a model

Append an entry to `CHAT_MODELS` / `EMBEDDING_MODELS` / `RERANK_MODELS` in `catalog.py`.
OpenAI-compatible providers need zero new code — just add the `ProviderConfig`
(base URL + env var name) and models. Native SDKs only for Anthropic/Cohere/Bedrock.

## Adding a strategy

Subclass `Strategy`, implement `run()` returning `StrategyResult`, register it in
`STRATEGIES`. It automatically becomes available in Python API, CLI, dashboard & HTTP API.

## Ground rules

- Every PR needs tests (`tests/`) — mock providers, never call real APIs in CI.
- Type hints required; `ruff check src/` must pass.
- Keep the public API surface small and documented.
