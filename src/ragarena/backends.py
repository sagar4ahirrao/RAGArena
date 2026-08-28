"""
Pluggable vector-store backends — so the vector DB itself becomes something
you can benchmark, not a fixed part of the framework.

    from ragarena import VectorIndex

    index = VectorIndex(embedding_model="...", backend="faiss")

Every backend implements the same tiny interface (`add` / `search` / `__len__`),
so swapping one changes only where vectors live and how ANN search runs —
identical corpus, identical embeddings, identical strategies on top. That's
what makes a fair "which vector DB performs best for my data" comparison
possible.

Backends that run in-process (no server to stand up): ``numpy`` (default,
exact cosine, zero deps), ``faiss``, ``chroma``, ``qdrant`` (in-memory mode),
``lancedb``. Server-backed stores (Weaviate, Milvus, pgvector, Elasticsearch,
Redis, Pinecone) are reachable through the same interface by subclassing
``VectorBackend`` — see ``register_backend()``.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# (text, metadata, score) triples, best-first
SearchHits = List[Tuple[str, Dict[str, Any], float]]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-10)


def _matches_filter(meta: Dict[str, Any], filter: Optional[Dict[str, Any]]) -> bool:
    if not filter:
        return True
    return all(meta.get(k) == v or (isinstance(v, list) and meta.get(k) in v)
               for k, v in filter.items())


class VectorBackend(ABC):
    """Storage + nearest-neighbour search for one corpus of embeddings.

    Implementations receive already-embedded vectors — embedding is the
    router's job, not the store's, so every backend is compared on identical
    vectors rather than on whatever embedding each DB bundles.
    """

    name: str = "base"

    @abstractmethod
    def add(self, vectors: np.ndarray, texts: List[str],
            metadatas: List[Dict[str, Any]]) -> None: ...

    @abstractmethod
    def search(self, vector: np.ndarray, k: int = 5,
               filter: Optional[Dict[str, Any]] = None) -> SearchHits: ...

    @abstractmethod
    def __len__(self) -> int: ...


class NumpyBackend(VectorBackend):
    """Exact cosine similarity over an in-memory matrix. Zero dependencies,
    exact (not approximate) results — the right default for evaluation, where
    a small corpus and correct ranking matter more than ANN speed."""

    name = "numpy"

    def __init__(self) -> None:
        self._matrix: Optional[np.ndarray] = None
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

    def add(self, vectors, texts, metadatas):
        vecs = _normalize(np.asarray(vectors, dtype=np.float32))
        self._matrix = vecs if self._matrix is None else np.vstack([self._matrix, vecs])
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

    def search(self, vector, k=5, filter=None):
        if self._matrix is None or not self.texts:
            return []
        q = _normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))
        sims = (self._matrix @ q.T).ravel()
        out: SearchHits = []
        for idx in np.argsort(-sims):
            meta = self.metadatas[int(idx)]
            if not _matches_filter(meta, filter):
                continue
            out.append((self.texts[int(idx)], meta, float(sims[idx])))
            if len(out) >= k:
                break
        return out

    def __len__(self):
        return len(self.texts)


class FaissBackend(VectorBackend):
    """FAISS ``IndexFlatIP`` over L2-normalized vectors (inner product on
    normalized vectors == cosine). Filtering is applied after search, so a
    highly selective filter may return fewer than k hits — it over-fetches to
    compensate."""

    name = "faiss"

    def __init__(self) -> None:
        import faiss  # noqa: F401  (import here so the dep stays optional)
        self._faiss = faiss
        self._index = None
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

    def add(self, vectors, texts, metadatas):
        vecs = _normalize(np.asarray(vectors, dtype=np.float32))
        if self._index is None:
            self._index = self._faiss.IndexFlatIP(vecs.shape[1])
        self._index.add(vecs)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

    def search(self, vector, k=5, filter=None):
        if self._index is None or not self.texts:
            return []
        q = _normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))
        fetch = min(len(self.texts), k * 10 if filter else k)
        scores, idxs = self._index.search(q, fetch)
        out: SearchHits = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            meta = self.metadatas[int(idx)]
            if not _matches_filter(meta, filter):
                continue
            out.append((self.texts[int(idx)], meta, float(score)))
            if len(out) >= k:
                break
        return out

    def __len__(self):
        return len(self.texts)


class ChromaBackend(VectorBackend):
    """Chroma via its in-memory (``EphemeralClient``) mode by default, or a
    persistent directory when ``persist_directory`` is given."""

    name = "chroma"

    def __init__(self, collection_name: Optional[str] = None,
                 persist_directory: Optional[str] = None) -> None:
        import chromadb
        self._client = (chromadb.PersistentClient(path=persist_directory)
                        if persist_directory else chromadb.EphemeralClient())
        self._collection = self._client.get_or_create_collection(
            name=collection_name or f"ragarena_{uuid.uuid4().hex[:8]}",
            metadata={"hnsw:space": "cosine"},
        )
        self._count = 0

    def add(self, vectors, texts, metadatas):
        vecs = _normalize(np.asarray(vectors, dtype=np.float32))
        ids = [f"{self._count + i}" for i in range(len(texts))]
        # Chroma rejects empty metadata dicts and non-primitive values
        cleaned = [{k: v for k, v in m.items() if isinstance(v, (str, int, float, bool))}
                   or {"_": ""} for m in metadatas]
        self._collection.add(ids=ids, embeddings=vecs.tolist(),
                             documents=texts, metadatas=cleaned)
        self._count += len(texts)

    def search(self, vector, k=5, filter=None):
        if self._count == 0:
            return []
        q = _normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))
        res = self._collection.query(query_embeddings=q.tolist(),
                                     n_results=min(k * 10 if filter else k, self._count))
        out: SearchHits = []
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            meta = dict(meta or {})
            if not _matches_filter(meta, filter):
                continue
            out.append((text, meta, 1.0 - float(dist)))   # cosine distance -> similarity
            if len(out) >= k:
                break
        return out

    def __len__(self):
        return self._count


class QdrantBackend(VectorBackend):
    """Qdrant, defaulting to its in-process ``:memory:`` mode so no server is
    needed for a benchmark run; pass ``url=`` to hit a real deployment."""

    name = "qdrant"

    def __init__(self, collection_name: Optional[str] = None,
                 url: Optional[str] = None, api_key: Optional[str] = None) -> None:
        from qdrant_client import QdrantClient
        self._qmodels = __import__("qdrant_client.models", fromlist=["models"])
        self._client = (QdrantClient(url=url, api_key=api_key) if url
                        else QdrantClient(":memory:"))
        self._name = collection_name or f"ragarena_{uuid.uuid4().hex[:8]}"
        self._count = 0

    def add(self, vectors, texts, metadatas):
        vecs = _normalize(np.asarray(vectors, dtype=np.float32))
        if self._count == 0:
            self._client.recreate_collection(
                collection_name=self._name,
                vectors_config=self._qmodels.VectorParams(
                    size=vecs.shape[1], distance=self._qmodels.Distance.COSINE),
            )
        points = [
            self._qmodels.PointStruct(id=self._count + i, vector=vecs[i].tolist(),
                                      payload={"text": texts[i], "metadata": metadatas[i]})
            for i in range(len(texts))
        ]
        self._client.upsert(collection_name=self._name, points=points)
        self._count += len(texts)

    def search(self, vector, k=5, filter=None):
        if self._count == 0:
            return []
        q = _normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))[0]
        hits = self._client.search(collection_name=self._name, query_vector=q.tolist(),
                                   limit=min(k * 10 if filter else k, self._count))
        out: SearchHits = []
        for h in hits:
            payload = h.payload or {}
            meta = payload.get("metadata", {}) or {}
            if not _matches_filter(meta, filter):
                continue
            out.append((payload.get("text", ""), meta, float(h.score)))
            if len(out) >= k:
                break
        return out

    def __len__(self):
        return self._count


class LanceDBBackend(VectorBackend):
    """LanceDB — embedded, file-backed (no server). Defaults to a temp
    directory so a benchmark run leaves nothing behind unless ``uri`` is set."""

    name = "lancedb"

    def __init__(self, uri: Optional[str] = None, table_name: Optional[str] = None) -> None:
        import tempfile

        import lancedb
        self._db = lancedb.connect(uri or tempfile.mkdtemp(prefix="ragarena_lance_"))
        self._table_name = table_name or f"ragarena_{uuid.uuid4().hex[:8]}"
        self._table = None
        self._metas: List[Dict[str, Any]] = []
        self._count = 0

    def add(self, vectors, texts, metadatas):
        vecs = _normalize(np.asarray(vectors, dtype=np.float32))
        # metadata is kept alongside rather than in the table so arbitrary
        # nested dicts don't have to fit a fixed Arrow schema
        rows = [{"vector": vecs[i].tolist(), "text": texts[i], "idx": self._count + i}
                for i in range(len(texts))]
        if self._table is None:
            self._table = self._db.create_table(self._table_name, data=rows)
        else:
            self._table.add(rows)
        self._metas.extend(metadatas)
        self._count += len(texts)

    def search(self, vector, k=5, filter=None):
        if self._table is None or self._count == 0:
            return []
        q = _normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))[0]
        rows = (self._table.search(q.tolist())
                .limit(min(k * 10 if filter else k, self._count)).to_list())
        out: SearchHits = []
        for r in rows:
            meta = self._metas[int(r["idx"])] if int(r["idx"]) < len(self._metas) else {}
            if not _matches_filter(meta, filter):
                continue
            # LanceDB returns L2 distance on the normalized vectors;
            # ||a-b||^2 = 2 - 2cos  =>  cos = 1 - dist/2
            score = 1.0 - float(r.get("_distance", 0.0)) / 2.0
            out.append((r["text"], meta, score))
            if len(out) >= k:
                break
        return out

    def __len__(self):
        return self._count


BACKENDS: Dict[str, Callable[..., VectorBackend]] = {
    "numpy": NumpyBackend,
    "faiss": FaissBackend,
    "chroma": ChromaBackend,
    "qdrant": QdrantBackend,
    "lancedb": LanceDBBackend,
}


def register_backend(name: str, factory: Callable[..., VectorBackend]) -> None:
    """Register a custom VectorBackend under `name`, usable as
    ``VectorIndex(backend=name)`` — the extension point for server-backed
    stores (Weaviate, Milvus, pgvector, Elasticsearch, Redis, Pinecone) and
    for anything proprietary."""
    BACKENDS[name] = factory


def get_backend(backend: Any = "numpy", **kwargs: Any) -> VectorBackend:
    """Resolve a backend name (or pass through an already-built
    VectorBackend instance)."""
    if isinstance(backend, VectorBackend):
        return backend
    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown vector backend '{backend}'. Available: {sorted(BACKENDS)}. "
            f"Register your own with ragarena.register_backend(name, factory)."
        )
    try:
        return BACKENDS[backend](**kwargs)
    except ImportError as e:
        raise ImportError(
            f"Vector backend '{backend}' needs an extra package that isn't installed: {e}. "
            f"Install it, or use backend='numpy' (built in, no dependencies)."
        ) from e


def list_backends() -> List[Dict[str, Any]]:
    """Every registered backend and whether its dependencies are importable
    here — so a benchmark can skip unavailable stores instead of crashing."""
    out = []
    for name, factory in sorted(BACKENDS.items()):
        try:
            factory()
            available = True
            reason = ""
        except ImportError as e:
            available, reason = False, str(e)
        except Exception as e:                    # needs a running server / config
            available, reason = False, f"{type(e).__name__}: {e}"
        out.append({"backend": name, "available": available, "reason": reason})
    return out
