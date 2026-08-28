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

# LanceDB and Chroma both load native Arrow/Rust libraries. Once LanceDB has
# actually written a table, constructing a Chroma client in the same process
# aborts the interpreter outright — not an exception we could catch, the
# process simply dies. That matters here because the headline use case is
# looping over every backend to benchmark them, which is exactly the pattern
# that triggers it. The reverse order (Chroma first) is fine, so we warn and
# name the fix rather than silently dying mid-benchmark.
_LANCEDB_USED = False


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
    """Chroma in-memory (``EphemeralClient``) by default, a persistent
    directory when ``persist_directory`` is given, or a Chroma server when
    ``host``/``port`` are given."""

    name = "chroma"

    def __init__(self, collection_name: Optional[str] = None,
                 persist_directory: Optional[str] = None,
                 host: Optional[str] = None, port: int = 8000) -> None:
        if _LANCEDB_USED:
            import warnings
            warnings.warn(
                "Creating a Chroma backend after the LanceDB backend has been used in this "
                "process can abort the interpreter (a native Arrow/Rust library conflict — "
                "the process dies rather than raising). Benchmark Chroma BEFORE LanceDB, or "
                "run each backend in its own process.",
                RuntimeWarning, stacklevel=2,
            )
        import chromadb
        if host:
            self._client = chromadb.HttpClient(host=host, port=port)
        elif persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()
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
        global _LANCEDB_USED
        _LANCEDB_USED = True
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


class WeaviateBackend(VectorBackend):
    """Weaviate (server). Vectors are supplied by ragarena, so the collection
    is created with vectorizer 'none' — Weaviate stores and searches, it does
    not embed."""

    name = "weaviate"

    def __init__(self, url: str = "http://localhost:8080", collection_name: Optional[str] = None,
                 api_key: Optional[str] = None, grpc_port: int = 50051) -> None:
        import weaviate
        from weaviate.classes.init import Auth
        from urllib.parse import urlparse
        u = urlparse(url)
        self._client = weaviate.connect_to_custom(
            http_host=u.hostname, http_port=u.port or 8080, http_secure=(u.scheme == "https"),
            grpc_host=u.hostname, grpc_port=grpc_port, grpc_secure=False,
            auth_credentials=Auth.api_key(api_key) if api_key else None,
            skip_init_checks=True,
        )
        self._wvc = __import__("weaviate.classes", fromlist=["classes"])
        # Weaviate collection names must start with a capital letter
        self._name = collection_name or f"Ragarena{uuid.uuid4().hex[:8]}"
        self._count = 0

    def add(self, vectors, texts, metadatas):
        vecs = _normalize(np.asarray(vectors, dtype=np.float32))
        if self._count == 0 and not self._client.collections.exists(self._name):
            self._client.collections.create(
                self._name,
                vectorizer_config=self._wvc.config.Configure.Vectorizer.none(),
                vector_index_config=self._wvc.config.Configure.VectorIndex.hnsw(
                    distance_metric=self._wvc.config.VectorDistances.COSINE),
            )
        coll = self._client.collections.get(self._name)
        with coll.batch.dynamic() as batch:
            for i in range(len(texts)):
                batch.add_object(properties={"text": texts[i], "idx": self._count + i},
                                 vector=vecs[i].tolist())
        self._metas = getattr(self, "_metas", [])
        self._metas.extend(metadatas)
        self._count += len(texts)

    def search(self, vector, k=5, filter=None):
        if self._count == 0:
            return []
        q = _normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))[0]
        coll = self._client.collections.get(self._name)
        res = coll.query.near_vector(
            near_vector=q.tolist(), limit=min(k * 10 if filter else k, self._count),
            return_metadata=self._wvc.query.MetadataQuery(distance=True))
        out: SearchHits = []
        for o in res.objects:
            idx = int(o.properties.get("idx", -1))
            meta = self._metas[idx] if 0 <= idx < len(self._metas) else {}
            if not _matches_filter(meta, filter):
                continue
            dist = o.metadata.distance if o.metadata and o.metadata.distance is not None else 0.0
            out.append((o.properties.get("text", ""), meta, 1.0 - float(dist)))
            if len(out) >= k:
                break
        return out

    def __len__(self):
        return self._count

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass


class ElasticsearchBackend(VectorBackend):
    """Elasticsearch dense_vector field with cosine similarity.

    ``index_type`` defaults to ``"flat"`` (exact brute-force search) rather
    than letting Elasticsearch choose. Since 8.12 its default for an indexed
    dense_vector is ``int8_hnsw``, which scalar-quantizes vectors to 8 bits —
    that introduces roughly 0.4% error in the returned similarity, enough to
    reorder documents that are genuinely close together, so an evaluation run
    would silently score a *different* retrieval than the one being measured.
    Verified here: on a 3-doc corpus with two documents 0.19% apart, the
    int8 default ranked them backwards while ``flat``/``hnsw`` matched exact
    cosine to 6 decimal places.

    Pass ``index_type="hnsw"`` (float32, approximate) or ``"int8_hnsw"``
    deliberately if you are benchmarking ANN behaviour at scale.
    """

    name = "elasticsearch"

    def __init__(self, url: str = "http://localhost:9200", index_name: Optional[str] = None,
                 api_key: Optional[str] = None, basic_auth: Optional[Any] = None,
                 index_type: str = "flat") -> None:
        from elasticsearch import Elasticsearch
        self._es = Elasticsearch(url, api_key=api_key, basic_auth=basic_auth,
                                 verify_certs=False, request_timeout=30)
        self._index = index_name or f"ragarena_{uuid.uuid4().hex[:8]}"
        self._index_type = index_type
        self._metas: List[Dict[str, Any]] = []
        self._count = 0

    def add(self, vectors, texts, metadatas):
        from elasticsearch.helpers import bulk
        vecs = _normalize(np.asarray(vectors, dtype=np.float32))
        if self._count == 0 and not self._es.indices.exists(index=self._index):
            self._es.indices.create(index=self._index, mappings={"properties": {
                "text": {"type": "text"}, "idx": {"type": "integer"},
                "vector": {"type": "dense_vector", "dims": int(vecs.shape[1]),
                           "index": True, "similarity": "cosine",
                           "index_options": {"type": self._index_type}}}})
        bulk(self._es, [{"_index": self._index,
                         "_source": {"text": texts[i], "idx": self._count + i,
                                     "vector": vecs[i].tolist()}}
                        for i in range(len(texts))])
        self._es.indices.refresh(index=self._index)   # make docs searchable now
        self._metas.extend(metadatas)
        self._count += len(texts)

    def search(self, vector, k=5, filter=None):
        if self._count == 0:
            return []
        q = _normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))[0]
        size = min(k * 10 if filter else k, self._count)
        res = self._es.search(index=self._index, size=size,
                              knn={"field": "vector", "query_vector": q.tolist(),
                                   "k": size, "num_candidates": max(size * 2, 10)})
        out: SearchHits = []
        for hit in res["hits"]["hits"]:
            src = hit["_source"]
            idx = int(src.get("idx", -1))
            meta = self._metas[idx] if 0 <= idx < len(self._metas) else {}
            if not _matches_filter(meta, filter):
                continue
            # ES cosine score is (1 + cos) / 2 -> recover raw cosine
            out.append((src.get("text", ""), meta, float(hit["_score"]) * 2.0 - 1.0))
            if len(out) >= k:
                break
        return out

    def __len__(self):
        return self._count


class RedisBackend(VectorBackend):
    """Redis with the RediSearch vector index (redis-stack)."""

    name = "redis"

    def __init__(self, url: str = "redis://localhost:6379", index_name: Optional[str] = None) -> None:
        import redis
        self._redis = redis.from_url(url)
        self._index = index_name or f"ragarena_{uuid.uuid4().hex[:8]}"
        self._prefix = f"{self._index}:"
        self._metas: List[Dict[str, Any]] = []
        self._count = 0

    def add(self, vectors, texts, metadatas):
        from redis.commands.search.field import TextField, VectorField
        try:                                    # redis-py >= 5.1
            from redis.commands.search.index_definition import IndexDefinition, IndexType
        except ImportError:                     # redis-py < 5.1 (camelCase module)
            from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        vecs = _normalize(np.asarray(vectors, dtype=np.float32))
        if self._count == 0:
            self._redis.ft(self._index).create_index(
                fields=[TextField("text"),
                        VectorField("vector", "FLAT",
                                    {"TYPE": "FLOAT32", "DIM": int(vecs.shape[1]),
                                     "DISTANCE_METRIC": "COSINE"})],
                definition=IndexDefinition(prefix=[self._prefix], index_type=IndexType.HASH))
        pipe = self._redis.pipeline()
        for i in range(len(texts)):
            pipe.hset(f"{self._prefix}{self._count + i}",
                      mapping={"text": texts[i], "idx": self._count + i,
                               "vector": vecs[i].astype(np.float32).tobytes()})
        pipe.execute()
        self._metas.extend(metadatas)
        self._count += len(texts)

    def search(self, vector, k=5, filter=None):
        from redis.commands.search.query import Query
        if self._count == 0:
            return []
        q = _normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))[0]
        size = min(k * 10 if filter else k, self._count)
        query = (Query(f"*=>[KNN {size} @vector $vec AS score]")
                 .sort_by("score").return_fields("text", "idx", "score")
                 .dialect(2))
        res = self._redis.ft(self._index).search(
            query, query_params={"vec": q.astype(np.float32).tobytes()})
        out: SearchHits = []
        for doc in res.docs:
            idx = int(getattr(doc, "idx", -1))
            meta = self._metas[idx] if 0 <= idx < len(self._metas) else {}
            if not _matches_filter(meta, filter):
                continue
            # RediSearch reports COSINE *distance* (1 - cos)
            out.append((getattr(doc, "text", ""), meta, 1.0 - float(getattr(doc, "score", 0.0))))
            if len(out) >= k:
                break
        return out

    def __len__(self):
        return self._count


class PgVectorBackend(VectorBackend):
    """PostgreSQL + the pgvector extension."""

    name = "pgvector"

    def __init__(self, dsn: str = "postgresql://postgres:postgres@localhost:5432/postgres",
                 table_name: Optional[str] = None) -> None:
        import psycopg2
        self._psycopg2 = psycopg2
        self._dsn = dsn
        self._table = table_name or f"ragarena_{uuid.uuid4().hex[:8]}"
        self._metas: List[Dict[str, Any]] = []
        self._count = 0
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()

    def _connect(self):
        return self._psycopg2.connect(self._dsn)

    def add(self, vectors, texts, metadatas):
        vecs = _normalize(np.asarray(vectors, dtype=np.float32))
        with self._connect() as conn, conn.cursor() as cur:
            if self._count == 0:
                cur.execute(f'CREATE TABLE IF NOT EXISTS "{self._table}" '
                            f'(idx INTEGER PRIMARY KEY, text TEXT, vector vector({int(vecs.shape[1])}))')
            for i in range(len(texts)):
                cur.execute(f'INSERT INTO "{self._table}" (idx, text, vector) VALUES (%s,%s,%s) '
                            f'ON CONFLICT (idx) DO NOTHING',
                            (self._count + i, texts[i], "[" + ",".join(map(str, vecs[i].tolist())) + "]"))
            conn.commit()
        self._metas.extend(metadatas)
        self._count += len(texts)

    def search(self, vector, k=5, filter=None):
        if self._count == 0:
            return []
        q = _normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))[0]
        qs = "[" + ",".join(map(str, q.tolist())) + "]"
        size = min(k * 10 if filter else k, self._count)
        with self._connect() as conn, conn.cursor() as cur:
            # '<=>' is pgvector's cosine DISTANCE operator
            cur.execute(f'SELECT text, idx, 1 - (vector <=> %s::vector) AS score '
                        f'FROM "{self._table}" ORDER BY vector <=> %s::vector LIMIT %s',
                        (qs, qs, size))
            rows = cur.fetchall()
        out: SearchHits = []
        for text, idx, score in rows:
            meta = self._metas[int(idx)] if 0 <= int(idx) < len(self._metas) else {}
            if not _matches_filter(meta, filter):
                continue
            out.append((text, meta, float(score)))
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
    "weaviate": WeaviateBackend,
    "elasticsearch": ElasticsearchBackend,
    "redis": RedisBackend,
    "pgvector": PgVectorBackend,
}

# Backends that need a running server; list_backends() shouldn't try to
# connect to these just to report availability.
SERVER_BACKENDS = frozenset({"weaviate", "elasticsearch", "redis", "pgvector"})


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
        raise ImportError(_import_error_help(backend, e)) from e


def _import_error_help(backend: str, err: ImportError) -> str:
    """Turn a backend's raw ImportError into something actionable.

    A missing package and an *incompatible* one fail the same way here, but
    need opposite fixes — telling someone to install a package they already
    have sends them the wrong direction, so the known version conflicts are
    named explicitly.
    """
    msg = str(err)
    # chromadb pins opentelemetry only as '>=1.2.0', so pip happily leaves a
    # stale exporter behind while the rest of the stack moves on. The grpc
    # exporter imports private symbols from otlp-proto-common, so the two must
    # be the same version or the import blows up inside chromadb's telemetry.
    if "_create_exp_backoff_generator" in msg or (
        "opentelemetry" in msg and backend == "chroma"
    ):
        return (
            f"Vector backend 'chroma' failed to import because the installed OpenTelemetry "
            f"packages are at mismatched versions (chromadb only requires '>=1.2.0', so pip "
            f"can leave an old exporter in place):\n    {msg}\n"
            f"Fix by aligning them, e.g.:\n"
            f"    pip install -U 'opentelemetry-exporter-otlp-proto-grpc' "
            f"'opentelemetry-exporter-otlp' 'opentelemetry-sdk' 'opentelemetry-api'\n"
            f"(they must all report the same version). Or use backend='numpy' "
            f"(built in, no dependencies)."
        )
    return (
        f"Vector backend '{backend}' needs a package that isn't installed, or one that is "
        f"installed at an incompatible version: {msg}\n"
        f"Install the extras with: pip install 'ragarena[vectordb]'  — or use "
        f"backend='numpy' (built in, no dependencies)."
    )


def list_backends() -> List[Dict[str, Any]]:
    """Every registered backend and whether it's usable here — so a benchmark
    can skip unavailable stores instead of crashing.

    Server-backed stores are reported on whether their *client library*
    imports, not by dialing a server: they need connection details the caller
    supplies (``VectorIndex(backend="pgvector", dsn=...)``), so trying to
    connect with defaults here would report a false negative.
    """
    out = []
    for name, factory in sorted(BACKENDS.items()):
        needs_server = name in SERVER_BACKENDS
        try:
            if needs_server:
                _import_check(name)
            else:
                factory()
            available, reason = True, ("needs a running server — pass connection details"
                                       if needs_server else "")
        except ImportError as e:
            available, reason = False, f"missing dependency: {e}"
        except Exception as e:
            available, reason = False, f"{type(e).__name__}: {e}"
        out.append({"backend": name, "available": available,
                    "needs_server": needs_server, "reason": reason})
    return out


def _import_check(name: str) -> None:
    """Import a server backend's client library without connecting."""
    import importlib
    module = {"weaviate": "weaviate", "elasticsearch": "elasticsearch",
              "redis": "redis", "pgvector": "psycopg2"}[name]
    importlib.import_module(module)
