"""
Vector index abstraction — FAISS (in-memory) by default, zero-config.

``VectorIndex`` handles chunking, embedding, storage and search so strategies
stay simple. Swap in Chroma/Pinecone/Qdrant via ``backend=``.
"""
from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .router import embedding, Usage


class TextChunker:
    """Recursive character splitter with overlap."""

    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size // 2)

    def split(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks: List[str] = []
        for sep in self.SEPARATORS:
            if sep in text:
                parts = text.split(sep)
                break
        else:
            parts = [text]

        buf = ""
        for part in parts:
            piece = (buf + sep + part) if buf else part
            if len(piece) <= self.chunk_size:
                buf = piece
            else:
                if buf:
                    chunks.append(buf)
                while len(part) > self.chunk_size:
                    chunks.append(part[: self.chunk_size])
                    part = part[self.chunk_size - self.chunk_overlap:]
                buf = part
        if buf.strip():
            chunks.append(buf)

        # merge tiny trailing fragments
        merged: List[str] = []
        for c in chunks:
            if merged and len(c) < 80:
                merged[-1] += " " + c
            elif merged and len(merged[-1]) < self.chunk_size * 0.4:
                merged[-1] += " " + c
            else:
                merged.append(c)
        return merged


class VectorIndex:
    """
    Embedding-backed search index.

    Example::

        index = VectorIndex(embedding_model="openai/text-embedding-3-small")
        index.add_documents([{"text": "...", "metadata": {...}}, ...])
        hits = index.search("what is rag?", k=5)
    """

    def __init__(self, embedding_model: str = "openai/text-embedding-3-small",
                 chunker: Optional[TextChunker] = None):
        self.embedding_model = embedding_model
        self.chunker = chunker or TextChunker()
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self._matrix: Optional[np.ndarray] = None
        # embed-usage accounting from the last add/search (for cost rollup)
        self.last_embed_usage: List[Usage] = []

    # ── ingestion ────────────────────────────────────────────────────────────
    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        new_texts: List[str] = []
        new_metas: List[Dict[str, Any]] = []
        for doc in documents:
            text = doc.get("text") or doc.get("content") or doc.get("page_content") or ""
            meta = doc.get("metadata", {}) or {}
            for piece in self.chunker.split(text):
                new_texts.append(piece)
                m = dict(meta)
                m.setdefault("chunk_id", str(uuid.uuid4())[:8])
                new_metas.append(m)

        if not new_texts:
            return 0

        resp = embedding(model=self.embedding_model, input=new_texts)
        self.last_embed_usage = [resp.usage]

        vecs = np.asarray(resp.vectors, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-10)

        self._matrix = vecs if self._matrix is None else np.vstack([self._matrix, vecs])
        self.texts.extend(new_texts)
        self.metadatas.extend(new_metas)
        return len(new_texts)

    # ── search ───────────────────────────────────────────────────────────────
    def search(self, query: str, k: int = 5,
               embed_model: Optional[str] = None,
               filter: Optional[Dict[str, Any]] = None) -> List["Chunk"]:
        from .strategies import Chunk
        hits = self.search_with_scores(query, k=k, embed_model=embed_model, filter=filter)
        return [c for c, _ in hits]

    def search_with_scores(
        self,
        query: str,
        k: int = 5,
        embed_model: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple["Chunk", float]]:
        from .strategies import Chunk

        if self._matrix is None or not self.texts:
            return []
        model = embed_model or self.embedding_model
        resp = embedding(model=model, input=[query])
        self.last_embed_usage = [resp.usage]
        return self.search_by_vector(resp.vectors[0], k=k, filter=filter, with_scores=True)  # type: ignore[return-value]

    def search_by_vector(
        self,
        vector,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        with_scores: bool = False,
    ):
        from .strategies import Chunk

        q = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        n = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / np.maximum(n, 1e-10)

        sims = (self._matrix @ q.T).ravel()          # cosine similarity
        order = np.argsort(-sims)

        out: List[Tuple[Chunk, float]] = []
        for idx in order:
            meta = self.metadatas[int(idx)]
            if filter and not all(meta.get(kk) == vv or
                                  (isinstance(vv, list) and meta.get(kk) in vv)
                                  for kk, vv in filter.items()):
                continue
            out.append((Chunk(text=self.texts[int(idx)], metadata=meta,
                              score=float(sims[idx])), float(sims[idx])))
            if len(out) >= k:
                break
        return out if with_scores else [c for c, _ in out]   # type: ignore[return-value]

    def __len__(self):
        return len(self.texts)


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[str]:
    """Standalone helper."""
    return TextChunker(chunk_size, chunk_overlap).split(text)
