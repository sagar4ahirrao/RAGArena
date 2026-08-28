"""
Vector index abstraction — FAISS (in-memory) by default, zero-config.

``VectorIndex`` handles chunking, embedding, storage and search so strategies
stay simple. Swap in Chroma/Pinecone/Qdrant via ``backend=``.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .router import embedding, Usage

# Content types preserved intact (not sentence-split) by the multimodal pipeline.
_TYPED_DOCS = {"table", "image", "equation"}


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


@dataclass
class MultimodalDocument:
    """
    A single document of a specific content type.

    Tables, images and equations are kept intact (not sentence-split) so their
    structure survives retrieval — the approach used by multimodal RAG systems.
    """

    content: str
    doc_type: str = "text"          # text | table | image | equation
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        m = dict(self.metadata)
        m["doc_type"] = self.doc_type
        return {"text": self.content, "metadata": m}


class VectorIndex:
    """
    Embedding-backed search index.

    Example::

        index = VectorIndex(embedding_model="openai/text-embedding-3-small")
        index.add_documents([{"text": "...", "metadata": {...}}, ...])
        hits = index.search("what is rag?", k=5)
    """

    def __init__(self, embedding_model: str = "openai/text-embedding-3-small",
                 chunker: Optional[TextChunker] = None,
                 backend: Any = "numpy", **backend_kwargs: Any):
        from .backends import get_backend

        self.embedding_model = embedding_model
        self.chunker = chunker or TextChunker()
        # `backend` swaps only where vectors are stored and searched — same
        # embeddings, same strategies on top — so vector stores themselves can
        # be benchmarked against each other. See ragarena.list_backends().
        self.backend = get_backend(backend, **backend_kwargs)
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        # embed-usage accounting from the last add/search (for cost rollup)
        self.last_embed_usage: List[Usage] = []

    # ── ingestion ────────────────────────────────────────────────────────────
    def add_documents(self, documents: List[Union[Dict[str, Any], "MultimodalDocument"]]) -> int:
        new_texts: List[str] = []
        new_metas: List[Dict[str, Any]] = []
        for raw in documents:
            doc = raw.to_dict() if isinstance(raw, MultimodalDocument) else raw
            text = doc.get("text") or doc.get("content") or doc.get("page_content") or ""
            meta = dict(doc.get("metadata", {}) or {})
            doc_type = (meta.get("doc_type") or doc.get("doc_type") or "text").lower()

            # tables / images / equations are preserved intact (only split if huge)
            if doc_type in _TYPED_DOCS and len(text) > self.chunker.chunk_size:
                pieces = self.chunker.split(text)
            elif doc_type in _TYPED_DOCS:
                pieces = [text] if text.strip() else []
            else:
                pieces = self.chunker.split(text)

            for piece in pieces:
                new_texts.append(piece)
                m = dict(meta)
                m["doc_type"] = doc_type
                m.setdefault("chunk_id", str(uuid.uuid4())[:8])
                new_metas.append(m)

        if not new_texts:
            # Documents came in but none produced indexable text — most often a
            # folder of images (parse_file() returns them with empty `text` and
            # the bytes in `images`). Silently returning 0 leaves an empty index
            # and an evaluation that retrieves nothing, with no clue why.
            if documents:
                import warnings
                image_like = sum(
                    1 for raw in documents
                    if isinstance(raw, dict) and raw.get("images") and not (raw.get("text") or "").strip()
                )
                hint = (" They look like image documents — convert them with "
                        "ragarena.to_multimodal() and use strategy='multimodal' "
                        "to retrieve over images." if image_like else "")
                warnings.warn(
                    f"add_documents(): {len(documents)} document(s) produced 0 indexable "
                    f"text chunks, so nothing was added to the index.{hint}",
                    UserWarning, stacklevel=2,
                )
            return 0

        resp = embedding(model=self.embedding_model, input=new_texts)
        self.last_embed_usage = [resp.usage]

        vecs = np.asarray(resp.vectors, dtype=np.float32)
        self.backend.add(vecs, new_texts, new_metas)
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

        if not self.texts:
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

        hits = self.backend.search(np.asarray(vector, dtype=np.float32), k=k, filter=filter)
        out: List[Tuple[Chunk, float]] = [
            (Chunk(text=text, metadata=meta, score=score), score)
            for text, meta, score in hits
        ]
        return out if with_scores else [c for c, _ in out]   # type: ignore[return-value]

    def __len__(self):
        return len(self.texts)


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[str]:
    """Standalone helper."""
    return TextChunker(chunk_size, chunk_overlap).split(text)
