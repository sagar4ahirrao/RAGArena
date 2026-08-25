"""
Graph-augmented retrieval index (LightRAG-inspired dual-level retrieval).

``GraphIndex`` wraps a :class:`~ragarena.index.VectorIndex` and layers a
lightweight knowledge graph on top of it:

* entities are extracted from every chunk (via the LLM, with a deterministic
  keyword fallback when the model is unavailable);
* chunks that share entities are clustered into *communities*;
* queries can be answered at two levels:

  - **local**  — entity-precise retrieval: match the query's entities to graph
    nodes and pull the connected chunks (great for "who/what" factual lookups);
  - **global** — macro-theme retrieval: summarise each community, then rank the
    communities by relevance to the query and synthesise a cross-document
    answer (great for "how/why" analytical questions);
  - **hybrid** / **mix** — combine both levels.

The graph is built lazily and cached on the wrapped index so a shared index in
``compare()`` is only processed once.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .router import completion, embedding, Usage


_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your", "are",
    "was", "were", "been", "being", "have", "has", "had", "will", "would", "can",
    "could", "should", "may", "might", "must", "about", "which", "their", "there",
    "what", "when", "where", "which", "how", "why", "than", "then", "they", "them",
    "such", "some", "more", "most", "other", "these", "those", "our", "its", "his",
    "her", "but", "not", "all", "any", "one", "two", "also", "each", "between",
}


def _norm(token: str) -> str:
    return re.sub(r"[^a-z0-9+#.\-]", "", token.lower().strip())


def _keyword_entities(text: str, max_entities: int = 12) -> Set[str]:
    """Deterministic fallback entity extractor (no LLM required)."""
    entities: Set[str] = set()
    for m in re.findall(r"[A-Za-z][A-Za-z0-9+#.\-]+(?:\s[A-Za-z0-9+#.\-]+){0,2}", text):
        parts = m.split()
        # keep multi-word phrases only when capitalised, else single tokens
        if len(parts) > 1 and any(p[0].isupper() for p in parts):
            ent = _norm(m)
            if ent:
                entities.add(ent)
            continue
        tok = _norm(m)
        if len(tok) >= 4 and tok not in _STOPWORDS:
            entities.add(tok)
    # also capture Capitalised multi-word concepts
    return set(list(entities)[:max_entities])


class GraphIndex:
    """Entity/community graph layered over a ``VectorIndex``."""

    MODES = ("local", "global", "hybrid", "mix")

    def __init__(self, vector_index: "VectorIndex"):
        self.vi = vector_index
        self.entities: List[Set[str]] = []          # chunk_idx -> entities
        self.entity_map: Dict[str, Set[int]] = {}   # entity -> chunk indices
        self.communities: List[List[int]] = []       # list of chunk-index groups
        self.community_summaries: Dict[int, str] = {}  # community_id -> summary
        self._summary_vecs: Dict[int, List[float]] = {}  # community_id -> embedding
        self._built = False

    # ── construction ──────────────────────────────────────────────────────────
    @classmethod
    def from_documents(cls, documents: List[Dict[str, Any]],
                       embedding_model: str = "openai/text-embedding-3-small") -> "GraphIndex":
        from .index import VectorIndex
        vi = VectorIndex(embedding_model=embedding_model)
        vi.add_documents(documents)
        return cls(vi)

    # ── building the graph ────────────────────────────────────────────────────
    def build(self, llm_model: Optional[str] = None) -> "GraphIndex":
        if self._built or len(self.vi) == 0:
            return self
        self.entities = [_extract_entities(t, llm_model) for t in self.vi.texts]
        self.entity_map = {}
        for idx, ents in enumerate(self.entities):
            for e in ents:
                self.entity_map.setdefault(e, set()).add(idx)
        self._cluster()
        self._built = True
        return self

    def _cluster(self) -> None:
        """Union-find over chunks that share at least one entity."""
        n = len(self.vi)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for ent, idxs in self.entity_map.items():
            idxs = list(idxs)
            for other in idxs[1:]:
                union(idxs[0], other)

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        # sort communities by size (desc) so global search favours breadth
        self.communities = sorted(groups.values(), key=len, reverse=True)

    # ── local (entity-precise) retrieval ───────────────────────────────────────
    def local_search(self, query: str, k: int = 5,
                     llm_model: Optional[str] = None,
                     embedding_model: Optional[str] = None,
                     filter: Optional[Dict[str, Any]] = None) -> List["Chunk"]:
        from .strategies import Chunk
        self.build(llm_model)
        q_entities = _extract_entities(query, llm_model)

        # score every chunk by entity overlap + a dense-similarity contribution
        dense = self.vi.search_with_scores(query, k=min(k * 4, len(self.vi)),
                                           embed_model=embedding_model, filter=filter)
        dense_score = {id(c.text): s for c, s in dense}

        scored: List[Tuple[Chunk, float]] = []
        seen = set()
        for idx, chunk in enumerate(self.vi.texts):
            overlap = len(q_entities & self.entities[idx])
            if overlap == 0 and chunk[:200] not in dense_score:
                continue
            sim = dense_score.get(chunk[:200], 0.0)
            score = overlap * 1.0 + sim
            if overlap == 0 and sim < 0.3:
                continue
            if chunk[:200] in seen:
                continue
            seen.add(chunk[:200])
            ch = Chunk(text=chunk, metadata=self.vi.metadatas[idx], score=float(score))
            scored.append((ch, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        # guarantee at least top dense results if no entity match
        if not scored and dense:
            scored = [(c, s) for c, s in dense]
        return [c for c, _ in scored[:k]]

    # ── global (macro-theme) retrieval ─────────────────────────────────────────
    def _ensure_summaries(self, llm_model: Optional[str]) -> None:
        for cid in range(len(self.communities)):
            if cid in self.community_summaries:
                continue
            chunk_ids = self.communities[cid]
            blob = "\n\n".join(self.vi.texts[i] for i in chunk_ids[:8])
            summary = _summarize(blob, llm_model)
            self.community_summaries[cid] = summary
            try:
                vec = embedding(model=self.vi.embedding_model, input=[summary]).vectors[0]
                self._summary_vecs[cid] = vec
            except Exception:
                self._summary_vecs[cid] = []

    def global_search(self, query: str, k: int = 5,
                      n_communities: int = 4,
                      llm_model: Optional[str] = None,
                      embedding_model: Optional[str] = None,
                      filter: Optional[Dict[str, Any]] = None) -> Tuple[List["Chunk"], str]:
        from .strategies import Chunk
        from .strategies import _rag_prompt
        self.build(llm_model)
        if not self.communities:
            return [], ""
        self._ensure_summaries(llm_model)

        # rank communities by summary similarity to the query
        try:
            qvec = embedding(model=embedding_model or self.vi.embedding_model,
                             input=[query]).vectors[0]
        except Exception:
            qvec = None

        ranked = []
        for cid, summary in self.community_summaries.items():
            if qvec is not None and self._summary_vecs.get(cid):
                s = _cosine(qvec, self._summary_vecs[cid])
            else:
                s = 0.0
            ranked.append((cid, s))
        ranked.sort(key=lambda x: x[1], reverse=True)
        top = [cid for cid, _ in ranked[:n_communities]]

        chosen_chunks: List[Chunk] = []
        theme_context: List[str] = []
        for cid in top:
            theme_context.append(f"[theme] {self.community_summaries[cid]}")
            for i in self.communities[cid][:4]:
                meta = self.vi.metadatas[i]
                if filter and not _passes(meta, filter):
                    continue
                if self.vi.texts[i][:200] not in {c.text[:200] for c in chosen_chunks}:
                    chosen_chunks.append(Chunk(text=self.vi.texts[i], metadata=meta))
        theme_block = "\n".join(theme_context)
        return chosen_chunks[:k], theme_block

    # ── unified dispatch ────────────────────────────────────────────────────────
    def search(self, query: str, mode: str = "hybrid", k: int = 5,
               llm_model: Optional[str] = None,
               embedding_model: Optional[str] = None,
               filter: Optional[Dict[str, Any]] = None) -> Tuple[List["Chunk"], str]:
        self.build(llm_model)
        if mode == "local":
            return self.local_search(query, k=k, llm_model=llm_model,
                                     embedding_model=embedding_model, filter=filter), ""
        if mode == "global":
            return self.global_search(query, k=k, llm_model=llm_model,
                                      embedding_model=embedding_model, filter=filter)
        if mode == "mix":
            local = self.local_search(query, k=k, llm_model=llm_model,
                                     embedding_model=embedding_model, filter=filter)
            gchunks, theme = self.global_search(query, k=k, llm_model=llm_model,
                                                embedding_model=embedding_model, filter=filter)
            merged = _merge(local, gchunks)[:k]
            return merged, theme
        # hybrid (default)
        local = self.local_search(query, k=k, llm_model=llm_model,
                                 embedding_model=embedding_model, filter=filter)
        gchunks, theme = self.global_search(query, k=k, llm_model=llm_model,
                                            embedding_model=embedding_model, filter=filter)
        merged = _merge(local, gchunks)[:k]
        return merged, theme


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_entities(text: str, llm_model: Optional[str]) -> Set[str]:
    if llm_model:
        try:
            resp = completion(model=llm_model, temperature=0, max_tokens=200, messages=[
                {"role": "user", "content":
                 "Extract the key entities (proper nouns, technologies, concepts, "
                 "metrics, people, organisations) from the text. "
                 "Return ONLY a JSON array of short lowercase phrases, e.g. "
                 '["rag", "vector database"]. No prose.\n\nText:\n' + text[:3000]}])
            match = re.search(r"\[.*?\]", resp.text, re.DOTALL)
            if match:
                items = json.loads(match.group(0))
                return {_norm(str(i)) for i in items if _norm(str(i))}
        except Exception:
            pass
    return _keyword_entities(text)


def _summarize(text: str, llm_model: Optional[str]) -> str:
    if not llm_model:
        return text[:400]
    try:
        resp = completion(model=llm_model, temperature=0, max_tokens=120, messages=[
            {"role": "user", "content":
             "Write one concise factual summary (max 2 sentences) covering the main "
             "points of the text.\n\n" + text[:3000]}])
        return resp.text.strip()
    except Exception:
        return text[:400]


def _cosine(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _passes(meta: Dict[str, Any], filter: Dict[str, Any]) -> bool:
    return all(meta.get(k) == v or (isinstance(v, list) and meta.get(k) in v)
               for k, v in filter.items())


def _merge(a: List["Chunk"], b: List["Chunk"]) -> List["Chunk"]:
    out: List["Chunk"] = []
    seen = set()
    for c in a + b:
        if c.text[:200] in seen:
            continue
        seen.add(c.text[:200])
        out.append(c)
    return out
