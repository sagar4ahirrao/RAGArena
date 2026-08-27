"""
Synthetic test-set generation — turn source documents into a Q&A dataset
automatically, so evaluating RAG strategies doesn't require hand-writing
questions first.

    from ragarena import generate_testset, evaluate

    questions, references = generate_testset(documents, n=20, model="openai/gpt-4o-mini")
    report = evaluate(questions=questions, reference_answers=references,
                       documents=documents, strategy="hybrid")
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .index import TextChunker
from .router import completion

_TYPE_PROMPTS = {
    "simple": (
        "Write ONE factual question that is answerable directly and only from "
        "the passage below, plus its correct answer. The question should read "
        "naturally, as something a real user would ask — do not reference "
        '"the passage" or "the document" in the question.'
    ),
    "reasoning": (
        "Write ONE question that requires reasoning over multiple facts stated "
        "in the passage below (not a single fact lookup — e.g. comparing, "
        "combining, or inferring from two or more details), plus its correct "
        "answer."
    ),
    "multi_context": (
        "Write ONE question whose answer genuinely requires combining "
        "information from BOTH passages below (not answerable from either "
        "alone), plus its correct answer synthesizing both."
    ),
}

DEFAULT_TYPE_MIX = {"simple": 0.5, "reasoning": 0.3, "multi_context": 0.2}


@dataclass
class TestCase:
    question: str
    reference_answer: str
    question_type: str
    source_chunks: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "reference_answer": self.reference_answer,
            "question_type": self.question_type,
            "source_chunks": self.source_chunks,
        }


def _extract_qa(text: str) -> Optional[Tuple[str, str]]:
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(match.group(0)) if match else json.loads(text)
        q, a = str(data.get("question", "")).strip(), str(data.get("answer", "")).strip()
        return (q, a) if q and a else None
    except Exception:
        return None


def _chunks_from_documents(documents: List[Dict[str, Any]],
                           chunk_size: int, chunk_overlap: int) -> List[str]:
    chunker = TextChunker(chunk_size, chunk_overlap)
    chunks: List[str] = []
    for doc in documents:
        text = doc.get("text", "") if isinstance(doc, dict) else str(doc)
        chunks.extend(c for c in chunker.split(text) if len(c.strip()) > 40)
    return chunks


def _build_plan(n: int, type_mix: Dict[str, float], seed: Optional[int]) -> List[str]:
    rng = random.Random(seed)
    plan: List[str] = []
    for qtype, frac in type_mix.items():
        plan.extend([qtype] * max(0, round(n * frac)))
    while len(plan) < n:
        plan.append("simple")
    plan = plan[:n]
    rng.shuffle(plan)
    return plan


def _generate_cases(
    documents: List[Dict[str, Any]],
    n: int,
    model: Any,
    chunk_size: int,
    chunk_overlap: int,
    question_type_mix: Optional[Dict[str, float]],
    seed: Optional[int],
) -> List[TestCase]:
    chunks = _chunks_from_documents(documents, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("No chunk-able text found in `documents` — nothing to generate questions from.")

    mix = question_type_mix or DEFAULT_TYPE_MIX
    plan = _build_plan(n, mix, seed)
    rng = random.Random(seed)

    cases: List[TestCase] = []
    for qtype in plan:
        if qtype == "multi_context" and len(chunks) >= 2:
            src = rng.sample(chunks, 2)
            passage = f"PASSAGE 1:\n{src[0]}\n\nPASSAGE 2:\n{src[1]}"
        else:
            qtype = "simple" if qtype == "multi_context" else qtype
            src = [rng.choice(chunks)]
            passage = src[0]

        instr = _TYPE_PROMPTS[qtype]
        try:
            resp = completion(model=model, temperature=0.7, messages=[
                {"role": "system", "content":
                    f"You write evaluation questions for a RAG benchmark. {instr}\n"
                    f'Respond ONLY with JSON: {{"question": "...", "answer": "..."}}'},
                {"role": "user", "content": passage[:6000]},
            ])
            qa = _extract_qa(resp.text)
        except Exception:
            qa = None
        if qa:
            cases.append(TestCase(question=qa[0], reference_answer=qa[1],
                                  question_type=qtype, source_chunks=src))

    return cases


def generate_testset(
    documents: List[Dict[str, Any]],
    n: int = 20,
    model: Any = "openai/gpt-4o-mini",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    question_type_mix: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[Optional[str]]]:
    """
    Generate a synthetic (questions, reference_answers) pair from source
    documents, shaped exactly for evaluate()/compare()/recommend_strategy().

    Without this, comparing RAG strategies at any real scale requires
    hand-writing a large, diverse question set — the biggest friction point
    in evaluating "which strategy is actually best for my data." This
    samples chunks from your documents and has an LLM write grounded
    questions (and correct reference answers) against them, mixing simple
    single-fact questions with harder reasoning and multi-passage questions
    so the resulting benchmark isn't trivially easy for every strategy.

    Args:
        documents: same shape as evaluate()'s `documents` — [{"text": ...}].
        n: how many questions to generate.
        model: LLM that writes the questions+answers — any 'provider/name'
            string, or a bring-your-own-model object (LangChain/callable),
            passed straight through to completion().
        question_type_mix: fraction of each type to generate, default
            {"simple": 0.5, "reasoning": 0.3, "multi_context": 0.2}.
        seed: for reproducible chunk sampling (the LLM call itself is still
            non-deterministic unless the model/provider guarantees it).

    Returns:
        (questions, reference_answers) ready for
        ``evaluate(questions=questions, reference_answers=references, ...)``.
    """
    cases = _generate_cases(documents, n, model, chunk_size, chunk_overlap,
                            question_type_mix, seed)
    return [c.question for c in cases], [c.reference_answer for c in cases]


def generate_testset_detailed(
    documents: List[Dict[str, Any]],
    n: int = 20,
    model: Any = "openai/gpt-4o-mini",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    question_type_mix: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
) -> List[TestCase]:
    """Like generate_testset(), but returns full TestCase objects (with
    question_type and source_chunks) instead of flat (questions, references)
    tuples — useful for inspecting, filtering, or saving the generated set
    before running an evaluation."""
    return _generate_cases(documents, n, model, chunk_size, chunk_overlap,
                           question_type_mix, seed)
