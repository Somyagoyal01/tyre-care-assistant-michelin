"""Answer generators behind one interface: offline ScriptedLLM, optional real LLM via env var."""
from __future__ import annotations

import logging
import os
from typing import Protocol

from .retriever import Doc, Retriever, split_sentences

log = logging.getLogger("assistant.llm")


class AnswerGenerator(Protocol):
    def generate(self, question: str, docs: list[Doc]) -> str: ...


class ScriptedLLM:
    """Extractive 'LLM': picks the most query-relevant sentences from the retrieved docs.

    Deterministic and offline, so tests and evals are reproducible. It cannot paraphrase
    or reason across documents, which is the main limitation versus a real model.
    """

    def __init__(self, retriever: Retriever, max_sentences: int = 3):
        self.retriever = retriever
        self.max_sentences = max_sentences

    def generate(self, question: str, docs: list[Doc]) -> str:
        cands = [(s, d.doc_id) for d in docs for s in split_sentences(d.text)]
        scores = self.retriever.sentence_scores(question, [s for s, _ in cands])
        ranked = sorted(zip(scores, range(len(cands))), reverse=True)[: self.max_sentences]
        picked = [cands[i][0] for _, i in sorted(ranked, key=lambda t: t[1])]  # keep document order
        return " ".join(picked)


class OpenAICompatibleLLM:
    """Real-LLM path (not used in tests). Set ASSISTANT_LLM=openai and OPENAI_API_KEY.

    Any OpenAI-compatible endpoint works (set OPENAI_BASE_URL, e.g. a Gemini-compatible proxy).
    """

    def __init__(self, model: str | None = None):
        from openai import OpenAI  # imported lazily so offline mode needs no extra package

        self.client = OpenAI()
        self.model = model or os.getenv("ASSISTANT_MODEL", "gpt-4o-mini")

    def generate(self, question: str, docs: list[Doc]) -> str:
        context = "\n\n".join(f"[{d.doc_id}] {d.text}" for d in docs)
        prompt = ("Answer the tyre-care question using ONLY the context. If the context does not "
                  "contain the answer, say so. Never say a damaged tyre is safe to drive.\n\n"
                  f"Context:\n{context}\n\nQuestion: {question}")
        resp = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0)
        return resp.choices[0].message.content.strip()


def get_generator(retriever: Retriever) -> AnswerGenerator:
    if os.getenv("ASSISTANT_LLM", "scripted").lower() == "openai":
        try:
            return OpenAICompatibleLLM()
        except Exception as exc:  # missing package or key -> stay usable offline
            log.warning("real LLM unavailable (%s); falling back to ScriptedLLM", exc)
    return ScriptedLLM(retriever)
