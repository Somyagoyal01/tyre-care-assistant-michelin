"""TF-IDF retriever over the markdown knowledge base (offline, no API keys)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"


@dataclass
class Doc:
    doc_id: str
    title: str
    text: str


def load_kb(kb_dir: Path = KB_DIR) -> list[Doc]:
    docs = []
    for p in sorted(kb_dir.glob("*.md")):
        lines = p.read_text(encoding="utf-8").splitlines()
        title = lines[0].lstrip("# ").strip()
        body = " ".join(l for l in lines[1:] if l.strip() and not l.startswith(">"))
        docs.append(Doc(p.stem, title, body))
    if not docs:
        raise FileNotFoundError(f"No knowledge-base docs in {kb_dir}")
    return docs


_SUFFIXES = (("ations", "at"), ("ation", "at"), ("ions", ""), ("ion", ""), ("ing", ""), ("age", ""),
             ("ers", ""), ("er", ""), ("ed", ""), ("es", ""), ("s", ""), ("e", ""))


def stem(tok: str) -> str:
    """Crude suffix stripper so 'rotate'/'rotation'/'rotating' share a stem (no NLTK dependency)."""
    for suf, repl in _SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            return tok[: -len(suf)] + repl
    return tok


def analyze(text: str) -> list[str]:
    toks = [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in ENGLISH_STOP_WORDS]
    return toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class Retriever:
    def __init__(self, docs: list[Doc] | None = None):
        self.docs = docs or load_kb()
        # Title is repeated to give it extra weight; stemmed word 1-2 grams, stop words removed.
        corpus = [f"{d.title} {d.title} {d.text}" for d in self.docs]
        self.vec = TfidfVectorizer(analyzer=analyze, sublinear_tf=True)
        self.matrix = self.vec.fit_transform(corpus)

    def search(self, query: str, k: int = 3) -> list[tuple[Doc, float]]:
        scores = linear_kernel(self.vec.transform([query]), self.matrix).ravel()
        order = scores.argsort()[::-1][:k]
        return [(self.docs[i], float(scores[i])) for i in order]

    def sentence_scores(self, query: str, sentences: list[str]) -> list[float]:
        if not sentences:
            return []
        return linear_kernel(self.vec.transform([query]), self.vec.transform(sentences)).ravel().tolist()
