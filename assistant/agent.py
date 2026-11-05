"""Agent graph (LangGraph) wiring the router to the tool, RAG and refusal nodes, with logging."""
from __future__ import annotations

import re

import json
import logging
import time
from typing import Optional, TypedDict

from .llm import get_generator
from .retriever import Retriever
from .router import classify
from .tyre_size import (SIZE_RE, TyreSizeError, catalogue_matches, check_fitment, find_service_descriptions,
                        find_sizes)

log = logging.getLogger("assistant")

MIN_RETRIEVAL_SCORE = 0.08  # below this, the KB has nothing relevant -> refuse (tuned on dev set only)

SAFETY_MSG = ("This sounds safety-critical. Please do not drive on this tyre. Fit the spare or call "
              "roadside assistance, and have a qualified tyre professional inspect it before any further use. "
              "I can't confirm a damaged tyre is safe remotely.")
OOS_MSG = ("Sorry, I can only help with general tyre care and tyre size/fitment questions. For prices, "
           "medical, legal or other topics please contact the relevant professional or your dealer.")


class State(TypedDict, total=False):
    question: str
    route: str
    answer: str
    doc_ids: list[str]
    tool_result: Optional[dict]


class TyreAssistant:
    def __init__(self, retriever: Retriever | None = None):
        self.retriever = retriever or Retriever()
        self.generator = get_generator(self.retriever)
        self.graph = self._build_graph()

    # ---- nodes ----
    def _classify(self, s: State) -> State:
        return {"route": classify(s["question"])}

    def _rag(self, s: State) -> State:
        hits = self.retriever.search(s["question"], k=3)
        if hits[0][1] < MIN_RETRIEVAL_SCORE:
            return {"route": "out_of_scope", "answer": OOS_MSG, "doc_ids": []}
        docs = [d for d, score in hits[:2] if score >= MIN_RETRIEVAL_SCORE]
        ids = [d.doc_id for d in docs]
        text = self.generator.generate(s["question"], docs)
        return {"answer": f"{text}\n\nSources: {', '.join(ids)}", "doc_ids": ids}

    def _safety(self, s: State) -> State:
        ids = [d.doc_id for d, score in self.retriever.search(s["question"], k=2) if score > 0]
        return {"answer": f"{SAFETY_MSG}\n\nSources: {', '.join(ids)}", "doc_ids": ids}

    def _refuse(self, s: State) -> State:
        return {"answer": OOS_MSG, "doc_ids": []}

    def _tool(self, s: State) -> State:
        q = s["question"]
        try:
            sizes = find_sizes(q)
        except TyreSizeError:
            sizes = []
        services = find_service_descriptions(q)
        docs = ["size_marking", "load_index", "speed_rating"]
        if not sizes:
            if services:  # e.g. "what does 91V mean"
                load, speed = services[0]
                from .tyre_size import LOAD_INDEX_KG, SPEED_KMH
                ans = (f"{load}{speed}: load index {load} = {LOAD_INDEX_KG.get(load)} kg per tyre; "
                       f"speed rating {speed} = {SPEED_KMH.get(speed)} km/h.")
                return {"answer": f"{ans}\n\nSources: load_index, speed_rating", "doc_ids": docs[1:],
                        "tool_result": {"action": "decode"}}
            return self._rag(s) | {"route": "rag"}
        cand, req_size = _assign_roles(q, sizes)
        req_load = req_size.load_index if req_size else (services[0][0] if services else None)
        req_speed = req_size.speed_symbol if req_size else (services[0][1] if services else None)
        ql = q.lower()
        if req_size or services:
            res = check_fitment(cand, req_load, req_speed, req_size)
            target = req_size.label if req_size else f"{req_load}{req_speed}"
            verdict = "Yes, it is a suitable match" if res.fits else "No, it is not a suitable match"
            ans = (f"{verdict}: {cand.label} vs required {target}. Reasons: " + "; ".join(res.reasons) +
                   ". Always confirm the size is approved in your vehicle handbook.")
            tr = {"action": "fitment", "fits": res.fits}
        elif any(w in ql for w in ("catalog", "catalogue", "stock", "available", "options", "which tyres")):
            rows = catalogue_matches(cand)
            listing = ", ".join(f"{r['sku']} ({r['size']}, {r['season']})" for r in rows) or "none"
            ans = f"Synthetic catalogue options matching {cand.label} or better: {listing}."
            tr = {"action": "catalogue", "skus": [r["sku"] for r in rows]}
            docs = ["load_index", "speed_rating"]
        else:
            ans = cand.describe()
            tr = {"action": "decode", "load_kg": cand.load_kg, "speed_kmh": cand.speed_kmh}
        return {"answer": f"{ans}\n\nSources: {', '.join(docs)}", "doc_ids": docs, "tool_result": tr}

    # ---- graph ----
    def _build_graph(self):
        nodes = {"tool": self._tool, "rag": self._rag, "safety": self._safety, "out_of_scope": self._refuse}
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:  # hand-written fallback with identical behaviour
            log.info("langgraph not installed; using hand-written router")
            return _FallbackGraph(self._classify, nodes)
        g = StateGraph(State)
        g.add_node("classify", self._classify)
        for name, fn in nodes.items():
            g.add_node(name, fn)
            g.add_edge(name, END)
        g.set_entry_point("classify")
        g.add_conditional_edges("classify", lambda s: s["route"], {n: n for n in nodes})
        return g.compile()

    def ask(self, question: str) -> dict:
        t0 = time.perf_counter()
        question = (question or "").strip()
        if not question:
            return {"route": "error", "answer": "Please ask a question.", "doc_ids": [], "latency_ms": 0.0}
        try:
            out = self.graph.invoke({"question": question})
            result = {"route": out["route"], "answer": out["answer"], "doc_ids": out.get("doc_ids", []),
                      "tool_result": out.get("tool_result")}
        except Exception as exc:  # never crash the chat; log and return a safe message
            log.exception("ask failed")
            result = {"route": "error", "answer": f"Sorry, something went wrong ({type(exc).__name__}).",
                      "doc_ids": [], "tool_result": None}
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        log.info(json.dumps({"event": "ask", "q": question[:80], "route": result["route"],
                             "docs": result["doc_ids"], "latency_ms": result["latency_ms"]}))
        return result


class _FallbackGraph:
    def __init__(self, classify_fn, nodes):
        self.classify_fn, self.nodes = classify_fn, nodes

    def invoke(self, state: dict) -> dict:
        state = {**state, **self.classify_fn(state)}
        return {**state, **self.nodes[state["route"]](state)}

_REQ_CUE = re.compile(r"(manual|handbook|placard|needs?|requires?|required|recommended|specified|original|instead of|fitted|came with)\W*(\w+\W+){0,4}$", re.I)
_CAND_CUE = re.compile(r"(offer(?:s|ing)?|shop|dealer|put|fit|buy|got|selling)\W*(\w+\W+){0,4}$", re.I)


def _assign_roles(q: str, sizes):
    """Decide which mentioned size is the candidate and which is the required one.

    v1.1 fix (post-freeze, see README): v1.0 always treated the first size as the
    candidate, which reversed the check for "manual says X, shop offers Y".
    """
    if len(sizes) < 2:
        return sizes[0], None
    spans = [m.start() for m in SIZE_RE.finditer(q)][:2]
    roles = []
    for start in spans:
        before = q[max(0, start - 40):start]
        roles.append("req" if _REQ_CUE.search(before) else "cand" if _CAND_CUE.search(before) else None)
    if roles[0] == "req" or roles[1] == "cand":
        return sizes[1], sizes[0]
    return sizes[0], sizes[1]

