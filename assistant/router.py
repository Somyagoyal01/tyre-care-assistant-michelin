"""Rule-based intent classifier: tool / rag / safety (escalate) / out_of_scope (refuse)."""
from __future__ import annotations

import re

from .tyre_size import SIZE_RE, find_service_descriptions

# Safety-critical: visible damage or "can I keep driving" on a compromised tyre.
SAFETY_PATTERNS = [
    r"\bbulg\w*", r"\bbubble\w*", r"\b(cords?|wires?|steel)\b.{0,25}\b(show\w*|visible|expos\w*|out)\b",
    r"\b(show\w*|visible|expos\w*)\b.{0,25}\b(cords?|wires?|steel)\b",
    r"\bsidewall\b.{0,30}\b(cut|crack\w*|damage\w*|tear|torn|gash)\b",
    r"\b(cut|crack\w*|damage\w*|tear|torn|gash)\b.{0,30}\bsidewall\b",
    r"\bblow\s?out\b", r"\btread separat\w*", r"\b(bald|no tread)\b",
    r"\b(drive|driving|keep going)\b.{0,40}\b(flat|nail|screw|puncture\w*)\b",
    r"\b(flat|nail|screw|puncture\w*)\b.{0,40}\b(drive|driving|keep going)\b",
]
# Out of scope: pricing, medical, legal, and unrelated topics.
OOS_PATTERNS = [
    r"\b(price|prices|cost|costs|cheap\w*|discount\w*|offer|rupees|rs\.?|inr|emi)\b",
    r"\b(injur\w*|doctor|hospital|medic\w*|pain)\b",
    r"\b(lawyer|sue|legal action|court|insurance claim|compensation|warranty claim)\b",
    r"\b(weather|recipe|cricket|movie|stock market|poem|joke|bitcoin|election)\b",
]


def classify(question: str) -> str:
    q = question.lower()
    if any(re.search(p, q) for p in SAFETY_PATTERNS):
        return "safety"
    if any(re.search(p, q) for p in OOS_PATTERNS):
        return "out_of_scope"
    if SIZE_RE.search(question) or find_service_descriptions(question):
        return "tool"
    return "rag"
