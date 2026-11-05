"""Tyre size parsing, decoding and fitment checks (the structured-data 'tool' path)."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "tyre_specs.csv"

# Subset of the standard load-index table (index -> max load per tyre, kg).
LOAD_INDEX_KG = {
    75: 387, 76: 400, 77: 412, 78: 425, 79: 437, 80: 450, 81: 462, 82: 475,
    83: 487, 84: 500, 85: 515, 86: 530, 87: 545, 88: 560, 89: 580, 90: 600,
    91: 615, 92: 630, 93: 650, 94: 670, 95: 690, 96: 710, 97: 730, 98: 750,
    99: 775, 100: 800, 101: 825, 102: 850, 103: 875, 104: 900, 105: 925,
    106: 950, 107: 975, 108: 1000, 109: 1030, 110: 1060,
}
# Speed symbol -> max sustained speed (km/h). Note H sits between U and V.
SPEED_KMH = {
    "L": 120, "M": 130, "N": 140, "P": 150, "Q": 160, "R": 170, "S": 180,
    "T": 190, "U": 200, "H": 210, "V": 240, "W": 270, "Y": 300,
}

SIZE_RE = re.compile(
    r"(?P<width>\d{3})\s*/\s*(?P<aspect>\d{2})\s*(?P<cons>Z?R|D|B|-)?\s*(?P<rim>\d{2})"
    r"(?:\s*(?P<load>\d{2,3})\s*(?P<speed>[A-Za-z])\b)?",
    re.IGNORECASE,
)
SERVICE_RE = re.compile(r"\b(?P<load>\d{2,3})\s?(?P<speed>[LMNPQRSTUHVWY])\b", re.IGNORECASE)


class TyreSizeError(ValueError):
    """Raised when a tyre size string cannot be parsed."""


@dataclass
class TyreSize:
    width_mm: int
    aspect_ratio: int
    construction: str
    rim_in: int
    load_index: Optional[int] = None
    speed_symbol: Optional[str] = None

    @property
    def load_kg(self) -> Optional[int]:
        return LOAD_INDEX_KG.get(self.load_index) if self.load_index else None

    @property
    def speed_kmh(self) -> Optional[int]:
        return SPEED_KMH.get(self.speed_symbol) if self.speed_symbol else None

    @property
    def overall_diameter_mm(self) -> float:
        return round(self.rim_in * 25.4 + 2 * self.width_mm * self.aspect_ratio / 100, 1)

    @property
    def label(self) -> str:
        sd = f" {self.load_index}{self.speed_symbol}" if self.load_index else ""
        return f"{self.width_mm}/{self.aspect_ratio} {self.construction}{self.rim_in}{sd}"

    def describe(self) -> str:
        cons = {"R": "radial", "ZR": "radial (ZR, high-speed)", "D": "diagonal/cross-ply",
                "B": "bias-belted", "-": "cross-ply"}.get(self.construction, self.construction)
        parts = [f"{self.label}: width {self.width_mm} mm, aspect ratio {self.aspect_ratio}% "
                 f"(sidewall about {round(self.width_mm * self.aspect_ratio / 100)} mm), {cons} construction, "
                 f"fits a {self.rim_in}-inch rim, overall diameter about {self.overall_diameter_mm:.0f} mm"]
        if self.load_index:
            kg = f"{self.load_kg} kg" if self.load_kg else "not in this demo's load table"
            parts.append(f"load index {self.load_index} = {kg} per tyre")
        if self.speed_symbol:
            kmh = f"{self.speed_kmh} km/h" if self.speed_kmh else "unknown symbol"
            parts.append(f"speed rating {self.speed_symbol} = {kmh}")
        return "; ".join(parts) + "."


def parse_size(text: str) -> TyreSize:
    m = SIZE_RE.search(text)
    if not m:
        raise TyreSizeError(f"No tyre size found in: {text!r}")
    cons = (m.group("cons") or "R").upper()
    load = int(m.group("load")) if m.group("load") else None
    speed = m.group("speed").upper() if m.group("speed") else None
    if speed and speed not in SPEED_KMH:
        load, speed = None, None  # e.g. trailing word, not a speed symbol
    return TyreSize(int(m.group("width")), int(m.group("aspect")), cons, int(m.group("rim")), load, speed)


def find_sizes(text: str) -> list[TyreSize]:
    return [parse_size(m.group(0)) for m in SIZE_RE.finditer(text)]


def find_service_descriptions(text: str) -> list[tuple[int, str]]:
    """Standalone service descriptions like '91V' that are not part of a full size."""
    stripped = SIZE_RE.sub(" ", text)
    return [(int(m.group("load")), m.group("speed").upper()) for m in SERVICE_RE.finditer(stripped)
            if int(m.group("load")) in LOAD_INDEX_KG]


@dataclass
class FitmentResult:
    fits: bool
    reasons: list[str]


def check_fitment(candidate: TyreSize, req_load: Optional[int], req_speed: Optional[str],
                  required_size: Optional[TyreSize] = None) -> FitmentResult:
    reasons, fits = [], True
    if req_load is not None:
        if candidate.load_index is None:
            fits = False
            reasons.append("the candidate tyre has no load index given, so load capacity cannot be confirmed")
        elif candidate.load_index >= req_load:
            reasons.append(f"load index {candidate.load_index} ({candidate.load_kg} kg) meets or exceeds the "
                           f"required {req_load} ({LOAD_INDEX_KG.get(req_load)} kg)")
        else:
            fits = False
            reasons.append(f"load index {candidate.load_index} ({candidate.load_kg} kg) is LOWER than the "
                           f"required {req_load} ({LOAD_INDEX_KG.get(req_load)} kg)")
    if req_speed is not None:
        cs, rs = candidate.speed_kmh, SPEED_KMH.get(req_speed)
        if cs is None:
            fits = False
            reasons.append("the candidate tyre has no speed rating given")
        elif cs >= rs:
            reasons.append(f"speed rating {candidate.speed_symbol} ({cs} km/h) meets or exceeds the "
                           f"required {req_speed} ({rs} km/h)")
        else:
            fits = False
            reasons.append(f"speed rating {candidate.speed_symbol} ({cs} km/h) is LOWER than the "
                           f"required {req_speed} ({rs} km/h)")
    if required_size is not None:
        if candidate.rim_in != required_size.rim_in:
            fits = False
            reasons.append(f"rim diameter differs ({candidate.rim_in} in vs {required_size.rim_in} in), "
                           "so it needs different wheels and must be an approved option for the vehicle")
        diff = (candidate.overall_diameter_mm - required_size.overall_diameter_mm) / required_size.overall_diameter_mm
        if abs(diff) > 0.03:
            fits = False
            reasons.append(f"overall diameter differs by {diff:+.1%}, outside the usual 3% limit")
        else:
            reasons.append(f"overall diameter differs by {diff:+.1%}, within the usual 3% limit")
    return FitmentResult(fits, reasons)


def load_catalogue(path: Path = DATA_CSV) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["parsed"] = parse_size(r["size"])
    return rows


def catalogue_matches(required: TyreSize, path: Path = DATA_CSV) -> list[dict]:
    """Synthetic catalogue rows with the same dimensions and an equal-or-better service description."""
    out = []
    for r in load_catalogue(path):
        p = r["parsed"]
        if (p.width_mm, p.aspect_ratio, p.rim_in) != (required.width_mm, required.aspect_ratio, required.rim_in):
            continue
        if check_fitment(p, required.load_index, required.speed_symbol).fits:
            out.append(r)
    return out
