"""CPU-only helpers for Streamlit text offsets and entity display."""
from __future__ import annotations

import html
import re
from src.data.dataset import repair_iob2, spans

COLORS = {"PER": "#ffe1e6", "ORG": "#dcecff", "LOC": "#daf4df", "MISC": "#f9ebce"}


def tokenize_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    matches = list(re.finditer(r"\w+|[^\w\s]", text))
    return [m.group() for m in matches], [m.span() for m in matches]


def entities(tokens: list[str], offsets: list[tuple[int, int]], tags: list[str], text: str) -> list[dict]:
    if len(tokens) != len(offsets) or len(tags) != len(tokens):
        raise ValueError("Token, offset, and tag lengths must match")
    return [{"text": text[offsets[a][0]:offsets[b - 1][1]], "type": kind,
             "start": offsets[a][0], "end": offsets[b - 1][1]}
            for a, b, kind in spans(repair_iob2(tags))]


def highlight(text: str, found: list[dict]) -> str:
    parts = []
    position = 0
    for entity in sorted(found, key=lambda e: e["start"]):
        if entity["start"] < position or entity["end"] > len(text):
            raise ValueError("Overlapping or out-of-range entity offsets")
        parts.append(html.escape(text[position:entity["start"]]))
        color = COLORS[entity["type"]]
        parts.append('<span style="background:' + color + ';padding:2px;border-radius:3px">'
                     + html.escape(text[entity["start"]:entity["end"]])
                     + ' <small>' + html.escape(entity["type"]) + '</small></span>')
        position = entity["end"]
    parts.append(html.escape(text[position:]))
    return '<div style="line-height:2">' + "".join(parts) + '</div>'

