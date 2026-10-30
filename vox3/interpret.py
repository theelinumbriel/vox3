"""Turn a natural-language prompt into a structured corpus-fetch spec via Claude."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import anthropic

MODEL = "claude-sonnet-4-5"

SYSTEM = """You plan corpora for training a tiny character-level language model that will output "stylistically-flavored slop" in the style the user describes.

You decide which public text sources to pull from, and emit search queries for each. Your output must be a single JSON object matching this schema:

{
  "topic_summary": str,              // one-sentence restatement
  "target_mb": float,                // 1-10, how much text to aim for
  "sources": [
    {
      "type": "gutenberg" | "wikipedia" | "archive_org" | "youtube",
      "query": str,                  // search term for that source
      "max_items": int               // 1-30
    }
  ],
  "notes": str                       // what the model should sound like
}

Guidance:
- Lean on gutenberg for pre-1928 authors; wikipedia for genre/era pages; archive_org for ephemera (zines, pulps, magazines); youtube for interviews, lectures, modern voices.
- 3-6 sources total. Default target_mb=2.
- If the user names a specific living author (still copyrighted), use wikipedia + youtube rather than gutenberg.
- Return ONLY the JSON object, no prose."""


@dataclass
class SourceSpec:
    type: str
    query: str
    max_items: int = 10


@dataclass
class CorpusSpec:
    topic_summary: str
    target_mb: float
    sources: list[SourceSpec]
    notes: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def interpret(prompt: str) -> CorpusSpec:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text)
    return CorpusSpec(
        topic_summary=data["topic_summary"],
        target_mb=float(data.get("target_mb", 2.0)),
        sources=[SourceSpec(**s) for s in data["sources"]],
        notes=data.get("notes", ""),
        raw=data,
    )
