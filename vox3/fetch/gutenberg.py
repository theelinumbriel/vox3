"""Project Gutenberg via gutendex.com."""
from __future__ import annotations

import re
from pathlib import Path

import requests

UA = {"User-Agent": "vox3/0.1 (research)"}
GUTENDEX = "https://gutendex.com/books"

HEADER_RE = re.compile(r"\*\*\*\s*START OF .*?\*\*\*", re.IGNORECASE | re.DOTALL)
FOOTER_RE = re.compile(r"\*\*\*\s*END OF .*?\*\*\*", re.IGNORECASE | re.DOTALL)


def _strip_gutenberg(text: str) -> str:
    m = HEADER_RE.search(text)
    if m:
        text = text[m.end():]
    m = FOOTER_RE.search(text)
    if m:
        text = text[:m.start()]
    return text.strip()


def _pick_text_url(formats: dict) -> str | None:
    for key in ("text/plain; charset=utf-8", "text/plain; charset=us-ascii", "text/plain"):
        if key in formats and not formats[key].endswith(".zip"):
            return formats[key]
    for k, v in formats.items():
        if k.startswith("text/plain") and not v.endswith(".zip"):
            return v
    return None


def fetch(query: str, max_items: int, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = requests.get(GUTENDEX, params={"search": query, "languages": "en"}, headers=UA, timeout=30)
    r.raise_for_status()
    books = r.json().get("results", [])[:max_items]
    paths: list[Path] = []
    for b in books:
        url = _pick_text_url(b.get("formats", {}))
        if not url:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", b["title"].lower())[:60].strip("-")
        dest = out_dir / f"gutenberg-{b['id']}-{slug}.txt"
        if dest.exists():
            paths.append(dest)
            continue
        try:
            resp = requests.get(url, headers=UA, timeout=60)
            resp.raise_for_status()
            dest.write_text(_strip_gutenberg(resp.text), encoding="utf-8")
            paths.append(dest)
        except Exception as e:
            print(f"  gutenberg: skip {b['id']}: {e}")
    return paths
