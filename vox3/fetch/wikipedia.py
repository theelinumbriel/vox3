"""Wikipedia via the MediaWiki API."""
from __future__ import annotations

import re
from pathlib import Path

import requests

UA = {"User-Agent": "vox3/0.1 (research)"}
API = "https://en.wikipedia.org/w/api.php"


def _search(query: str, limit: int) -> list[str]:
    r = requests.get(
        API,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        },
        headers=UA,
        timeout=30,
    )
    r.raise_for_status()
    return [hit["title"] for hit in r.json().get("query", {}).get("search", [])]


def _extract(title: str) -> str:
    r = requests.get(
        API,
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": title,
            "format": "json",
            "redirects": 1,
        },
        headers=UA,
        timeout=30,
    )
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    for p in pages.values():
        return p.get("extract", "") or ""
    return ""


def fetch(query: str, max_items: int, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    titles = _search(query, max_items)
    paths: list[Path] = []
    for title in titles:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower())[:60].strip("-")
        dest = out_dir / f"wikipedia-{slug}.txt"
        if dest.exists():
            paths.append(dest)
            continue
        try:
            text = _extract(title)
            if len(text) < 500:
                continue
            dest.write_text(text, encoding="utf-8")
            paths.append(dest)
        except Exception as e:
            print(f"  wikipedia: skip {title}: {e}")
    return paths
