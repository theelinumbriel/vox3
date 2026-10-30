"""archive.org texts via advancedsearch + djvu.txt download."""
from __future__ import annotations

import re
from pathlib import Path

import requests

UA = {"User-Agent": "vox3/0.1 (research)"}
SEARCH = "https://archive.org/advancedsearch.php"


def _search(query: str, rows: int) -> list[str]:
    r = requests.get(
        SEARCH,
        params={
            "q": f"({query}) AND mediatype:(texts) AND format:(DJVUTXT)",
            "fl[]": "identifier",
            "rows": rows,
            "output": "json",
        },
        headers=UA,
        timeout=30,
    )
    r.raise_for_status()
    return [d["identifier"] for d in r.json().get("response", {}).get("docs", [])]


def _download(identifier: str) -> str:
    url = f"https://archive.org/download/{identifier}/{identifier}_djvu.txt"
    r = requests.get(url, headers=UA, timeout=90)
    r.raise_for_status()
    return r.text


def fetch(query: str, max_items: int, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    identifiers = _search(query, max_items)
    paths: list[Path] = []
    for ident in identifiers:
        slug = re.sub(r"[^a-z0-9]+", "-", ident.lower())[:80].strip("-")
        dest = out_dir / f"archive-{slug}.txt"
        if dest.exists():
            paths.append(dest)
            continue
        try:
            text = _download(ident)
            if len(text) < 500:
                continue
            dest.write_text(text, encoding="utf-8")
            paths.append(dest)
        except Exception as e:
            print(f"  archive_org: skip {ident}: {e}")
    return paths
