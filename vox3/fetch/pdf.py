"""Local PDF ingestion via PyMuPDF."""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf


def ingest(path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower())[:60].strip("-")
    dest = out_dir / f"pdf-{slug}.txt"
    chunks: list[str] = []
    with pymupdf.open(path) as doc:
        for page in doc:
            chunks.append(page.get_text())
    dest.write_text("\n".join(chunks), encoding="utf-8")
    return [dest]
