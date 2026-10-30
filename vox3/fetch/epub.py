"""Local EPUB ingestion via ebooklib + BeautifulSoup."""
from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT


def ingest(path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower())[:60].strip("-")
    dest = out_dir / f"epub-{slug}.txt"
    book = epub.read_epub(str(path))
    chunks: list[str] = []
    for item in book.get_items():
        if item.get_type() != ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")
        chunks.append(soup.get_text("\n"))
    dest.write_text("\n\n".join(chunks), encoding="utf-8")
    return [dest]
