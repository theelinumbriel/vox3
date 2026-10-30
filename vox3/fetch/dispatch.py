"""Route a CorpusSpec to the right fetcher modules."""
from __future__ import annotations

from pathlib import Path

from ..interpret import CorpusSpec
from . import archive_org, gutenberg, wikipedia, youtube

FETCHERS = {
    "gutenberg": gutenberg.fetch,
    "wikipedia": wikipedia.fetch,
    "archive_org": archive_org.fetch,
    "youtube": youtube.fetch,
}


def run(spec: CorpusSpec, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_paths: list[Path] = []
    for src in spec.sources:
        fn = FETCHERS.get(src.type)
        if not fn:
            print(f"  unknown source type: {src.type}")
            continue
        print(f"  fetching {src.type} '{src.query}' (max {src.max_items})")
        try:
            paths = fn(src.query, src.max_items, out_dir)
            print(f"    → {len(paths)} files")
            all_paths.extend(paths)
        except Exception as e:
            print(f"    ! {src.type} failed: {e}")
    return all_paths
