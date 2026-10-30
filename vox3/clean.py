"""Concatenate raw text files into corpus.txt, dedupe, normalize."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

WS = re.compile(r"[ \t]+")
BLANKS = re.compile(r"\n{3,}")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(c for c in text if c == "\n" or c == "\t" or ord(c) >= 0x20)
    text = WS.sub(" ", text)
    text = BLANKS.sub("\n\n", text)
    return text.strip()


def build_corpus(raw_paths: list[Path], out_path: Path) -> int:
    seen: set[str] = set()
    buf: list[str] = []
    for p in raw_paths:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        txt = _normalize(txt)
        for line in txt.split("\n"):
            key = line.strip().lower()
            if not key or len(key) < 3:
                buf.append(line)
                continue
            if key in seen:
                continue
            seen.add(key)
            buf.append(line)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(buf), encoding="utf-8")
    return out_path.stat().st_size
