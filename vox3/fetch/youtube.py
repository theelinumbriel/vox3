"""YouTube transcripts via yt-dlp (search + auto-subtitle extraction)."""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _vtt_to_text(vtt: str) -> str:
    lines = []
    seen = set()
    for raw in vtt.splitlines():
        line = raw.strip()
        if not line or line.startswith("WEBVTT") or line.startswith("NOTE"):
            continue
        if "-->" in line:
            continue
        if re.fullmatch(r"\d+", line):
            continue
        clean = re.sub(r"<[^>]+>", "", line)
        clean = re.sub(r"\s+", " ", clean).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        lines.append(clean)
    return "\n".join(lines)


def _search_ids(query: str, n: int) -> list[tuple[str, str]]:
    res = _run([
        "yt-dlp",
        f"ytsearch{n}:{query}",
        "--flat-playlist",
        "--dump-json",
        "--skip-download",
    ], timeout=120)
    out: list[tuple[str, str]] = []
    for line in res.stdout.splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        vid = obj.get("id")
        title = obj.get("title") or vid
        if vid:
            out.append((vid, title))
    return out


def fetch(query: str, max_items: int, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        ids = _search_ids(query, max_items)
    except Exception as e:
        print(f"  youtube: search failed: {e}")
        return []
    paths: list[Path] = []
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for vid, title in ids:
            slug = re.sub(r"[^a-z0-9]+", "-", title.lower())[:60].strip("-")
            dest = out_dir / f"youtube-{vid}-{slug}.txt"
            if dest.exists():
                paths.append(dest)
                continue
            try:
                _run([
                    "yt-dlp",
                    f"https://youtu.be/{vid}",
                    "--skip-download",
                    "--write-auto-sub",
                    "--write-sub",
                    "--sub-lang", "en.*",
                    "--sub-format", "vtt",
                    "-o", str(tdp / f"{vid}.%(ext)s"),
                ], timeout=120)
                vtts = list(tdp.glob(f"{vid}*.vtt"))
                if not vtts:
                    continue
                text = _vtt_to_text(vtts[0].read_text(encoding="utf-8", errors="ignore"))
                if len(text) < 300:
                    continue
                dest.write_text(text, encoding="utf-8")
                paths.append(dest)
            except Exception as e:
                print(f"  youtube: skip {vid}: {e}")
    return paths
