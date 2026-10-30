"""vox3 CLI: prompt → corpus → tiny model → slop."""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .clean import build_corpus
from .fetch import dispatch
from .fetch.epub import ingest as ingest_epub
from .fetch.pdf import ingest as ingest_pdf
from .interpret import CorpusSpec, SourceSpec, interpret
from .sample import sample
from .train import train


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "run"


def _run_dir(root: Path, label: str) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    p = root / f"{ts}-{_slugify(label)}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _spec_from_file(path: Path) -> CorpusSpec:
    return CorpusSpec(
        topic_summary=f"local file: {path.name}",
        target_mb=0.5,
        sources=[],
        notes="",
        raw={"local_file": str(path)},
    )


def _save_spec(spec: CorpusSpec, run: Path) -> None:
    data = {
        "topic_summary": spec.topic_summary,
        "target_mb": spec.target_mb,
        "notes": spec.notes,
        "sources": [asdict(s) for s in spec.sources],
        "raw": spec.raw,
    }
    (run / "spec.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))


def cmd_run(args) -> int:
    runs_root = Path(args.runs_dir).expanduser().resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    if args.file:
        src_path = Path(args.file).expanduser().resolve()
        if not src_path.exists():
            print(f"file not found: {src_path}")
            return 2
        spec = _spec_from_file(src_path)
        run = _run_dir(runs_root, src_path.stem)
        print(f"[run] {run.name}  (local file: {src_path.name})")
        _save_spec(spec, run)
        raw_dir = run / "raw"
        if src_path.suffix.lower() == ".pdf":
            paths = ingest_pdf(src_path, raw_dir)
        elif src_path.suffix.lower() == ".epub":
            paths = ingest_epub(src_path, raw_dir)
        elif src_path.suffix.lower() in {".txt", ".md"}:
            raw_dir.mkdir(parents=True, exist_ok=True)
            dest = raw_dir / f"txt-{src_path.stem}.txt"
            dest.write_text(src_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
            paths = [dest]
        else:
            print(f"unsupported file type: {src_path.suffix}")
            return 2
    else:
        prompt = args.prompt
        print(f"[interpret] {prompt!r}")
        spec = interpret(prompt)
        run = _run_dir(runs_root, spec.topic_summary)
        print(f"[run] {run.name}")
        print(f"  summary: {spec.topic_summary}")
        print(f"  notes:   {spec.notes}")
        print(f"  target:  ~{spec.target_mb} MB from {len(spec.sources)} sources")
        _save_spec(spec, run)
        print("[fetch]")
        paths = dispatch.run(spec, run / "raw")

    if not paths:
        print("no text fetched; aborting")
        return 1

    print(f"[clean] {len(paths)} files")
    corpus = run / "corpus.txt"
    size = build_corpus(paths, corpus)
    print(f"  corpus.txt: {size:,} bytes")

    print("[train]")
    ckpt = train(
        corpus,
        run,
        iters=args.iters,
        block_size=args.block_size,
        batch_size=args.batch_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )

    print("[sample]")
    out = run / "slop.txt"
    sample(ckpt, out, length=args.length, temperature=args.temperature, top_k=args.top_k)
    print(f"\n{'=' * 60}\n{out.read_text()[:1200]}\n{'=' * 60}")
    print(f"\nrun dir: {run}")
    return 0


def cmd_sample(args) -> int:
    ckpt = Path(args.checkpoint).expanduser().resolve()
    out = Path(args.out).expanduser().resolve() if args.out else ckpt.parent / "slop.txt"
    sample(ckpt, out, length=args.length, temperature=args.temperature, top_k=args.top_k, prompt=args.prompt or "\n")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="vox3", description="prompt in, stylistic slop out")
    p.add_argument("--version", action="version", version=f"vox3 {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="full pipeline: prompt → corpus → train → slop")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("prompt", nargs="?", help="natural-language description of the style")
    g.add_argument("--file", help="local .pdf, .epub, .txt, or .md to train on")
    r.add_argument("--runs-dir", default="runs")
    r.add_argument("--iters", type=int, default=3000)
    r.add_argument("--block-size", type=int, default=128)
    r.add_argument("--batch-size", type=int, default=64)
    r.add_argument("--n-layer", type=int, default=4)
    r.add_argument("--n-head", type=int, default=4)
    r.add_argument("--n-embd", type=int, default=256)
    r.add_argument("--length", type=int, default=2000)
    r.add_argument("--temperature", type=float, default=0.9)
    r.add_argument("--top-k", type=int, default=40)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("sample", help="sample from an existing checkpoint")
    s.add_argument("checkpoint", help="path to model.pt")
    s.add_argument("--out", help="output path (default: next to checkpoint)")
    s.add_argument("--length", type=int, default=2000)
    s.add_argument("--temperature", type=float, default=0.9)
    s.add_argument("--top-k", type=int, default=40)
    s.add_argument("--prompt", default="\n")
    s.set_defaults(func=cmd_sample)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
