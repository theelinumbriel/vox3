"""Local FastAPI UI for vox3: paste prompt + drag files, see slop."""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .clean import build_corpus
from .fetch import dispatch
from .fetch.epub import ingest as ingest_epub
from .fetch.pdf import ingest as ingest_pdf
from .interpret import CorpusSpec, interpret
from .sample import sample
from .train import train

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>vox3</title>
<style>
  :root { color-scheme: dark; }
  html, body { margin: 0; padding: 0; height: 100%; background: #0b0b0d; color: #e8e6e3; font: 15px/1.5 ui-monospace, Menlo, Consolas, monospace; }
  a { color: #8ab4f8; }
  .wrap { max-width: 820px; margin: 0 auto; padding: 32px 24px 80px; }
  h1 { font-size: 22px; margin: 0 0 4px; font-weight: 500; letter-spacing: 0.5px; }
  .sub { color: #7d7a74; margin-bottom: 28px; font-size: 13px; }
  textarea { width: 100%; min-height: 120px; background: #15151a; color: #e8e6e3; border: 1px solid #2a2a30; border-radius: 6px; padding: 12px; font: inherit; resize: vertical; box-sizing: border-box; }
  textarea:focus { outline: none; border-color: #4a4a52; }
  .drop { margin-top: 12px; border: 1px dashed #3a3a42; border-radius: 6px; padding: 20px; text-align: center; color: #7d7a74; background: #121216; }
  .drop.hover { border-color: #8ab4f8; color: #e8e6e3; background: #15151c; }
  .files { margin-top: 10px; font-size: 13px; color: #a09d97; }
  .files div { padding: 2px 0; }
  details { margin-top: 14px; color: #a09d97; }
  summary { cursor: pointer; user-select: none; }
  .row { display: flex; gap: 12px; margin-top: 10px; flex-wrap: wrap; }
  .row label { display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: #7d7a74; }
  .row input { background: #15151a; color: #e8e6e3; border: 1px solid #2a2a30; border-radius: 4px; padding: 6px 8px; font: inherit; width: 90px; }
  button { margin-top: 18px; background: #e8e6e3; color: #0b0b0d; border: 0; border-radius: 6px; padding: 10px 20px; font: inherit; font-weight: 500; cursor: pointer; }
  button:hover { background: #fff; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  pre.log { margin-top: 24px; background: #121216; border: 1px solid #2a2a30; border-radius: 6px; padding: 12px; height: 300px; overflow: auto; font-size: 12px; white-space: pre-wrap; word-break: break-word; }
  pre.slop { margin-top: 16px; background: #15151a; border: 1px solid #2a2a30; border-radius: 6px; padding: 16px; max-height: 500px; overflow: auto; font-size: 14px; white-space: pre-wrap; word-break: break-word; }
  .dim { color: #7d7a74; }
  .err { color: #f28b82; }
  .ok { color: #81c995; }
</style>
</head>
<body>
<div class="wrap">
  <h1>vox3</h1>
  <div class="sub">paste a style, drop some files, get slop.</div>

  <form id="f">
    <textarea id="prompt" name="prompt" placeholder="e.g., 1920s pulp sci-fi with breathless adjectives&#10;or leave blank and just drop files"></textarea>

    <div id="drop" class="drop">drop .pdf / .epub / .txt / .md here (or <label style="color:#8ab4f8;cursor:pointer;text-decoration:underline">pick<input id="pick" type="file" multiple style="display:none" accept=".pdf,.epub,.txt,.md"></label>)</div>
    <div id="files" class="files"></div>

    <details>
      <summary>training knobs</summary>
      <div class="row">
        <label>iters<input id="iters" type="number" value="3000" min="100" max="20000"></label>
        <label>n_layer<input id="n_layer" type="number" value="4" min="1" max="12"></label>
        <label>n_embd<input id="n_embd" type="number" value="256" min="32" max="768"></label>
        <label>block_size<input id="block_size" type="number" value="128" min="32" max="512"></label>
        <label>temperature<input id="temperature" type="number" step="0.1" value="0.9"></label>
        <label>top_k<input id="top_k" type="number" value="40"></label>
        <label>length<input id="length" type="number" value="2000"></label>
      </div>
    </details>

    <button id="go" type="submit">run</button>
  </form>

  <pre id="log" class="log"></pre>
  <pre id="slop" class="slop" style="display:none"></pre>
</div>

<script>
const dropped = [];
const drop = document.getElementById("drop");
const pick = document.getElementById("pick");
const filesDiv = document.getElementById("files");
const logEl = document.getElementById("log");
const slopEl = document.getElementById("slop");
const btn = document.getElementById("go");

function renderFiles() {
  filesDiv.innerHTML = dropped.map(f => `<div>+ ${f.name} <span class="dim">(${(f.size/1024).toFixed(1)} KB)</span></div>`).join("");
}
function addFiles(list) {
  for (const f of list) dropped.push(f);
  renderFiles();
}
["dragenter","dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("hover"); }));
["dragleave","drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("hover"); }));
drop.addEventListener("drop", e => { addFiles(e.dataTransfer.files); });
pick.addEventListener("change", e => { addFiles(e.target.files); e.target.value = ""; });

function log(line, cls="") {
  const span = document.createElement("span");
  if (cls) span.className = cls;
  span.textContent = line + "\n";
  logEl.appendChild(span);
  logEl.scrollTop = logEl.scrollHeight;
}

document.getElementById("f").addEventListener("submit", async (e) => {
  e.preventDefault();
  logEl.innerHTML = "";
  slopEl.style.display = "none";
  slopEl.textContent = "";
  btn.disabled = true;

  const fd = new FormData();
  fd.append("prompt", document.getElementById("prompt").value);
  for (const k of ["iters","n_layer","n_embd","block_size","temperature","top_k","length"]) {
    fd.append(k, document.getElementById(k).value);
  }
  for (const f of dropped) fd.append("files", f);

  try {
    const resp = await fetch("/run", { method: "POST", body: fd });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop();
      for (const chunk of parts) {
        const line = chunk.replace(/^data: /, "");
        if (!line) continue;
        if (line.startsWith("[SLOP]")) {
          slopEl.style.display = "block";
          slopEl.textContent = line.slice(6);
        } else if (line.startsWith("[ERR]")) {
          log(line.slice(5), "err");
        } else if (line.startsWith("[DONE]")) {
          log(line.slice(6), "ok");
        } else {
          log(line);
        }
      }
    }
  } catch (err) {
    log("fetch failed: " + err, "err");
  } finally {
    btn.disabled = false;
  }
});
</script>
</body>
</html>
"""


app = FastAPI(title="vox3")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return INDEX_HTML


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "run"


def _ingest_local(path: Path, raw_dir: Path) -> list[Path]:
    suf = path.suffix.lower()
    if suf == ".pdf":
        return ingest_pdf(path, raw_dir)
    if suf == ".epub":
        return ingest_epub(path, raw_dir)
    if suf in {".txt", ".md"}:
        raw_dir.mkdir(parents=True, exist_ok=True)
        dest = raw_dir / f"txt-{path.stem}.txt"
        dest.write_text(path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        return [dest]
    return []


class _Pipe(io.TextIOBase):
    def __init__(self, q: queue.Queue):
        self.q = q
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.q.put(line)
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            self.q.put(self._buf)
            self._buf = ""


def _pipeline(
    run_dir: Path,
    prompt: str,
    uploaded: list[Path],
    iters: int,
    n_layer: int,
    n_embd: int,
    block_size: int,
    temperature: float,
    top_k: int,
    length: int,
    q: queue.Queue,
) -> None:
    try:
        with contextlib.redirect_stdout(_Pipe(q)), contextlib.redirect_stderr(_Pipe(q)):
            paths: list[Path] = []
            raw_dir = run_dir / "raw"

            if uploaded:
                print(f"[ingest] {len(uploaded)} uploaded file(s)")
                for p in uploaded:
                    try:
                        paths.extend(_ingest_local(p, raw_dir))
                    except Exception as e:
                        print(f"  ingest failed for {p.name}: {e}")

            if prompt.strip():
                print(f"[interpret] {prompt!r}")
                spec = interpret(prompt)
                print(f"  summary: {spec.topic_summary}")
                print(f"  notes:   {spec.notes}")
                print(f"  sources: {len(spec.sources)}")
                (run_dir / "spec.json").write_text(
                    json.dumps(
                        {
                            "topic_summary": spec.topic_summary,
                            "target_mb": spec.target_mb,
                            "notes": spec.notes,
                            "sources": [asdict(s) for s in spec.sources],
                            "raw": spec.raw,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
                print("[fetch]")
                paths.extend(dispatch.run(spec, raw_dir))

            if not paths:
                q.put("[ERR]no files or prompt sources yielded any text")
                return

            print(f"[clean] {len(paths)} files")
            corpus = run_dir / "corpus.txt"
            size = build_corpus(paths, corpus)
            print(f"  corpus.txt: {size:,} bytes")

            print("[train]")
            ckpt = train(
                corpus,
                run_dir,
                iters=iters,
                block_size=block_size,
                batch_size=64,
                n_layer=n_layer,
                n_head=max(1, n_embd // 64),
                n_embd=n_embd,
            )

            print("[sample]")
            out = run_dir / "slop.txt"
            sample(ckpt, out, length=length, temperature=temperature, top_k=top_k)

            text = out.read_text()
            q.put(f"[DONE]run dir: {run_dir}")
            q.put(f"[SLOP]{text}")
    except Exception as e:
        tb = traceback.format_exc()
        q.put(f"[ERR]{e}")
        for line in tb.splitlines():
            q.put(f"[ERR]{line}")


@app.post("/run")
async def run_endpoint(
    prompt: str = Form(""),
    iters: int = Form(3000),
    n_layer: int = Form(4),
    n_embd: int = Form(256),
    block_size: int = Form(128),
    temperature: float = Form(0.9),
    top_k: int = Form(40),
    length: int = Form(2000),
    files: list[UploadFile] = File(default=[]),
):
    runs_root = Path.cwd() / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    label = _slugify(prompt[:40]) if prompt.strip() else _slugify(files[0].filename if files else "upload")
    run_dir = runs_root / f"{ts}-{label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    uploaded: list[Path] = []
    uploads_dir = run_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    for f in files or []:
        if not f.filename:
            continue
        dest = uploads_dir / f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        uploaded.append(dest)

    q: queue.Queue = queue.Queue()
    t = threading.Thread(
        target=_pipeline,
        args=(run_dir, prompt, uploaded, iters, n_layer, n_embd, block_size, temperature, top_k, length, q),
        daemon=True,
    )
    t.start()

    async def stream():
        yield f"data: [run] {run_dir.name}\n\n"
        while True:
            try:
                line = q.get(timeout=0.1)
                yield f"data: {line}\n\n"
            except queue.Empty:
                if not t.is_alive():
                    while not q.empty():
                        yield f"data: {q.get_nowait()}\n\n"
                    return
                await asyncio.sleep(0.1)

    return StreamingResponse(stream(), media_type="text/event-stream")


def serve(host: str = "127.0.0.1", port: int = 7777) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")
