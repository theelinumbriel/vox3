"""Train a char-level nanoGPT on corpus.txt."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .model import GPT, GPTConfig, num_params


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def _batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    ix = torch.randint(0, data.size(0) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def _lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    frac = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * frac))
    return min_lr + coeff * (max_lr - min_lr)


def train(
    corpus_path: Path,
    run_dir: Path,
    iters: int = 3000,
    block_size: int = 128,
    batch_size: int = 64,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    max_lr: float = 3e-4,
    eval_every: int = 250,
) -> Path:
    device = _pick_device()
    print(f"  device: {device}")
    text = corpus_path.read_text(encoding="utf-8")
    if len(text) < 2000:
        raise RuntimeError(f"corpus too small ({len(text)} chars) — add more sources")
    print(f"  corpus: {len(text):,} chars")

    stoi, itos = _build_vocab(text)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.95 * len(data))
    train_data, val_data = data[:split], data[split:]
    print(f"  vocab: {len(stoi)}, train tokens: {len(train_data):,}, val: {len(val_data):,}")

    cfg = GPTConfig(
        vocab_size=len(stoi),
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )
    model = GPT(cfg).to(device)
    print(f"  params: {num_params(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)
    warmup = max(50, iters // 20)
    min_lr = max_lr * 0.1

    last_train_loss = float("nan")
    last_val_loss = float("nan")
    t0 = time.time()
    for step in range(iters):
        lr = _lr(step, warmup, iters, max_lr, min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = _batch(train_data, block_size, batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % eval_every == 0 or step == iters - 1:
            model.eval()
            with torch.no_grad():
                vx, vy = _batch(val_data, block_size, batch_size, device)
                _, vloss = model(vx, vy)
            dt = time.time() - t0
            last_train_loss = float(loss.item())
            last_val_loss = float(vloss.item())
            print(f"  step {step:>5d}  train {last_train_loss:.3f}  val {last_val_loss:.3f}  lr {lr:.2e}  {dt:.1f}s")
            model.train()

    ckpt = {
        "model": model.state_dict(),
        "config": asdict(cfg),
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},
    }
    ckpt_path = run_dir / "model.pt"
    torch.save(ckpt, ckpt_path)
    (run_dir / "vocab.json").write_text(json.dumps({"stoi": stoi, "itos": {str(k): v for k, v in itos.items()}}, ensure_ascii=False))

    info = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus_bytes": corpus_path.stat().st_size,
        "corpus_chars": len(text),
        "vocab_size": len(stoi),
        "params": sum(p.numel() for p in model.parameters()),
        "config": asdict(cfg),
        "iters": iters,
        "train_loss": last_train_loss,
        "val_loss": last_val_loss,
        "wall_seconds": time.time() - t0,
        "device": device,
    }
    (run_dir / "info.json").write_text(json.dumps(info, indent=2))
    print(f"  saved: {ckpt_path}")
    return ckpt_path
