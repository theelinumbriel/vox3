"""Load a trained checkpoint and generate text."""
from __future__ import annotations

from pathlib import Path

import torch

from .model import GPT, GPTConfig


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sample(
    ckpt_path: Path,
    out_path: Path,
    length: int = 2000,
    temperature: float = 0.9,
    top_k: int = 40,
    prompt: str = "\n",
) -> Path:
    device = _pick_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    stoi = ckpt["stoi"]
    itos = {int(k): v for k, v in ckpt["itos"].items()}
    unk = next(iter(stoi.values()))
    start = torch.tensor([[stoi.get(c, unk) for c in prompt]], dtype=torch.long, device=device)
    out = model.generate(start, max_new_tokens=length, temperature=temperature, top_k=top_k)
    text = "".join(itos[int(i)] for i in out[0].tolist())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"  wrote {len(text)} chars → {out_path}")
    return out_path
