#!/usr/bin/env python3
"""Compare condition/velocity/LoRA deltas for energy vs gender transformer sliders."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/ml2/music/.cache/huggingface")

import torch
from safetensors.torch import load_file

ROOT = Path("/ml2/music/sliders-conceptmod")
sys.path.insert(0, str(ROOT))
from conceptmod.textsliders.lora import LoRANetwork
from conceptmod.textsliders.train_lora_music3 import _load_transformer

MODEL = Path("/ml2/music/models/MiniMax-Music3")


def _load_conds(cache_dir: Path) -> dict[str, torch.Tensor]:
    out = {}
    for path in cache_dir.glob("*.pt"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cond = payload["condition"] if isinstance(payload, dict) else payload
        prompt = payload.get("prompt", path.name) if isinstance(payload, dict) else path.name
        low = prompt.lower()
        if "pop-punk" in low or "soprano" in low or "woman singing" in low or "adult woman" in low:
            key = "pos"
        elif "lullaby" in low or "baritone" in low or "adult man" in low or "man singing" in low:
            key = "neg"
        else:
            key = "neu"
        out[key] = (cond.float(), prompt)
    return out


def _pair(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    n = min(a.shape[1], b.shape[1])
    x, y = a[:, :n].flatten(), b[:, :n].flatten()
    l2 = (x - y).norm().item()
    rel = l2 / (x.norm().item() + 1e-8)
    return l2, rel


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda:0")
    cases = {
        "energy": (
            ROOT / "cache/energy-v2",
            ROOT / "models/energy-slider-v2/energy_alpha8.0_rank8_full_last.safetensors",
        ),
        "gender": (
            ROOT / "cache/gender",
            ROOT / "models/gender-slider/gender_alpha8.0_rank8_full_last.safetensors",
        ),
    }
    for name, (cache, weights) in cases.items():
        transformer = _load_transformer(MODEL, device)
        print(f"transformer loaded for {name}")
        conds = _load_conds(cache)
        print(f"\n==== {name} conditions {list(conds)} ====")
        for k, (c, p) in conds.items():
            print(f"  {k} {tuple(c.shape)} {p[:70]!r}")
        l2, rel = _pair(conds["pos"][0], conds["neg"][0])
        print(f"  cond pos-neg L2={l2:.2f} rel={rel:.4f}")

        neu = conds["neu"][0].to(device=device, dtype=torch.bfloat16)
        pos = conds["pos"][0].to(device=device, dtype=torch.bfloat16)
        neg = conds["neg"][0].to(device=device, dtype=torch.bfloat16)
        n = min(neu.shape[1], pos.shape[1], neg.shape[1])
        neu, pos, neg = neu[:, :n], pos[:, :n], neg[:, :n]
        latents = torch.randn(1, 128, n, device=device, dtype=torch.bfloat16)
        t = torch.tensor([0.5], device=device, dtype=torch.bfloat16)
        vel_neu = transformer(latents, t, neu, return_dict=False)[0].float()
        vel_pos = transformer(latents, t, pos, return_dict=False)[0].float()
        vel_neg = transformer(latents, t, neg, return_dict=False)[0].float()
        d = (vel_pos - vel_neg).norm().item()
        print(f"  vel ||pos-neg||={d:.3f}  rel_to_neu={d / (vel_neu.norm().item() + 1e-8):.4f}")

        network = LoRANetwork(
            transformer,
            rank=8,
            alpha=8.0,
            multiplier=1.0,
            target_replace=["MiniMaxMusic3Attention"],
            train_method="full",
            delimiter="-",
        ).to(device)
        state = load_file(str(weights), device="cpu")
        missing, unexpected = network.load_state_dict(state, strict=False)
        print(f"  lora load missing={len(missing)} unexpected={len(unexpected)} modules={len(network.unet_loras)}")
        norms = [p.detach().float().norm().item() for p in network.parameters()]
        print(f"  lora param L2 sum={sum(norms):.3f} max={max(norms):.3f}")

        for scale in (0.0, 2.0, 4.0, 8.0):
            network.set_lora_slider(scale)
            with network:
                vel = transformer(latents, t, neu, return_dict=False)[0].float()
            delta = (vel - vel_neu).norm().item()
            print(f"  scale={scale:g} ||vel-vel0||={delta:.3f} rel={delta / (vel_neu.norm().item() + 1e-8):.4f}")
        # unwrap by leaving multipliers at 0
        del network, transformer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
