"""Notrigger-style slider on MiniMax Music 3's language model.

Vocal identity is chosen in the AR stage, not the flow transformer. This trains
LoRA on Qwen3Attention so a neutral caption + LoRA scale moves the prompt
hidden state toward a female or male caption.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HF_HOME = "/ml2/music/.cache/huggingface"
os.environ["HF_HOME"] = _HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{_HF_HOME}/hub"
os.environ["HF_HUB_CACHE"] = f"{_HF_HOME}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{_HF_HOME}/hub"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.lora import LoRANetwork

DEFAULT_MODEL = Path("/ml2/music/models/MiniMax-Music3")
TARGET_REPLACE = ["Qwen3Attention"]

_IM_START, _IM_END = "<|im_start|>", "<|im_end|>"
_CAPTION_START, _CAPTION_END = "<|caption_start|>", "<|caption_end|>"
_LYRICS_START, _LYRICS_END = "<|lyrics_start|>", "<|lyrics_end|>"
_AUDIO_START = "<|audio_start|>"


def _assemble(prompt: str, lyrics: str) -> str:
    from diffusers.modular_pipelines.minimax_music3.encoders import (
        _clean_caption,
        _normalize_lyrics,
    )

    return (
        f"{_IM_START}{_CAPTION_START}{_clean_caption(prompt)}{_CAPTION_END}"
        f"{_LYRICS_START}{_normalize_lyrics(lyrics)}{_LYRICS_END}{_IM_END}{_AUDIO_START}"
    )


def _load_rows(path: Path) -> tuple[list[dict], dict]:
    """All prompt rows (with `attributes` expansion) plus top-level label metadata."""
    from conceptmod.textsliders.train_lora_music3 import _expand_attributes

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    meta: dict = {}
    if isinstance(raw, dict):
        meta = {
            "plus_label": str(raw.get("plus_label") or ""),
            "minus_label": str(raw.get("minus_label") or ""),
            "recommended_range": raw.get("recommended_range") or [-2.0, 2.0],
        }
        raw = raw.get("rows")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"prompts file is empty: {path}")
    rows: list[dict] = []
    for item in raw:
        rows.extend(_expand_attributes(item))
    return rows, meta


def _tokenize(tokenizer, text: str, device: torch.device):
    out = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return out["input_ids"].to(device), out["attention_mask"].to(device)


@torch.no_grad()
def _encode_static(lm, input_ids, attention_mask) -> torch.Tensor:
    hidden = lm.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    # Last real token (the audio-start token) is what AR continues from.
    lengths = attention_mask.sum(dim=1) - 1
    gather = lengths.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
    last = hidden.gather(1, gather.clamp(min=0)).squeeze(1)
    return last.float()


def _encode_train(lm, input_ids, attention_mask) -> torch.Tensor:
    hidden = lm.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    lengths = attention_mask.sum(dim=1) - 1
    gather = lengths.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
    last = hidden.gather(1, gather.clamp(min=0)).squeeze(1)
    return last.float()


def _set_scale(network: LoRANetwork, scale: float) -> None:
    network.set_lora_slider(scale)
    for lora in network.unet_loras:
        lora.multiplier = float(scale)


def train(args: argparse.Namespace) -> Path:
    device = torch.device(f"cuda:{int(args.device)}")
    rows, prompts_meta = _load_rows(Path(args.prompts_file))
    if args.plus_label:
        prompts_meta["plus_label"] = args.plus_label
    if args.minus_label:
        prompts_meta["minus_label"] = args.minus_label

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.model_dir) / "tokenizer"),
        local_files_only=True,
    )
    lm = AutoModelForCausalLM.from_pretrained(
        str(Path(args.model_dir) / "language_model"),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    lm.to(device)
    lm.eval()
    lm.requires_grad_(False)

    beta = float(args.common_beta)
    row_data = []
    for index, row in enumerate(rows):
        lyrics = str(row["lyrics"])
        texts = {
            "neutral": _assemble(str(row.get("neutral") or row["target"]), lyrics),
            "positive": _assemble(str(row["positive"]), lyrics),
            "negative": _assemble(str(row["negative"]), lyrics),
        }
        tokens = {name: _tokenize(tokenizer, text, device) for name, text in texts.items()}
        with torch.no_grad():
            pos_tgt = _encode_static(lm, *tokens["positive"])
            neg_tgt = _encode_static(lm, *tokens["negative"])
            neu_ref = _encode_static(lm, *tokens["neutral"])
        target_cos = F.cosine_similarity(pos_tgt - neu_ref, neg_tgt - neu_ref, dim=-1).mean().item()
        print(
            f"row {index}: tokens={tokens['neutral'][0].shape[1]} "
            f"L2 pos={torch.norm(pos_tgt - neu_ref).item():.3f} "
            f"neg={torch.norm(neg_tgt - neu_ref).item():.3f} "
            f"cos(pos-neu, neg-neu)={target_cos:.3f}"
            f"{'  <- collapse inherited from raw targets' if target_cos > 0.3 else ''}"
        )
        if args.symmetric:
            # Raw pos/neg targets share a large common component vs neutral (both
            # captions add axis-independent detail); training against them makes
            # +1 and -1 point the same way. Antisymmetrize around neutral instead.
            axis = (pos_tgt - neg_tgt) / 2.0
            common = (pos_tgt + neg_tgt) / 2.0 - neu_ref
            tgt_plus = neu_ref + axis + beta * common
            tgt_minus = neu_ref - axis + beta * common
        else:
            tgt_plus, tgt_minus = pos_tgt, neg_tgt
        row_data.append(
            {
                "tokens": tokens["neutral"],
                "tgt_plus": tgt_plus,
                "tgt_minus": tgt_minus,
                "neu_ref": neu_ref,
            }
        )

    network = LoRANetwork(
        lm,
        rank=args.rank,
        alpha=args.alpha,
        multiplier=1.0,
        delimiter="-",
        target_replace=TARGET_REPLACE,
        prefix="lora_te",
        train_method="full",
    ).to(device)
    n_mod = len(network.unet_loras)
    if n_mod == 0:
        raise RuntimeError("LoRA wrapped 0 Qwen3Attention linears")
    print(f"LM LoRA modules={n_mod} rank={args.rank}")
    for p in network.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=1e-6)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics = save_dir / f"{args.name}_train.jsonl"
    metrics_handle = metrics.open("w")

    history = []
    pbar = tqdm(range(args.steps), desc="lm-slider")
    for step in pbar:
        data = row_data[step % len(row_data)]
        neu_ids, neu_mask = data["tokens"]
        tgt_plus, tgt_minus, neu_ref = data["tgt_plus"], data["tgt_minus"], data["neu_ref"]

        _set_scale(network, 1.0)
        pred_pos = _encode_train(lm, neu_ids, neu_mask)
        ploss = F.mse_loss(pred_pos, tgt_plus)

        _set_scale(network, -1.0)
        pred_neg = _encode_train(lm, neu_ids, neu_mask)
        nloss = F.mse_loss(pred_neg, tgt_minus)

        v_pos = pred_pos - neu_ref
        v_neg = pred_neg - neu_ref
        v_pos_t = tgt_plus - neu_ref
        v_neg_t = tgt_minus - neu_ref
        cos_pos = F.cosine_similarity(v_pos, v_pos_t, dim=-1).mean()
        cos_neg = F.cosine_similarity(v_neg, v_neg_t, dim=-1).mean()
        collapse = F.cosine_similarity(v_pos, v_neg, dim=-1).mean()
        pperc = (torch.norm(pred_pos - tgt_plus) / torch.norm(v_pos_t).clamp_min(1e-6)).item()
        nperc = (torch.norm(pred_neg - tgt_minus) / torch.norm(v_neg_t).clamp_min(1e-6)).item()

        loss = ploss + nloss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_value_(network.parameters(), clip_value=1.0)
        opt.step()

        row = {
            "step": step,
            "row": step % len(row_data),
            "loss": float(loss.detach()),
            "pperc": pperc,
            "nperc": nperc,
            "cos_pos": float(cos_pos.detach()),
            "cos_neg": float(cos_neg.detach()),
            "collapse": float(collapse.detach()),
        }
        history.append(row)
        pbar.set_description(
            f"loss {row['loss']:.4f} p%{pperc*100:.1f} n%{nperc*100:.1f} "
            f"c+ {row['cos_pos']:.2f} c- {row['cos_neg']:.2f} col {row['collapse']:.2f}"
        )
        metrics_handle.write(json.dumps(row) + "\n")
        metrics_handle.flush()

        if args.save_every and step > 0 and step % args.save_every == 0:
            network.save_weights(save_dir / f"{args.name}_step{step}.safetensors", dtype=torch.float32)

    metrics_handle.close()
    last = save_dir / f"{args.name}_last.safetensors"
    network.save_weights(last, dtype=torch.float32)
    summary = {
        "schema": 3,
        "name": args.name,
        "checkpoint": str(last),
        "weights": str(last),
        "modules": n_mod,
        "rank": args.rank,
        "alpha": args.alpha,
        "steps": args.steps,
        "lr": args.lr,
        "rows": len(row_data),
        "symmetric": bool(args.symmetric),
        "common_beta": beta,
        "first": history[0],
        "last": history[-1],
        "target_replace": TARGET_REPLACE,
        "kind": "language_model",
        "prefix": "lora_te",
        "delimiter": "-",
        "train_method": "full",
        # Trained at ±1 by construction, so the user scale is the raw multiplier.
        "unit_scale": 1.0,
        "plus_label": prompts_meta.get("plus_label", ""),
        "minus_label": prompts_meta.get("minus_label", ""),
        "recommended_range": prompts_meta.get("recommended_range", [-2.0, 2.0]),
        "prompts_file": args.prompts_file,
    }
    (save_dir / f"{args.name}_last.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return last


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="gender-lm")
    p.add_argument("--model_dir", default=str(DEFAULT_MODEL))
    p.add_argument("--prompts_file", required=True)
    p.add_argument("--save_dir", default="/ml2/music/sliders-conceptmod/models/gender-lm-slider")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--device", type=int, default=0)
    p.add_argument(
        "--symmetric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="antisymmetrize targets: tgt(±1) = neu ± (pos-neg)/2 (fixes +/- collapse)",
    )
    p.add_argument(
        "--common_beta",
        type=float,
        default=0.0,
        help="blend back beta*(midpoint - neutral) if symmetric ±1 sounds too weak",
    )
    p.add_argument("--plus_label", default=None, help="override sidecar plus_label")
    p.add_argument("--minus_label", default=None, help="override sidecar minus_label")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
