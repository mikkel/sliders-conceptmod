#!/usr/bin/env python3
"""MiniMax-H3 UNI diagnostic: lighting / concept gap vs identity leak.

No train loss. Frozen teachers + encoder hold + optional student LoRA
forwards on the same packed t2va stack as ``train_lora_minimax_h3.py``.

``--dummy`` is the CI / CPU path (no Hub, no GPU, no MiniMax-H3 weights).
Live: FL2VA / t2va, ``--device`` + optional ``--encoder_device``,
``--load_h3_lora`` for an existing adapter.

    PYTHONPATH=. python conceptmod/textsliders/diag_minimax_h3_uni.py --dummy \\
      --save_dir /tmp/h3-diag

    CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/diag_minimax_h3_uni.py \\
      --prompts_file conceptmod/textsliders/data/prompts-minimax-h3-chiaroscuro.yaml \\
      --load_h3_lora models/chiaroscuro-minimax-h3-uni-v5 \\
      --device cuda:0 --save_dir models/chiaroscuro-minimax-h3-uni-v5/diag
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.minimax_h3_backend import (
    DEFAULT_LORA_UP_INIT_STD,
    DEFAULT_MODEL,
    DEFAULT_VARIANT,
    DEFAULT_WORKFLOW,
    MiniMaxH3Backend,
    h3_pack_feature_dim,
)
from conceptmod.textsliders.minimax_h3_uni import (
    DEFAULT_HOLD_MODE,
    HOLD_MODES,
    apply_unused_hold,
    concept_token_ids,
    cosine_l2,
    embed_gap_energy_frac,
    hold_effectiveness_metrics,
    unused_hold_mask,
    unused_token_ids,
    velocity_pair,
)
from conceptmod.textsliders.train_lora_minimax_h3 import (
    build_backend,
    load_slider_rows,
)

HOW_TO_READ = {
    "lighting_gap": (
        "Frozen teacher packed velocity v_plus vs v_neu (scale 0, same "
        "video/audio noise). High L2 / lower cos = a lighting (concept) "
        "axis exists. Near-zero L2 = teachers collapsed — hold or captions "
        "killed the gap before the student can match it."
    ),
    "embed_gap_energy": (
        "After apply_unused_hold, fraction of ||held_plus − aligned_neu||^2 "
        "on concept-token rows vs held rows. concept_frac ~1 and held_frac ~0 "
        "means the remaining embed gap is lighting words. "
        "unheld_nonconcept_frac >0 is the attributes-mode identity leak "
        "(shared subject still encode(plus))."
    ),
    "hold": (
        "Encoder after non_concept hold. held_max_abs / held_mean_abs = "
        "max/mean |held_plus_row − encode(neu)_row| on held tokens "
        "(should be ~0). concept_mean_abs = mean |plus − neu| on free "
        "concept tokens (should be >0). Embed hold alone does not prove "
        "Omni LoRA will keep identity — read student metrics too."
    ),
    "student.scale0_vs_teacher_neu": (
        "Student scale 0 on the neu pack vs frozen neu teacher. "
        "cos~1 / l2~0 = adapter-off is identity."
    ),
    "student.scale1_vs_teacher_plus": (
        "Student scale 1 on the plus pack vs frozen plus teacher. "
        "High cos / low l2 = lighting match."
    ),
    "student.scale1_vs_scale0": (
        "Identity-drift proxy: student plus@1 vs student neu@0. "
        "High cos = structure held; low cos = rewrite (lighting and/or "
        "identity). Read with lighting_gap and scale1_vs_teacher_plus: "
        "lighting YES + this low = likely identity leak; lighting NO + "
        "this high = hold crushed the move (chiaro v4 room)."
    ),
    "student.neu_lora_on_vs_off": (
        "LoRA on vs off under the neu packed sequence only. "
        "Low cos = adapter rewrites neu structure (identity leak on the "
        "hold caption)."
    ),
}

DIAG_ROW_KEYS = (
    "index",
    "target",
    "positive",
    "neutral",
    "attributes",
    "n_held",
    "n_free",
    "n_concept",
    "lighting_gap",
    "embed_gap_energy",
    "hold",
    "student",
)

STUDENT_KEYS = (
    "scale0_vs_teacher_neu",
    "scale1_vs_teacher_plus",
    "scale1_vs_scale0",
    "neu_lora_on_vs_off",
)

COS_L2_KEYS = ("cos", "l2")


def parse_diag_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MiniMax-H3 UNI lighting-gap vs identity-leak diagnostic (no train)"
    )
    p.add_argument("--name", type=str, default="minimax-h3-uni-diag")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    p.add_argument("--variant", type=str, default=DEFAULT_VARIANT, choices=["FL2VA", "Ref2VA"])
    p.add_argument(
        "--workflow",
        type=str,
        default=DEFAULT_WORKFLOW,
        choices=["t2va", "fl2va", "ref2va"],
    )
    p.add_argument(
        "--prompts_file",
        type=str,
        default=str(
            Path(__file__).resolve().parent / "data" / "prompts-minimax-h3.yaml"
        ),
    )
    p.add_argument("--attributes", type=str, default="")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--lora_up_init_std", type=float, default=DEFAULT_LORA_UP_INIT_STD)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--encoder_device",
        type=str,
        default=None,
        help="Optional second GPU for Qwen3-VL-32B (same as train).",
    )
    p.add_argument("--short_side", type=int, default=768)
    p.add_argument(
        "--hold_mode",
        type=str,
        default=DEFAULT_HOLD_MODE,
        choices=list(HOLD_MODES),
    )
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument(
        "--load_h3_lora",
        type=str,
        default=None,
        help="Dir or .safetensors with custom lora_h3-* keys (not PEFT)",
    )
    p.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Cap slider rows (default: all, including attribute pins)",
    )
    p.add_argument(
        "--dummy",
        action="store_true",
        help="CPU mock packed sequence; no Hub, no GPU, no MiniMax-H3 weights",
    )
    return p.parse_args(argv)


def _shared_latents(
    backend: MiniMaxH3Backend,
    *,
    seed: int,
    index: int,
    batch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(int(seed) + int(index))
    video = torch.randn(
        batch, 2, h3_pack_feature_dim(backend.transformer, kind="video"), generator=g,
    )
    audio = torch.randn(
        batch, 2, h3_pack_feature_dim(backend.transformer, kind="audio"), generator=g,
    )
    return video, audio


def _vel(out) -> torch.Tensor:
    return velocity_pair(out.sample, out.audio_sample)


@torch.no_grad()
def diagnose_row(
    backend: MiniMaxH3Backend,
    row: dict[str, Any],
    *,
    hold_mode: str,
    seed: int,
    index: int,
) -> dict[str, Any]:
    """One slider yaml row (after unused-attribute pin)."""
    tokenizer = backend.tokenizer
    plus_enc = backend.encode_text(row["positive"], frozen=True)
    neu_enc = backend.encode_text(row["neutral"], frozen=True)
    concept = concept_token_ids(tokenizer, row["positive"], row["neutral"])
    unused = unused_token_ids(tokenizer, row.get("attributes") or [])
    hold_mask = unused_hold_mask(
        plus_enc.token_ids, unused, concept, hold_mode=hold_mode,
    )
    held_hidden = apply_unused_hold(
        plus_enc.embeds,
        neu_enc.embeds,
        plus_enc.token_ids,
        neu_enc.token_ids,
        hold_mask,
    )
    hold = hold_effectiveness_metrics(
        plus_enc.embeds,
        neu_enc.embeds,
        plus_enc.token_ids,
        neu_enc.token_ids,
        hold_mask,
        concept_ids=concept,
        held_hidden=held_hidden,
    )
    energy = embed_gap_energy_frac(
        plus_enc.embeds,
        neu_enc.embeds,
        plus_enc.token_ids,
        neu_enc.token_ids,
        hold_mask,
        concept_ids=concept,
        held_hidden=held_hidden,
    )
    video, audio = _shared_latents(
        backend, seed=seed, index=index, batch=int(plus_enc.embeds.shape[0]),
    )
    packed_plus = backend.pack_t2va(
        plus_enc,
        video_latents=video,
        audio_latents=audio,
        hold_neu=neu_enc,
        hold_mask=hold_mask,
    )
    packed_neu = backend.pack_t2va(
        neu_enc,
        video_latents=packed_plus.hidden_states.detach(),
        audio_latents=packed_plus.audio_hidden_states.detach(),
    )
    teacher_plus = backend.forward_velocity(packed_plus, scale=0.0)
    teacher_neu = backend.forward_velocity(packed_neu, scale=0.0)
    v_plus = _vel(teacher_plus)
    v_neu = _vel(teacher_neu)
    lighting_gap = cosine_l2(v_plus, v_neu)

    student_0_neu = _vel(backend.forward_velocity(packed_neu, scale=0.0))
    student_1_plus = _vel(backend.forward_velocity(packed_plus, scale=1.0))
    student_1_neu = _vel(backend.forward_velocity(packed_neu, scale=1.0))
    student = {
        "scale0_vs_teacher_neu": cosine_l2(student_0_neu, v_neu),
        "scale1_vs_teacher_plus": cosine_l2(student_1_plus, v_plus),
        "scale1_vs_scale0": cosine_l2(student_1_plus, student_0_neu),
        "neu_lora_on_vs_off": cosine_l2(student_1_neu, student_0_neu),
    }
    return {
        "index": int(index),
        "target": str(row.get("target") or row.get("neutral") or ""),
        "positive": row["positive"],
        "neutral": row["neutral"],
        "attributes": list(row.get("attributes") or []),
        "n_held": int(hold["n_held"]),
        "n_free": int(hold["n_free"]),
        "n_concept": int(hold["n_concept"]),
        "lighting_gap": lighting_gap,
        "embed_gap_energy": energy,
        "hold": {
            "held_max_abs": float(hold["held_max_abs"]),
            "held_mean_abs": float(hold["held_mean_abs"]),
            "concept_mean_abs": float(hold["concept_mean_abs"]),
        },
        "student": student,
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _aggregates(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    def _pick(path: tuple[str, ...]) -> list[float]:
        out: list[float] = []
        for row in rows:
            cur: Any = row
            for key in path:
                cur = cur[key]
            out.append(float(cur))
        return out

    return {
        "lighting_gap_cos_mean": _mean(_pick(("lighting_gap", "cos"))),
        "lighting_gap_l2_mean": _mean(_pick(("lighting_gap", "l2"))),
        "hold_max_abs_max": max(_pick(("hold", "held_max_abs"))) if rows else None,
        "hold_mean_abs_mean": _mean(_pick(("hold", "held_mean_abs"))),
        "concept_mean_abs_mean": _mean(_pick(("hold", "concept_mean_abs"))),
        "concept_gap_energy_frac_mean": _mean(
            _pick(("embed_gap_energy", "concept_frac"))
        ),
        "held_gap_energy_frac_mean": _mean(_pick(("embed_gap_energy", "held_frac"))),
        "scale0_vs_teacher_neu_cos_mean": _mean(
            _pick(("student", "scale0_vs_teacher_neu", "cos"))
        ),
        "scale1_vs_teacher_plus_cos_mean": _mean(
            _pick(("student", "scale1_vs_teacher_plus", "cos"))
        ),
        "scale1_vs_scale0_cos_mean": _mean(
            _pick(("student", "scale1_vs_scale0", "cos"))
        ),
        "neu_lora_on_vs_off_cos_mean": _mean(
            _pick(("student", "neu_lora_on_vs_off", "cos"))
        ),
    }


def format_diag_table(summary: dict[str, Any]) -> str:
    """Short stdout table. One line per slider row."""
    loaded = summary.get("load_h3_lora") or "none"
    header = (
        f"=== MiniMax-H3 UNI diag (dummy={summary.get('dummy')}, "
        f"hold={summary.get('hold_mode')}, lora={loaded}) ==="
    )
    cols = (
        f"{'#':>2}  {'target':<36}  {'light_cos':>9}  {'light_l2':>8}  "
        f"{'hold_max':>8}  {'cpt_abs':>7}  {'s0_neu':>6}  {'s1_plus':>7}  "
        f"{'s1_s0':>6}  {'neu_onoff':>9}"
    )
    lines = [header, cols]
    for row in summary.get("rows") or []:
        target = str(row.get("target") or row.get("neutral") or "")[:36]
        lg = row["lighting_gap"]
        hold = row["hold"]
        st = row["student"]
        lines.append(
            f"{row['index']:>2}  {target:<36}  "
            f"{lg['cos']:>9.4f}  {lg['l2']:>8.4f}  "
            f"{hold['held_max_abs']:>8.4f}  {hold['concept_mean_abs']:>7.4f}  "
            f"{st['scale0_vs_teacher_neu']['cos']:>6.3f}  "
            f"{st['scale1_vs_teacher_plus']['cos']:>7.3f}  "
            f"{st['scale1_vs_scale0']['cos']:>6.3f}  "
            f"{st['neu_lora_on_vs_off']['cos']:>9.3f}"
        )
    agg = summary.get("aggregates") or {}
    if agg:
        lines.append(
            "mean lighting_cos={:.4f} lighting_l2={:.4f}  "
            "s1_plus_cos={:.3f} s1_s0_cos={:.3f} neu_onoff_cos={:.3f}".format(
                agg.get("lighting_gap_cos_mean") or 0.0,
                agg.get("lighting_gap_l2_mean") or 0.0,
                agg.get("scale1_vs_teacher_plus_cos_mean") or 0.0,
                agg.get("scale1_vs_scale0_cos_mean") or 0.0,
                agg.get("neu_lora_on_vs_off_cos_mean") or 0.0,
            )
        )
    lines.append(
        "read: light_l2 high = plus/neu teachers differ. hold_max~0 = "
        "identity tokens pinned. s1_plus high = lighting match. "
        "s1_s0 high = structure held; low = rewrite. "
        "neu_onoff low = adapter rewrites neu."
    )
    return "\n".join(lines)


@torch.no_grad()
def run_diag(
    args: argparse.Namespace,
    backend: MiniMaxH3Backend | None = None,
) -> dict[str, Any]:
    """Run the diagnostic. Works with trainer args or ``parse_diag_args``."""
    rows = load_slider_rows(args.prompts_file, getattr(args, "attributes", "") or "")
    cap = getattr(args, "max_rows", None)
    if cap is None:
        cap = getattr(args, "sample_max_rows", None)
    if cap is not None:
        rows = rows[: max(0, int(cap))]
    if not rows:
        raise ValueError("no slider rows to diagnose")
    torch.manual_seed(int(args.seed))
    backend = backend or build_backend(args)
    loaded_lora = None
    if getattr(args, "load_h3_lora", None):
        loaded_lora = backend.load_trained(args.load_h3_lora)
        print(f"loaded lora_h3 weights from {loaded_lora}")
    hold_mode = str(getattr(args, "hold_mode", DEFAULT_HOLD_MODE))
    diagnosed: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        diagnosed.append(
            diagnose_row(
                backend, row, hold_mode=hold_mode, seed=int(args.seed), index=index,
            )
        )
    name = str(getattr(args, "name", None) or "minimax-h3-uni-diag")
    summary: dict[str, Any] = {
        "name": name,
        "backend": "minimax_h3",
        "recipe": "minimax_h3_uni_diag",
        "dummy": bool(args.dummy),
        "model_id": args.model_id,
        "variant": args.variant,
        "workflow": args.workflow,
        "hold_mode": hold_mode,
        "load_h3_lora": loaded_lora,
        "device": "cpu" if args.dummy else args.device,
        "encoder_device": (
            None if args.dummy else getattr(args, "encoder_device", None)
        ),
        "seed": int(args.seed),
        "n_rows": len(diagnosed),
        "how_to_read": HOW_TO_READ,
        "aggregates": _aggregates(diagnosed),
        "rows": diagnosed,
    }
    save_dir = Path(args.save_dir or f"models/{name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{name}_diag.json"
    out_path.write_text(json.dumps(summary, indent=2))
    table = format_diag_table(summary)
    print(table)
    print(f"wrote {out_path}")
    summary["table"] = table
    summary["path"] = str(out_path)
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return run_diag(parse_diag_args(argv))


if __name__ == "__main__":
    main()
