#!/usr/bin/env python3
"""LTX-2.5 UNI diagnostic: expression gap vs identity leak.

No train loss. Frozen teachers + PRE-connector hold + optional student
LoRA forwards on the same pack as ``train_lora_ltx25.py``.

``--dummy`` never downloads LTX-2.5. First live: if plus/neu velocity
cos ≈ 1.0, document ``transformer_full/`` as fallback — do not silently
train a dead gap.

    PYTHONPATH=. python conceptmod/textsliders/diag_ltx25_uni.py --dummy \\
      --save_dir /tmp/ltx25-diag
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

from conceptmod.textsliders.ltx25_backend import (
    DEFAULT_ENCODER_DEVICE,
    DEFAULT_LORA_UP_INIT_STD,
    DEFAULT_MODEL,
    DEFAULT_TRAIN_HEIGHT,
    DEFAULT_TRAIN_NUM_FRAMES,
    DEFAULT_TRAIN_WIDTH,
    DEFAULT_TRANSFORMER_SUBFOLDER,
    FULL_TRANSFORMER_SUBFOLDER,
    LTX25Backend,
    ltx_pack_feature_dim,
)
from conceptmod.textsliders.ltx25_uni import (
    DEFAULT_HOLD_MODE,
    HOLD_MODES,
    apply_unused_hold,
    cosine_l2,
    embed_gap_energy_frac,
    expression_gap_is_dead,
    hold_effectiveness_metrics,
    post_connector_mean_cos,
    resolve_concept_token_ids,
    unused_hold_mask,
    unused_token_ids,
    velocity_pair,
)
from conceptmod.textsliders.train_lora_ltx25 import (
    build_backend,
    load_slider_rows,
)

HOW_TO_READ = {
    "post_connector_video": (
        "Frozen encode(plus) vs encode(neu) after video connectors, "
        "valid-row mean_cos. Live smile sat ~0.68 — this is the working "
        "teacher gap. Transplanting plus concept-token (or full plus) "
        "embeds onto neu conditioning produces teeth/smile."
    ),
    "expression_gap": (
        "Frozen teacher velocity v_plus vs v_neu (scale 0, same noise). "
        "Live distilled plus/neu velocity cos ~0.9999 is a **dead** "
        "teacher — do not train DiT velocity UNI on smile/chiaro. "
        "Kept as a negative control."
    ),
    "embed_gap_energy": (
        "After PRE-connector apply_unused_hold, fraction of "
        "||held_plus − aligned_neu||^2 on concept-token rows. "
        "concept_frac ~1 and held_frac ~0 means the remaining embed gap "
        "is smile words. unheld_nonconcept_frac >0 is attributes-mode leak."
    ),
    "hold": (
        "PRE-connector encoder after non_concept hold. held_mean_abs ~0. "
        "concept_mean_abs >0. Hold after connectors cannot pin token ids."
    ),
    "student.scale0_vs_teacher_neu": (
        "Student scale 0 on the neu pack vs frozen neu. cos~1 = identity."
    ),
    "student.scale1_vs_teacher_plus": (
        "Student scale 1 on the **neu** pack (infer path) vs frozen plus. "
        "High cos = expression match."
    ),
    "student.scale1_vs_scale0": (
        "Identity-drift proxy: student neu@1 vs student neu@0."
    ),
    "student.neu_lora_on_vs_off": (
        "LoRA on vs off under the neu pack. Low cos = adapter rewrites neu."
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
    "expression_gap",
    "post_connector_video",
    "embed_gap_energy",
    "hold",
    "student",
    "dead_gap",
    "hold_stage",
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
        description="LTX-2.5 UNI expression-gap vs identity-leak diagnostic (no train)"
    )
    p.add_argument("--name", type=str, default="ltx25-uni-diag")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    p.add_argument("--transformer_subfolder", type=str, default=DEFAULT_TRANSFORMER_SUBFOLDER)
    p.add_argument(
        "--prompts_file",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "prompts-ltx25-smile.yaml"),
    )
    p.add_argument("--attributes", type=str, default="")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--lora_up_init_std", type=float, default=DEFAULT_LORA_UP_INIT_STD)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--encoder_device", type=str, default=DEFAULT_ENCODER_DEVICE)
    p.add_argument("--hold_mode", type=str, default=DEFAULT_HOLD_MODE, choices=list(HOLD_MODES))
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--load_ltx_lora", type=str, default=None)
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--dummy", action="store_true")
    return p.parse_args(argv)


def _shared_latents(backend: LTX25Backend, *, seed: int, index: int, batch: int):
    g = torch.Generator().manual_seed(int(seed) + int(index))
    video = torch.randn(
        batch, 2, ltx_pack_feature_dim(backend.transformer, kind="video"), generator=g,
    )
    audio = torch.randn(
        batch, 2, ltx_pack_feature_dim(backend.transformer, kind="audio"), generator=g,
    )
    return video, audio


def _vel(out) -> torch.Tensor:
    return velocity_pair(out.sample, out.audio_sample)


@torch.no_grad()
def diagnose_row(
    backend: LTX25Backend,
    row: dict[str, Any],
    *,
    hold_mode: str,
    seed: int,
    index: int,
) -> dict[str, Any]:
    tokenizer = backend.tokenizer
    plus_enc = backend.encode_text(row["positive"], frozen=True)
    neu_enc = backend.encode_text(row["neutral"], frozen=True)
    concept = resolve_concept_token_ids(
        tokenizer, row["positive"], row["neutral"], row.get("concept_words") or "",
    )
    unused = unused_token_ids(tokenizer, row.get("attributes") or [])
    hold_mask = unused_hold_mask(
        plus_enc.token_ids, unused, concept, hold_mode=hold_mode,
    )
    held_hidden = apply_unused_hold(
        plus_enc.embeds, neu_enc.embeds, plus_enc.token_ids, neu_enc.token_ids, hold_mask,
    )
    hold = hold_effectiveness_metrics(
        plus_enc.embeds, neu_enc.embeds, plus_enc.token_ids, neu_enc.token_ids,
        hold_mask, concept_ids=concept, held_hidden=held_hidden,
    )
    energy = embed_gap_energy_frac(
        plus_enc.embeds, neu_enc.embeds, plus_enc.token_ids, neu_enc.token_ids,
        hold_mask, concept_ids=concept, held_hidden=held_hidden,
    )
    video, audio = _shared_latents(
        backend, seed=seed, index=index, batch=int(plus_enc.embeds.shape[0]),
    )
    packed_plus = backend.pack_t2v(
        plus_enc,
        video_latents=video,
        audio_latents=audio,
        hold_neu=neu_enc,
        hold_mask=hold_mask,
        num_frames=DEFAULT_TRAIN_NUM_FRAMES,
        height=DEFAULT_TRAIN_HEIGHT,
        width=DEFAULT_TRAIN_WIDTH,
    )
    packed_neu = backend.pack_t2v(
        neu_enc,
        video_latents=packed_plus.hidden_states.detach(),
        audio_latents=packed_plus.audio_hidden_states.detach(),
        num_frames=DEFAULT_TRAIN_NUM_FRAMES,
        height=DEFAULT_TRAIN_HEIGHT,
        width=DEFAULT_TRAIN_WIDTH,
    )
    if packed_plus.hold_stage != "pre_connector":
        raise RuntimeError("diag hold must be PRE-connector")
    teacher_plus = backend.forward_velocity(packed_plus, scale=0.0)
    teacher_neu = backend.forward_velocity(packed_neu, scale=0.0)
    v_plus = _vel(teacher_plus)
    v_neu = _vel(teacher_neu)
    expression_gap = cosine_l2(v_plus, v_neu)
    post_video = post_connector_mean_cos(
        packed_plus.encoder_hidden_states,
        packed_neu.encoder_hidden_states,
        packed_plus.encoder_attention_mask,
        packed_neu.encoder_attention_mask,
    )
    student_0_neu = _vel(backend.forward_velocity(packed_neu, scale=0.0))
    student_1_neu = _vel(backend.forward_velocity(packed_neu, scale=1.0))
    student = {
        "scale0_vs_teacher_neu": cosine_l2(student_0_neu, v_neu),
        "scale1_vs_teacher_plus": cosine_l2(student_1_neu, v_plus),
        "scale1_vs_scale0": cosine_l2(student_1_neu, student_0_neu),
        "neu_lora_on_vs_off": cosine_l2(student_1_neu, student_0_neu),
    }
    return {
        "index": int(index),
        "target": str(row.get("target") or row.get("neutral") or "")[:80],
        "positive": row["positive"],
        "neutral": row["neutral"],
        "attributes": list(row.get("attributes") or []),
        "n_held": int(hold["n_held"]),
        "n_free": int(hold["n_free"]),
        "n_concept": int(hold["n_concept"]),
        "expression_gap": expression_gap,
        "post_connector_video": post_video,
        "embed_gap_energy": energy,
        "hold": {
            "held_max_abs": float(hold["held_max_abs"]),
            "held_mean_abs": float(hold["held_mean_abs"]),
            "concept_mean_abs": float(hold["concept_mean_abs"]),
        },
        "student": student,
        "dead_gap": expression_gap_is_dead(expression_gap),
        "hold_stage": "pre_connector",
        "n_prompt_tokens": packed_plus.n_prompt_tokens,
        "n_connector_tokens": int(packed_plus.encoder_hidden_states.shape[1]),
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _aggregates(rows: list[dict[str, Any]]) -> dict[str, float | None | bool]:
    def _pick(path: tuple[str, ...]) -> list[float]:
        out: list[float] = []
        for row in rows:
            cur: Any = row
            for key in path:
                cur = cur[key]
            out.append(float(cur))
        return out

    cos_mean = _mean(_pick(("expression_gap", "cos")))
    return {
        "expression_gap_cos_mean": cos_mean,
        "expression_gap_l2_mean": _mean(_pick(("expression_gap", "l2"))),
        "post_connector_video_cos_mean": _mean(_pick(("post_connector_video", "cos"))),
        "post_connector_video_l2_mean": _mean(_pick(("post_connector_video", "l2"))),
        "dead_gap": bool(cos_mean is not None and cos_mean >= 0.999),
        "hold_max_abs_max": max(_pick(("hold", "held_max_abs"))) if rows else None,
        "hold_mean_abs_mean": _mean(_pick(("hold", "held_mean_abs"))),
        "concept_mean_abs_mean": _mean(_pick(("hold", "concept_mean_abs"))),
        "concept_gap_energy_frac_mean": _mean(_pick(("embed_gap_energy", "concept_frac"))),
        "held_gap_energy_frac_mean": _mean(_pick(("embed_gap_energy", "held_frac"))),
        "scale0_vs_teacher_neu_cos_mean": _mean(_pick(("student", "scale0_vs_teacher_neu", "cos"))),
        "scale1_vs_teacher_plus_cos_mean": _mean(_pick(("student", "scale1_vs_teacher_plus", "cos"))),
        "scale1_vs_scale0_cos_mean": _mean(_pick(("student", "scale1_vs_scale0", "cos"))),
        "neu_lora_on_vs_off_cos_mean": _mean(_pick(("student", "neu_lora_on_vs_off", "cos"))),
    }


def format_diag_table(summary: dict[str, Any]) -> str:
    loaded = summary.get("load_ltx_lora") or "none"
    header = (
        f"=== LTX-2.5 UNI diag (dummy={summary.get('dummy')}, "
        f"hold={summary.get('hold_mode')}, lora={loaded}) ==="
    )
    cols = (
        f"{'#':>2}  {'post_cos':>8}  {'expr_cos':>9}  {'expr_l2':>8}  "
        f"{'hold_max':>8}  {'cpt_abs':>7}  {'s0_neu':>6}  {'s1_plus':>7}  "
        f"{'s1_s0':>6}  {'dead':>5}"
    )
    lines = [header, cols]
    for row in summary.get("rows") or []:
        eg = row["expression_gap"]
        pc = row.get("post_connector_video") or {"cos": 0.0}
        hold = row["hold"]
        st = row["student"]
        lines.append(
            f"{row['index']:>2}  "
            f"{pc['cos']:>8.4f}  "
            f"{eg['cos']:>9.4f}  {eg['l2']:>8.4f}  "
            f"{hold['held_max_abs']:>8.4f}  {hold['concept_mean_abs']:>7.4f}  "
            f"{st['scale0_vs_teacher_neu']['cos']:>6.3f}  "
            f"{st['scale1_vs_teacher_plus']['cos']:>7.3f}  "
            f"{st['scale1_vs_scale0']['cos']:>6.3f}  "
            f"{str(row['dead_gap']):>5}"
        )
    agg = summary.get("aggregates") or {}
    if agg:
        lines.append(
            "mean expr_cos={:.4f} expr_l2={:.4f} dead_gap={}  "
            "fallback={}".format(
                agg.get("expression_gap_cos_mean") or 0.0,
                agg.get("expression_gap_l2_mean") or 0.0,
                agg.get("dead_gap"),
                FULL_TRANSFORMER_SUBFOLDER if agg.get("dead_gap") else "none",
            )
        )
    lines.append(
        "read: post_cos ~0.68 = working embed gap (train embed-match). "
        "expr_cos≈1 = dead DiT velocity teacher — do not train velocity UNI. "
        "hold_max~0 = identity tokens pinned PRE-connector."
    )
    return "\n".join(lines)


@torch.no_grad()
def run_diag(
    args: argparse.Namespace,
    backend: LTX25Backend | None = None,
) -> dict[str, Any]:
    rows = load_slider_rows(args.prompts_file, getattr(args, "attributes", "") or "")
    cap = getattr(args, "max_rows", None)
    if cap is None:
        cap = getattr(args, "sample_max_rows", None)
    if cap is not None:
        rows = rows[: max(0, int(cap))]
    backend = backend or build_backend(args)
    loaded = None
    if getattr(args, "load_ltx_lora", None):
        loaded = backend.load_trained(args.load_ltx_lora)
    diagnosed = [
        diagnose_row(
            backend, row,
            hold_mode=str(getattr(args, "hold_mode", DEFAULT_HOLD_MODE)),
            seed=int(args.seed),
            index=i,
        )
        for i, row in enumerate(rows)
    ]
    aggregates = _aggregates(diagnosed)
    summary = {
        "dummy": bool(getattr(args, "dummy", False)),
        "model_id": getattr(args, "model_id", DEFAULT_MODEL),
        "transformer_subfolder": getattr(args, "transformer_subfolder", DEFAULT_TRANSFORMER_SUBFOLDER),
        "hold_mode": str(getattr(args, "hold_mode", DEFAULT_HOLD_MODE)),
        "hold_stage": "pre_connector",
        "load_ltx_lora": loaded,
        "n_rows": len(diagnosed),
        "rows": diagnosed,
        "aggregates": aggregates,
        "how_to_read": HOW_TO_READ,
        "recipe": "ltx25_uni_diag",
        "dead_gap": bool(aggregates.get("dead_gap")),
        "transformer_full_fallback": FULL_TRANSFORMER_SUBFOLDER,
        "device": "cpu" if getattr(args, "dummy", False) else getattr(args, "device", "cuda:0"),
        "encoder_device": None if getattr(args, "dummy", False) else getattr(args, "encoder_device", DEFAULT_ENCODER_DEVICE),
    }
    if summary["dead_gap"]:
        print(
            f"DEAD GAP: plus/neu velocity cos ≈ 1.0. Do not silently train "
            f"distilled transformer/. Fallback: --transformer_subfolder "
            f"{FULL_TRANSFORMER_SUBFOLDER}."
        )
    save_dir = Path(args.save_dir or f"models/{getattr(args, 'name', 'ltx25-uni-diag')}")
    save_dir.mkdir(parents=True, exist_ok=True)
    name = getattr(args, "name", "ltx25-uni-diag")
    path = save_dir / f"{name}_diag.json"
    path.write_text(json.dumps(summary, indent=2))
    print(format_diag_table(summary))
    print(f"wrote {path}")
    return summary


def main(argv: list[str] | None = None) -> dict:
    return run_diag(parse_diag_args(argv))


if __name__ == "__main__":
    main()
