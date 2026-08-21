#!/usr/bin/env python3
"""Render CONDITION-space interpolation between neutral and a pole.

The teacher-4s renders show that fractional-strength velocity-space composition
craters the level: axis at post-CFG net ~1.0 renders -54% rms, and even the
honest pole edge at post-CFG 0.85 renders -53% -- while at the full caption-swap
identity point both land exactly on their REF. So the path a velocity slider
sweeps between neutral and a pole passes through a quiet basin that neither
endpoint occupies.

This script asks whether the CONDITION tensor is a better axis: render with
cond(u) = (1-u)*cond_neu + u*cond_pole fed to the conditional CFG branch, for a
ladder of u. If the feature curves (rms/centroid/hi4k) move monotonically toward
the pole's REF as u goes 0 -> 1, then smooth intermediate strengths exist in
condition space and the trainer's intermediate-strength targets should be built
from interpolated-condition teachers (or the slider should live upstream, e.g.
an LM/condition-encoder slider). If it craters the same way, the prompt pair
itself is the problem. Diagnostic only; does not ship.

    python scripts/render_cond_interp.py --pole pos --us 0,0.25,0.5,0.75,1 \
      --out_dir eval/listen/condmix-4s
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/ml2/music/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import soundfile as sf
import torch

_ROOT = Path("/ml2/music/sliders-conceptmod")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from conceptmod.textsliders.infer_music3 import _load_pipeline, _to_wav_array  # noqa: E402
from conceptmod.textsliders.train_lora_music3 import (  # noqa: E402
    build_conditions,
    load_prompts,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts_file", type=Path,
                    default=_ROOT / "conceptmod/textsliders/data/prompts-triphop-v3-single.yaml")
    ap.add_argument("--cache_dir", type=Path, default=_ROOT / "cache/triphop-v3")
    ap.add_argument("--model_dir", type=Path, default=Path("/ml2/music/models/MiniMax-Music3"))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--pole", choices=["pos", "neg"], default="pos")
    ap.add_argument("--us", default="0,0.25,0.5,0.75,1",
                    help="interpolation weights; u=0 is neutral, u=1 the pole caption's cond")
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args(argv)

    device = f"cuda:{int(args.device)}"
    prompts, _ = load_prompts(args.prompts_file)
    prompt = prompts[0]
    entries = build_conditions(
        prompts=[prompt], cache_dir=args.cache_dir, duration=args.duration,
        seeds=[args.seed], device=torch.device(device), model_dir=args.model_dir,
        skip_ar=True, dummy=False,
    )
    conds = entries[0][1]
    pipe = _load_pipeline(args.model_dir, device)
    dtype = next(pipe.transformer.parameters()).dtype
    cond_neu = conds["neutral"].to(device=device, dtype=dtype)
    cond_pole = conds["positive" if args.pole == "pos" else "negative"].to(device=device, dtype=dtype)
    n = min(cond_neu.shape[1], cond_pole.shape[1])
    cond_neu, cond_pole = cond_neu[:, :n], cond_pole[:, :n]

    original_forward = pipe.transformer.forward
    state = {"u": 0.0, "swapped": 0}

    def mixed_forward(hidden_states, timestep, encoder_hidden_states, return_dict=True):
        # Replace only the conditional branch (nonzero condition) with the mix,
        # cropped/expanded to the incoming length. The uncond zeros pass through,
        # so CFG treats the mix exactly as it would treat a real caption's cond.
        if encoder_hidden_states.abs().any():
            u = state["u"]
            m = min(n, encoder_hidden_states.shape[1])
            mix = (1.0 - u) * cond_neu[:, :m] + u * cond_pole[:, :m]
            if mix.shape[0] != hidden_states.shape[0]:
                mix = mix.expand(hidden_states.shape[0], -1, -1)
            if mix.shape[1] != encoder_hidden_states.shape[1]:
                pad = encoder_hidden_states[:, mix.shape[1]:]
                mix = torch.cat([mix, pad], dim=1)
            encoder_hidden_states = mix
            state["swapped"] += 1
        return original_forward(hidden_states, timestep, encoder_hidden_states, return_dict=return_dict)

    pipe.transformer.forward = mixed_forward
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sr = int(pipe.sampling_rate)
    for u in [float(x) for x in args.us.split(",") if x.strip()]:
        state["u"], state["swapped"] = u, 0
        gen = torch.Generator(device).manual_seed(int(args.seed))
        print(f"rendering cond-mix pole={args.pole} u={u:g}", flush=True)
        audio = pipe(prompt=prompt.neutral, lyrics=prompt.lyrics,
                     audio_duration=float(args.duration), generator=gen, output="audios")[0]
        wav = _to_wav_array(audio)
        tag = "zero" if u == 0 else f"plus{u:g}"
        dest = args.out_dir / f"condmix_{args.pole}_{tag}.wav"
        sf.write(str(dest), wav, sr)
        rms = float(np.asarray(wav, dtype=np.float64).std())
        print(f"  wrote {dest.name}  rms={rms:.4f}  swapped={state['swapped']}", flush=True)
    pipe.transformer.forward = original_forward
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
