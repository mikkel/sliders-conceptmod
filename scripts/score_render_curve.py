#!/usr/bin/env python3
"""Score a slider by where its render curve goes, not by velocity cosine.

The probe scores an adapter at x_t drawn from *unperturbed* trajectories, which
is exactly where an open-loop adapter looks fine; at inference it drives its own
trajectory and the error compounds. So acceptance has to be a render.

The concept axis is a mixture: the caption swap moves brightness a lot and level
a little. Sweeping the slider traces a curve through (delta rms, delta centroid)
relative to the ladder's own scale-0 clip. Ground truth is a single point --
REF_pos relative to REF_neu. A good slider's curve passes near that point; the
open-loop failure is a curve that dives in level long before it arrives in
brightness.

Reported per folder:
  * the ladder, as percentages against its own zero clip
  * the setting where brightness matches ground truth, and the LEVEL ERROR there
    (the number that separates a working slider from a collapsing one)

    python scripts/score_render_curve.py eval/listen/traj-4s/*/ \
      --refs eval/listen/teacher-4s
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np


def load(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768
    if ch > 1:  # interleaved stereo read as mono fabricates high-frequency content
        a = a.reshape(-1, ch).mean(1)
    return a, sr


def feats(a: np.ndarray, sr: int) -> dict[str, float]:
    N, H = 2048, 512
    idx = np.arange(0, max(1, len(a) - N), H)
    fr = np.stack([a[i:i + N] for i in idx]) * np.hanning(N)
    S = np.abs(np.fft.rfft(fr, axis=1)) + 1e-12
    fq = np.fft.rfftfreq(N, 1 / sr)
    return {
        "rms": float(a.std()),
        "centroid": float((S * fq).sum(1).mean() / S.sum(1).mean()),
        "hi4k": float((S[:, fq > 4000].sum(1) / S.sum(1)).mean()),
    }


def scale_of(stem: str) -> float | None:
    tag = stem.split("_")[-1]
    if tag == "zero":
        return 0.0
    if tag.startswith("minus"):
        return -float(tag[len("minus"):])
    if tag.startswith("plus"):
        return float(tag[len("plus"):])
    return None


def ladder(folder: Path, pattern: str) -> dict[float, dict[str, float]]:
    out: dict[float, dict[str, float]] = {}
    for wav in folder.glob(pattern):
        s = scale_of(wav.stem)
        if s is not None:
            out[s] = feats(*load(wav))
    return out


def pct(cur: dict[str, float], base: dict[str, float], key: str) -> float:
    return (cur[key] - base[key]) / base[key] * 100.0


def crossing(points: list[tuple[float, float]], target: float) -> float | None:
    """First scale where the (monotone-ish) curve reaches `target`, linearly."""
    for (s0, y0), (s1, y1) in zip(points, points[1:]):
        if (y0 - target) * (y1 - target) <= 0 and y1 != y0:
            return s0 + (target - y0) * (s1 - s0) / (y1 - y0)
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--refs", type=Path, required=True,
                    help="folder holding REF_neu.wav and REF_pos.wav at the SAME duration "
                    "and seed -- levels are not comparable across durations")
    ap.add_argument("--pattern", default="*slider*.wav")
    args = ap.parse_args(argv)

    neu = feats(*load(args.refs / "REF_neu.wav"))
    pos = feats(*load(args.refs / "REF_pos.wav"))
    gt = {k: pct(pos, neu, k) for k in ("rms", "centroid", "hi4k")}
    print(f"ground truth (caption swap, neutral -> + pole, {args.refs.name}): "
          f"rms {gt['rms']:+.1f}%  centroid {gt['centroid']:+.1f}%  hi4k {gt['hi4k']:+.1f}%\n")

    rows = []
    for folder in args.folders:
        if not folder.is_dir():
            print(f"missing {folder}")
            continue
        lad = ladder(folder, args.pattern)
        if 0.0 not in lad:
            print(f"{folder.name}: no zero clip")
            continue
        base = lad[0.0]
        print(f"=== {folder.name} ===")
        print(f"  {'scale':>7} {'rms':>9} {'centroid':>10} {'hi4k':>9}")
        curve = []
        for s in sorted(lad):
            d = {k: pct(lad[s], base, k) for k in ("rms", "centroid", "hi4k")}
            print(f"  {s:7.3f} {d['rms']:+8.1f}% {d['centroid']:+9.1f}% {d['hi4k']:+8.1f}%")
            curve.append((s, d))
        pts_c = [(s, d["centroid"]) for s, d in curve]
        s_star = crossing(pts_c, gt["centroid"])
        if s_star is None:
            reach = max(d["centroid"] for _, d in curve)
            print(f"  brightness never reaches ground truth (+{gt['centroid']:.1f}%); "
                  f"tops out at {reach:+.1f}%\n")
            rows.append((folder.name, None, None))
            continue
        pts_r = [(s, d["rms"]) for s, d in curve]
        rms_at = np.interp(s_star, [s for s, _ in pts_r], [y for _, y in pts_r])
        err = rms_at - gt["rms"]
        print(f"  brightness-matched at scale {s_star:.3f}: rms {rms_at:+.1f}% "
              f"(ground truth {gt['rms']:+.1f}%, level error {err:+.1f} pts)\n")
        rows.append((folder.name, s_star, err))

    if len(rows) > 1:
        print(f"{'run':34s} {'match scale':>11s} {'level error':>12s}")
        for name, s, err in sorted(rows, key=lambda r: abs(r[2]) if r[2] is not None else 1e9):
            if err is None:
                print(f"{name:34s} {'-':>11s} {'never bright':>12s}")
            else:
                print(f"{name:34s} {s:11.3f} {err:+11.1f}p")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
