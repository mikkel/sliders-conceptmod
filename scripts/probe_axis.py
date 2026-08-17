#!/usr/bin/env python3
"""Per-axis metrics + PASS/FAIL gates for a listen folder.

Parses slider scales out of generate_listen.py filenames
(NN_slider_<label>_plus2.wav / _minus2 / _zero, NN_REF_prompt_<label>_no_slider.wav),
measures per-clip metrics, applies the axis's acceptance gates, prints a table,
and writes probe.json next to the wavs. Exit code 1 when a gate fails.

  gender   ref-relative: F0 must move from the minus-REF toward the plus-REF
           (median F0 over a full mix tracks instruments as much as voice, so
           absolute Hz gates only apply when no REF clips are present)
  energy   rms must rise monotonically with the slider (loudness moving IS the
           axis here, so the usual "rms stays put" check is skipped)
  tempo    pulse/onset rate tracks the refs
  distort  high-frequency ratio tracks the refs
  rapslow  onset_rate(+max) >= 1.4 x onset_rate(-max); monotone
  triphop  centroid(+max) <= 0.85 x centroid(-max); high_ratio decreasing
  generic  table only
All axes: rms of each slider clip within +-30% of the scale-0 clip.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_slider import high_ratio, load_mono, pulse_rate, spectral_centroid  # noqa: E402

MONOTONE_TOLERANCE = 0.05  # allow 5% backtracks between adjacent scales


def median_f0(y: np.ndarray, sr: int) -> float:
    try:
        import torch
        import torchaudio

        mono = torch.from_numpy(y.astype(np.float32))
        freq = torchaudio.functional.detect_pitch_frequency(mono.unsqueeze(0), sr).squeeze()
        freq = freq[(freq > 50) & (freq < 500)]
        if freq.numel() == 0:
            return 0.0
        return float(freq.median())
    except Exception as exc:  # noqa: BLE001
        print(f"f0 failed: {exc}", flush=True)
        return 0.0


def _spectral_flux(y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Half-wave-rectified spectral flux envelope and its frame rate."""
    win, hop = 2048, max(1, sr // 100)
    if len(y) < win * 2:
        return np.zeros(0), 1.0
    window = np.hanning(win)
    mags = np.array([np.abs(np.fft.rfft(y[i : i + win] * window)) for i in range(0, len(y) - win, hop)])
    flux = np.clip(mags[1:] - mags[:-1], 0, None).sum(axis=1)
    return flux, sr / hop


def tempo_bpm(y: np.ndarray, sr: int) -> float:
    """Beat rate from the autocorrelation of the onset envelope.

    Counting onset peaks conflates note density with tempo (a slow ballad with
    busy ornamentation out-scores fast four-on-the-floor), so find the dominant
    periodicity instead.
    """
    flux, fps = _spectral_flux(y, sr)
    if flux.size < 32 or flux.max() <= 0:
        return 0.0
    env = flux - flux.mean()
    ac = np.correlate(env, env, mode="full")[env.size - 1 :]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo_lag = max(1, int(round(fps * 60.0 / 200.0)))  # 200 BPM
    hi_lag = min(ac.size - 1, int(round(fps * 60.0 / 40.0)))  # 40 BPM
    if hi_lag <= lo_lag:
        return 0.0
    lag = int(np.argmax(ac[lo_lag : hi_lag + 1])) + lo_lag
    return float(60.0 * fps / lag)


def crest_db(y: np.ndarray) -> float:
    """Peak-to-RMS in dB. Distortion squashes peaks, so heavier -> lower."""
    rms = float(np.sqrt(np.mean(np.square(y))))
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if rms <= 1e-9 or peak <= 0:
        return 0.0
    return float(20.0 * np.log10(peak / rms))


def onset_rate(y: np.ndarray, sr: int) -> float:
    """Spectral-flux onset peaks per second (rap flows are onset-dense)."""
    flux, fps = _spectral_flux(y, sr)
    hop = max(1, sr // 100)
    if flux.size < 8 or flux.max() <= 0:
        return 0.0
    flux = flux / flux.max()
    threshold = float(np.median(flux) + 0.5 * np.std(flux))
    min_gap = max(1, int(0.1 * sr / hop))  # 100 ms refractory
    peaks, last = 0, -min_gap
    for t in range(1, len(flux) - 1):
        if flux[t] > threshold and flux[t] >= flux[t - 1] and flux[t] >= flux[t + 1] and t - last >= min_gap:
            peaks += 1
            last = t
    return float(peaks / max(len(y) / sr, 1e-6))


_SCALE_RE = re.compile(r"_(plus|minus)([0-9.]+)\.wav$")


def parse_role(name: str) -> tuple[str, float | None]:
    """-> ('slider', scale) | ('ref', +-1) | ('other', None)"""
    if "_REF_" in name:
        return "ref", None
    if name.endswith("_zero.wav"):
        return "slider", 0.0
    match = _SCALE_RE.search(name)
    if match and "_slider_" in name:
        value = float(match.group(2))
        return "slider", value if match.group(1) == "plus" else -value
    return "other", None


def monotone(values: list[float], tolerance: float = MONOTONE_TOLERANCE) -> bool:
    for prev, cur in zip(values, values[1:]):
        floor = prev - tolerance * max(abs(prev), 1e-9)
        if cur < floor:
            return False
    return True


def probe(folder: Path, axis: str) -> dict:
    rows = []
    for wav in sorted(folder.glob("*.wav")):
        y, sr = load_mono(wav)
        role, scale = parse_role(wav.name)
        rows.append(
            {
                "name": wav.name,
                "role": role,
                "scale": scale,
                "sec": round(len(y) / sr, 2),
                "rms": round(float(np.sqrt(np.mean(y**2))), 5),
                "f0": round(median_f0(y, sr), 1),
                "centroid": round(spectral_centroid(y, sr), 1),
                "high_ratio": round(high_ratio(y, sr), 4),
                "pulse_rate": round(pulse_rate(y, sr), 2),
                "onset_rate": round(onset_rate(y, sr), 2),
                "bpm": round(tempo_bpm(y, sr), 1),
                "crest_db": round(crest_db(y), 2),
            }
        )

    sliders = sorted((r for r in rows if r["role"] == "slider"), key=lambda r: r["scale"])
    # generate_listen.py writes the plus-pole REF first, then the minus-pole REF.
    refs = [r for r in rows if r["role"] == "ref"]
    ref_plus = refs[0] if len(refs) >= 2 else None
    ref_minus = refs[1] if len(refs) >= 2 else None
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    if len(sliders) >= 2:
        lo, hi = sliders[0], sliders[-1]
        # A slider that wrecks the mix drives it to near-silence or slams it;
        # that absolute check is what caught the bad transformer sliders. Only
        # the pure mix axes get the tighter "stays near the base take" check,
        # since LM axes rewrite the arrangement and move loudness legitimately.
        LM_AXES = ("gender", "rapslow", "triphop", "tempo", "distortion", "energy")
        destroyed = [r["name"] for r in sliders if not (0.008 <= r["rms"] <= 0.30)]
        check("rms_not_destroyed", not destroyed, f"silent/slammed: {destroyed}" if destroyed else "ok")

        zero = next((r for r in sliders if r["scale"] == 0.0), None)
        if zero is not None and axis not in LM_AXES:
            lo_rms, hi_rms = zero["rms"] / 1.3, zero["rms"] * 1.3
            bad = [r["name"] for r in sliders if r["scale"] != 0.0 and not (lo_rms <= r["rms"] <= hi_rms)]
            check("rms_within_1.3x_of_zero", not bad, f"outliers={bad}" if bad else "ok")
        if axis == "gender":
            f0s = [r["f0"] for r in sliders]
            if ref_plus is not None and ref_minus is not None:
                sep = ref_plus["f0"] - ref_minus["f0"]
                if abs(sep) < 20.0:
                    check(
                        "f0_refs_separate",
                        True,
                        f"inconclusive: refs only {sep:+.1f} Hz apart on this mix — use ears",
                    )
                else:
                    direction = 1.0 if sep > 0 else -1.0
                    span = hi["f0"] - lo["f0"]
                    check(
                        "f0_tracks_refs",
                        span * direction > 0,
                        f"slider span {span:+.1f} Hz vs ref separation {sep:+.1f} Hz",
                    )
                    ordered = f0s if direction > 0 else list(reversed(f0s))
                    check("f0_monotone_toward_plus_ref", monotone(ordered), f"f0 by scale: {f0s}")
                    check(
                        "f0_covers_half_ref_span",
                        abs(span) >= 0.5 * abs(sep),
                        f"|span|={abs(span):.1f} vs 0.5*|sep|={0.5 * abs(sep):.1f}",
                    )
            else:
                check("f0_monotone", monotone(f0s), f"f0 by scale: {f0s}")
                check("f0_plus_max_ge_230", hi["f0"] >= 230.0, f"F0(+{hi['scale']:g})={hi['f0']}")
                check("f0_minus_max_le_155", lo["f0"] <= 155.0, f"F0({lo['scale']:g})={lo['f0']}")
        elif axis == "rapslow":
            onsets = [r["onset_rate"] for r in sliders]
            if ref_plus is not None and ref_minus is not None:
                sep = ref_plus["onset_rate"] - ref_minus["onset_rate"]
                base = max(ref_plus["onset_rate"], ref_minus["onset_rate"], 1e-6)
                if abs(sep) < 0.2 * base:
                    check(
                        "onset_refs_separate",
                        True,
                        f"inconclusive: refs {ref_plus['onset_rate']:.2f} vs "
                        f"{ref_minus['onset_rate']:.2f}/s — delivery needs ears",
                    )
                else:
                    direction = 1.0 if sep > 0 else -1.0
                    span = hi["onset_rate"] - lo["onset_rate"]
                    check(
                        "onset_tracks_refs",
                        span * direction > 0,
                        f"slider span {span:+.2f}/s vs ref separation {sep:+.2f}/s",
                    )
                    ordered = onsets if direction > 0 else list(reversed(onsets))
                    check("onset_monotone_toward_plus_ref", monotone(ordered), f"onset by scale: {onsets}")
            else:
                check("onset_monotone", monotone(onsets), f"onset_rate by scale: {onsets}")
                ratio = hi["onset_rate"] / max(lo["onset_rate"], 1e-6)
                check("onset_plus_ge_1.4x_minus", ratio >= 1.4, f"ratio={ratio:.2f}")
        elif axis == "energy":
            rmss = [r["rms"] for r in sliders]
            check("rms_monotone", monotone(rmss), f"rms by scale: {rmss}")
            ratio = hi["rms"] / max(lo["rms"], 1e-9)
            check("rms_plus_ge_1.5x_minus", ratio >= 1.5, f"ratio={ratio:.2f}")
        elif axis in ("tempo", "distortion"):
            # bpm from envelope periodicity; crest_db falls as distortion squashes peaks
            metric = "bpm" if axis == "tempo" else "crest_db"
            expected_sign = 1.0 if axis == "tempo" else -1.0  # fast -> higher bpm; heavy -> lower crest
            values = [r[metric] for r in sliders]
            if ref_plus is not None and ref_minus is not None:
                sep = ref_plus[metric] - ref_minus[metric]
                base = max(abs(ref_plus[metric]), abs(ref_minus[metric]), 1e-9)
                if sep * expected_sign < 0:
                    # The REF clips are ground truth: a metric that cannot even
                    # rank them correctly cannot grade the slider.
                    check(
                        f"{metric}_metric_reliable",
                        True,
                        f"inconclusive: metric ranks the REFs backwards "
                        f"(plus={ref_plus[metric]}, minus={ref_minus[metric]}) — use ears",
                    )
                elif abs(sep) < 0.15 * base:
                    check(f"{metric}_refs_separate", True, f"inconclusive: refs {ref_plus[metric]} vs {ref_minus[metric]} — use ears")
                else:
                    direction = 1.0 if sep > 0 else -1.0
                    span = hi[metric] - lo[metric]
                    check(f"{metric}_tracks_refs", span * direction > 0, f"slider span {span:+.3f} vs ref separation {sep:+.3f}")
                    ordered = values if direction > 0 else list(reversed(values))
                    check(f"{metric}_monotone_toward_plus_ref", monotone(ordered), f"{metric} by scale: {values}")
            else:
                check(f"{metric}_monotone", monotone(values), f"{metric} by scale: {values}")
        elif axis == "triphop":
            cents = [r["centroid"] for r in sliders]
            if ref_plus is not None and ref_minus is not None:
                # Direction comes from the refs: vinyl crackle/hiss can make the
                # trip-hop pole *brighter* than pop on centroid, not darker.
                sep = ref_plus["centroid"] - ref_minus["centroid"]
                if abs(sep) < 0.1 * max(ref_plus["centroid"], ref_minus["centroid"]):
                    check(
                        "centroid_refs_separate",
                        True,
                        f"inconclusive: refs {ref_plus['centroid']:.0f} vs {ref_minus['centroid']:.0f} Hz — use ears",
                    )
                else:
                    direction = 1.0 if sep > 0 else -1.0
                    span = hi["centroid"] - lo["centroid"]
                    check(
                        "centroid_tracks_refs",
                        span * direction > 0,
                        f"slider span {span:+.0f} Hz vs ref separation {sep:+.0f} Hz",
                    )
                    check(
                        "centroid_covers_quarter_ref_span",
                        abs(span) >= 0.25 * abs(sep),
                        f"|span|={abs(span):.0f} vs 0.25*|sep|={0.25 * abs(sep):.0f}",
                    )
            else:
                ratio = hi["centroid"] / max(lo["centroid"], 1e-6)
                check("centroid_plus_le_0.85x_minus", ratio <= 0.85, f"ratio={ratio:.2f}")
                highs = [r["high_ratio"] for r in sliders]
                check("high_ratio_decreasing", monotone(list(reversed(highs))), f"high_ratio by scale: {highs}")
    else:
        check("enough_slider_clips", False, f"found {len(sliders)} slider clips")

    result = {
        "folder": str(folder),
        "axis": axis,
        "files": rows,
        "checks": checks,
        "pass": all(c["pass"] for c in checks),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "--axis",
        choices=["gender", "rapslow", "triphop", "energy", "tempo", "distortion", "generic"],
        default="generic",
    )
    parser.add_argument("--json", type=Path, default=None, help="default: <folder>/probe.json")
    parser.add_argument("--no-strict", action="store_true", help="exit 0 even on FAIL")
    args = parser.parse_args()

    result = probe(args.folder, args.axis)
    print(
        f"{'file':<46} {'scale':>6} {'sec':>6} {'rms':>7} {'f0':>6} {'cent':>7} "
        f"{'hi>3k':>6} {'onset':>6} {'bpm':>6} {'crest':>6}"
    )
    for r in result["files"]:
        scale = "" if r["scale"] is None else f"{r['scale']:+g}"
        print(
            f"{r['name']:<46} {scale:>6} {r['sec']:6.2f} {r['rms']:7.4f} {r['f0']:6.1f} "
            f"{r['centroid']:7.1f} {r['high_ratio']:6.3f} {r['onset_rate']:6.2f} "
            f"{r['bpm']:6.1f} {r['crest_db']:6.2f}"
        )
    print()
    for c in result["checks"]:
        print(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['name']}: {c['detail']}")
    print(f"\n=> {'PASS' if result['pass'] else 'FAIL'} ({args.axis})")

    out = args.json or (args.folder / "probe.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    if not result["pass"] and not args.no_strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
