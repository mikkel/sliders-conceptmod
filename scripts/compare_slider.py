#!/usr/bin/env python3
"""Print RMS / brightness / pulse stats for a listen folder.

Not a gender classifier. Use it to check that energy, distortion, and tempo
move in the expected direction on long clips.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True)
    return audio.mean(axis=1).astype(np.float64), int(sample_rate)


def spectral_centroid(y: np.ndarray, sr: int) -> float:
    n = int(2 ** np.floor(np.log2(max(len(y), 2))))
    spec = np.abs(np.fft.rfft(y[:n] * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    denom = spec.sum() + 1e-12
    return float((freqs * spec).sum() / denom)


def high_ratio(y: np.ndarray, sr: int, cut: float = 3000.0) -> float:
    n = int(2 ** np.floor(np.log2(max(len(y), 2))))
    spec = np.abs(np.fft.rfft(y[:n] * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    return float(spec[freqs >= cut].sum() / (spec.sum() + 1e-12))


def pulse_rate(y: np.ndarray, sr: int) -> float:
    hop = max(1, sr // 100)
    win = max(hop * 4, sr // 20)
    env = np.array(
        [np.sqrt(np.mean(np.square(y[i : i + win]))) for i in range(0, max(len(y) - win, 1), hop)]
    )
    if env.size < 8:
        return 0.0
    env = env - np.median(env)
    env = np.clip(env, 0, None)
    if env.max() <= 0:
        return 0.0
    env = env / env.max()
    peaks = (env[1:-1] > env[:-2]) & (env[1:-1] > env[2:]) & (env[1:-1] > 0.35)
    return float(peaks.sum() * (sr / hop) / max(len(y) / sr, 1e-6))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for folder in args.dirs:
        print(f"\n== {folder} ==")
        print(f"{'file':<42} {'sec':>6} {'rms':>7} {'centHz':>8} {'hi>3k':>7} {'pulse/s':>8}")
        for wav in sorted(folder.glob("*.wav")):
            y, sr = load_mono(wav)
            print(
                f"{wav.name:<42} {len(y)/sr:6.2f} {np.sqrt(np.mean(y**2)):7.4f} "
                f"{spectral_centroid(y, sr):8.0f} {high_ratio(y, sr):7.3f} "
                f"{pulse_rate(y, sr):8.2f}"
            )


if __name__ == "__main__":
    main()
