#!/usr/bin/env python3
"""Fail if a listen folder is missing files, too short, or silent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

MIN_RMS = 1e-3


def inspect(path: Path) -> tuple[float, float]:
    audio, sample_rate = sf.read(str(path), always_2d=True)
    duration = float(audio.shape[0]) / float(sample_rate)
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
    return duration, rms


def check_dir(folder: Path, min_duration: float, expect: int) -> list[str]:
    errors: list[str] = []
    wavs = sorted(folder.glob("*.wav"))
    if len(wavs) < expect:
        errors.append(f"{folder}: {len(wavs)} wavs, expected >= {expect}")
    if not (folder / "LISTEN.md").exists():
        errors.append(f"{folder}: missing LISTEN.md")
    for wav in wavs:
        try:
            duration, rms = inspect(wav)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{wav}: unreadable ({exc})")
            continue
        if duration < min_duration:
            errors.append(f"{wav.name}: {duration:.2f}s < {min_duration:.2f}s")
        if rms < MIN_RMS:
            errors.append(f"{wav.name}: silent rms={rms:.6f}")
        print(f"ok {duration:6.2f}s rms={rms:.4f} {folder.name}/{wav.name}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--min-duration", type=float, default=7.5)
    parser.add_argument("--expect", type=int, default=5)
    args = parser.parse_args()
    errors: list[str] = []
    for folder in args.dirs:
        if not folder.is_dir():
            errors.append(f"missing directory: {folder}")
            continue
        errors.extend(check_dir(folder, args.min_duration, args.expect))
    if errors:
        print("VERIFY FAILED", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        return 1
    print("VERIFY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
