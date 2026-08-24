# Engineering notes (Music 3 sliders)

Scan-first pages for operators. Campaign narrative stays in
[MUSIC3.md](../MUSIC3.md); the gate contract is [SCORING.md](../SCORING.md).

| page | what it answers |
|---|---|
| [2d-analysis.md](2d-analysis.md) | Do SD / Music 3 TF / LM losses track the slider axis? When do `--attributes` cancel leak? CPU only. |
| [tf-leak.md](tf-leak.md) | Is the default TF leak a trainer bug? No — energy / distortion captions put BPM in `pos − neg`. Do not add `--attributes` to TF. |
| [scripts.md](scripts.md) | Which `scripts/` entrypoint to run (render, probe, calibrate, pipeline). |

CPU suites (no Hub, no GPU, no Music 3 weights):

```bash
PYTHONPATH=. pytest tests/test_2d_slider_geometry.py tests/test_tf_leak.py -q
PYTHONPATH=. python analysis/slider2d/run_analysis.py --out docs/2d-analysis
PYTHONPATH=. python analysis/tf_leak/run_leak.py --out docs/tf-leak
```

Shared loss extract: `conceptmod/textsliders/slider_targets.py`.
`train_lora_music3.py` imports `music3_slider_loss` from there.
