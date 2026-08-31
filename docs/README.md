# Docs index

Operator map for this repo. Train cards live next to the trainer that
owns them. Do not mix Music 3 defaults into an image/video env (and
the reverse).

## Start here

| if you need | read |
|---|---|
| Music 3 trainer defaults, shipped sliders, GPU pitfalls | [MUSIC3.md](../MUSIC3.md) |
| TF-only gate contract (G0–G7) | [SCORING.md](../SCORING.md) |
| paired recipe-comparison runbook | [slider_pipeline/README.md](../slider_pipeline/README.md) |
| listen folders / play order | [eval/listen/README.md](../eval/listen/README.md) |
| which `prompts-*.yaml` to pass | [prompts.md](prompts.md) |

The pipeline scores **transformer sliders only**. A working LM half
changes the arrangement on purpose; do not gate LM checkpoints with
it.

## Opt-in image / video trainers

Each backend is a separate script. None of them change
`train_lora_music3.py` / `train_lm_slider_music3.py` (`--lm_target v9`,
`--pole_mode hidden`). `--dummy` is the CI path: no Hub, no GPU.

| backend | trainer | card | live box | student +1 caption |
|---|---|---|---|---|
| **Krea** (Raw → Turbo) | `train_lora_krea.py` | [krea-slider.md](krea-slider.md) | A100 (~48–80 GB) | plus (velocity) or neu (TE embed UNI) |
| **Anima** (2B DiT) | `train_lora_anima.py` | [anima-slider.md](anima-slider.md) | 4090 / 512 px | **neu** (plus is teacher) |
| **Sana** 0.6B | `train_lora_sana.py` | [sana-slider.md](sana-slider.md) | cheap GPU (Modal / RunPod) | **neu** |
| **Z-Image Turbo** | `train_lora_zimage.py` | [zimage-slider.md](zimage-slider.md) | 6B DiT box | **neu** |
| **LTX-2.5** video | `train_lora_ltx25.py` | [ltx25-slider.md](ltx25-slider.md) | **A100 80GB**, TE on CPU. Not 4090 / B300 | **neu** (hold PRE-connector) |
| **MiniMax-H3** t2va | `train_lora_minimax_h3.py` | [minimax-h3-slider.md](minimax-h3-slider.md) | **B200 / B300** (~135 GB); 2×H100 needs `--encoder_device` | plus pack (velocity UNI) |

Shared pitfalls (verified in the trainers, not folklore):

- **Zero-init LoRA-up** is UNI identity on H3 / LTX (`loss 0.0000`).
  Default is `N(0, 0.02)`.
- **Train +1 on the plus caption, infer on neu** misses the concept
  (Sana age dud / H3 caption coupling). Sana / ZiT / LTX / Anima train
  +1 on **neu**.
- **`--hold_mode attributes`** on H3 / LTX only pins yaml unused
  tokens. Shared subject tokens stay free and the Omni/DiT LoRA
  rewrites clothes / props. Default is `non_concept`.
- **Do not** `pip install -r requirements.txt` into the Music 3 env.
  H3 on B300 needs `torch 2.13.0+cu130`; LTX needs a git Diffusers
  with LTX-2.5. Those wheels stay on their boxes.
- PEFT `set_adapter_scale` often no-ops (Krea #74). Continuous
  scales write `LoraLayer.scaling = (alpha/r) * scale`.

## Music 3 target formulas

CPU-pure copies of the live losses live in
`conceptmod/textsliders/slider_targets.py`. The module docstring is
the contract: Music 3 `--lm_target` recipes (`v9`, `pair_odd_sub_e`,
`faithful_*`), SD enhance/erase, and the image UNI helpers (ZiT / Krea
embed). 2-D fixture write-ups that import it:

| page | what it grades |
|---|---|
| [2d-analysis.md](2d-analysis.md) | method table on a synthetic energetic×gender field |
| [tf-leak.md](tf-leak.md) | energy/distortion caption leak (BPM in `pos−neg`) |
| [lm-live-cells.md](lm-live-cells.md) | live `v9` / hold-ê cells |
| [lm-2d-scoreboard.md](lm-2d-scoreboard.md) | compiled 2-D / high-D / sheet board |
| [lm-sheet-goodhart.md](lm-sheet-goodhart.md) | why `c+` / `collapse` are not the success metric |
| [lm-highd-leftover.md](lm-highd-leftover.md) | leftover mix under live width |
| [lm-roles.md](lm-roles.md) | `faithful_plus_neu_roles` role split |
| [lm-lyric-hold.md](lm-lyric-hold.md) / [lm-lyric-orth.md](lm-lyric-orth.md) / [lm-lyric-recall.md](lm-lyric-recall.md) | lyric-span UNI variants |
| [lm-plus-exam.md](lm-plus-exam.md) / [lm-plus-neu-exam.md](lm-plus-neu-exam.md) / [lm-pair-exam.md](lm-pair-exam.md) | plus-only / plus+neu / pair exams |
| [lm-even-leftover.md](lm-even-leftover.md) / [lm-hold-overlap.md](lm-hold-overlap.md) | even leftover / hold overlap |
| [lm-v9-2d.md](lm-v9-2d.md) / [lm-v9-mismatch.md](lm-v9-mismatch.md) / [lm-faithful-2d.md](lm-faithful-2d.md) / [lm-rich-2d.md](lm-rich-2d.md) | older 2-D cells |

Default LM recipe is still `--lm_target v9` / `--pole_mode hidden`.
The `faithful_*` and `semantic_kl` cards are opt-in — see MUSIC3.md
“Current LM trainer defaults”.
