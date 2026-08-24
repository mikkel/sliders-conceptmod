# Script index

`scripts/` is the Music 3 operator toolbox. Prefer the pipeline for
recipe comparisons; use these one-offs for a shipped slider, an LM half,
or a diagnostic the pipeline does not own.

Hard constraints that apply to several tools:

- Use the `minimax-music3` conda env. Do not `pip install -r requirements.txt`.
- Paths and the pipeline interpreter are hardcoded to
  `/home/mikkel/anaconda3/envs/minimax-music3/bin/python` and
  `/ml2/music/sliders-conceptmod`.
- `--device` is **not** the same everywhere. `generate_listen.py` indexes
  `CUDA_VISIBLE_DEVICES`; `train_lora_music3.py` does
  `setdefault(CUDA_VISIBLE_DEVICES, "0")` then `cuda:{--device}`. Pipeline
  `--gpu N` already isolates one GPU and passes `--device 0`. Details in
  MUSIC3.md “GPU `--device` is not the same in every script”.
- Condition-interpolation and teacher-guidance diagnostics refuse
  durations that chunk (>8 s / 200 AR frames). Stay at 4 s or 8 s.
- Pipeline gates are **transformer-only**. Do not score LM halves with
  `slider_pipeline.py` or `slider_score.py`.

## Train, compare, adopt

| script | when | constraint |
|---|---|---|
| `slider_pipeline.py` | Paired recipe sweep on one frozen caption pair | TF only. Runbook: [slider_pipeline/README.md](../slider_pipeline/README.md). |
| `run_gpu0_sliders.sh` | Replay the **v2** energy / distortion / tempo / space / gender-lm family | Still `--lr 1e-4`. Does **not** train dust or v4 LM halves. |
| `compare_slider_runs.py` | Rank finished TF runs from `eval` blocks in `*_train.jsonl` | Probe `cos` does not predict renders. Accept on a ladder. |
| `eval_slider_probe.py` | Same probe, after the fact, for checkpoints with no `eval` block | Needs prompts + cache. Rank/alpha/targets come from the sidecar. |
| `compare_slider.py` | Print RMS / brightness / pulse for one listen folder | Single-folder diagnostic, not a ranking. |

```bash
$PY scripts/slider_pipeline.py selftest
$PY scripts/compare_slider_runs.py models/*/[a-z]*_train.jsonl --by_t
$PY scripts/eval_slider_probe.py models/triphop-tf-v3/*_attn_last.safetensors \
  --prompts_file conceptmod/textsliders/data/prompts-triphop-v3-single.yaml \
  --cache_dir cache/triphop-v3 --by_t
```

## Render

| script | when | constraint |
|---|---|---|
| `conceptmod/textsliders/generate_listen.py` | Neutral-caption scale ladder + REF caption swaps | `--accept_short` default **on**. `--accept_silent` is off unless the pipeline needs a collapse on disk. `--raw_scales` ignores sidecar `unit_scale`. |
| `render_shipped_slider.py` | Studio path: `scale × ratio × gain × unit_scale` as `app/` resolves it | Use this for catalog A/B, not face-value `generate_listen`. |
| `render_stack_sweep.py` | Mixed TF+LM stacks at a combined `sum(|multiplier|)` | Measured `combined_budget` (28.0, currently a no-op). |
| `render_slider_sweep.py` | Continuous mix of one shipped slider across takes | Studio-resolved scales. |
| `render_lora_ramp.py` | One generate; LoRA gain ramps over audio time | |
| `render_song_ramp.py` | One song; every shipped slider ramps in sequence | |
| `render_duet_lines.py` | Gender slider flips per lyric line | LM gender only. |
| `render_duet_song.py` | Gender slider alternates by section | LM gender only. |
| `render_end_ab.py` | Same-seed song-ending A/B (base vs LoRA) | LM ending / planreg work. |
| `render_cond_interp.py` | G0 axis: `cond(u) = (1−u)·neu + u·pole` | ≤8 s. Pipeline `onboard` calls this. |
| `render_teacher_guidance.py` | Live `v_neu + s·g·(v_pos − v_neg)` with no LoRA | Ceiling of the TF objective. `--both_branches` vs cond-only changes post-CFG net (1.0 vs 1.7). |

`generate_listen.py` is resume-safe. Existing wavs that pass the active
checks are skipped unless `--force`. Silent clips are rejected unless
`--accept_silent` (pipeline uses that + `--retry_seeds 0` so a collapse
stays on the requested seed).

## Probe and score

| script | when | constraint |
|---|---|---|
| `probe_axis.py` | Per-folder PASS/FAIL vs REF clips (F0 / onset / rms / crest) | Tempo estimator ranks even the REFs backwards; distortion REFs are indistinguishable — those axes say “use ears”. |
| `probe_lm_axis_signal.py` | Does the AR LM encode this axis at all? | Run **before** training an LM half. |
| `probe_render_direction.py` | Does the render move the same mixture as the folder’s REF swap? | Quote the scale whose **level** matches the swap, not ±2. |
| `score_render_curve.py` | Accept/reject from the render curve, not probe `cos` | Level error at the brightness-matched setting is the number that separates collapse from a working slider. |
| `slider_score.py` | Gated audible-effect scalar (SCORING.md) | TF ladders. Axis from cond-interp, not caption swaps. |
| `verify_listen.py` | Fail if a listen folder is missing / short / silent | Used by `run_gpu0_sliders.sh verify`. |

## Calibrate

| script | when | constraint |
|---|---|---|
| `calibrate_slider_scale.py` | Write `unit_scale` from 3-timestep velocity deltas | **Over-drives** these checkpoints (triphop +2 went near-silent). Do not trust it for shipping. |
| `calibrate_by_render.py` | Fit `rms(s) = rms(0)·exp(−k·s)` on a rendered ladder | The strength signal this project trusts. `--write` updates the sidecar. |
| `conceptmod/textsliders/calibrate_scale.py` | Trainer post-step sidecar write | No longer KeyErrors on `x0_*.pt` in a shared cache. |

## Listen sessions and indexes

| script | when |
|---|---|
| `build_listen_index.py` | `index.html` over a listen tree (pipeline `report` tries this). |
| `build_ab_session.py` | Blind 2AFC under `eval/listen/abtest/` from ladder folders. |
| `score_ab_session.py` | Score a session against ground truth + a candidate metric. |

## Blind-spot audit

| script | when |
|---|---|
| `blindspot_metrics.py` | Features a listener hears that rms/centroid/hi4k/flatness/crest miss. |
| `blindspot_whisper.py` | Whisper-based measures over render folders. |
| `blindspot_analyze.py` | Regenerate `scratchpad/blindspots.md` from the scan CSVs. |

## Maintenance

| script | when | constraint |
|---|---|---|
| `normalize_lora_keys.py` | Rewrite `lora_unet--x` double-delimiter keys to `lora_unet-x` | Pre-fix `--targets full` checkpoints. Writes in place after `.bak` unless `--out`. |
| `diag_slider_signal.py` | Compare condition/velocity/LoRA deltas (energy vs gender TF) | Diagnostic, not a gate. |

ComfyUI convert is **not** in this repo. Use
`scripts/convert_lora_comfyui.py` on [mikkel/conceptmod](https://github.com/mikkel/conceptmod)
`main` (MUSIC3.md “ComfyUI”).
