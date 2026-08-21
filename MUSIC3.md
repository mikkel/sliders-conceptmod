# MiniMax Music 3 concept sliders

Trained weights and 20-second listening examples are published to the Hub:
**[ntc-ai/minimax-music3-concept-sliders](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders)**
(`weights/` mirrors `models/`, `samples/` mirrors `eval/listen/`). They are kept out
of git because they run to ~680 MB.

Use the `minimax-music3` conda env. **Never** `pip install -r requirements.txt` (it pins ancient torch/diffusers).

GPU 0 only:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/ml2/music/sliders-conceptmod
export HF_HUB_OFFLINE=1
export HF_HOME=/ml2/music/.cache/huggingface
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
```

## What works

| slider | LoRA host | 8s | 20s | last-step cos / residual |
|---|---|---|---|---|
| **energy** (quiet ↔ loud) | flow transformer `MiniMaxMusic3Attention` | `eval/listen/energy/` | `eval/listen/energy-20s/` and `energy-30s/` | cos ~0.43 |
| **distortion** (acoustic ↔ metal) | transformer | `eval/listen/distortion/` | `eval/listen/distortion-20s/` | cos ~0.43 |
| **tempo** (slow ↔ fast) | transformer | — | `eval/listen/tempo-20s/` | cos ~0.47 |
| **space** (dry ↔ wet) | transformer | — | `eval/listen/space-20s/` | cos ~0.42 |
| **gender** (male ↔ female) | **language model** `Qwen3Attention` | `eval/listen/gender-lm/` | `eval/listen/gender-lm-20s/` | residual ~5%, cos ~1.0 |

v3 adds language-model halves for energy, tempo and distortion, so those axes
now change the *arrangement* (how hard the band plays, the composed BPM, which
instruments are used) as well as the mix. See "Which host does an axis belong
to?" below.

A transformer LoRA **cannot** change singer gender. That is decided in the AR language model. `eval/listen/gender/` is the failed transformer attempt.

Working recipe: rank 8, alpha 8, 500 steps, lr `1e-4`, train duration 4s (AR cache). Generation duration is independent of train duration.

## v3 recipe (August 2026)

Four trainer bugs were found and fixed; v3 sliders (`models/*-v3/`) use:

1. **On-manifold x_t** — the model's flow time runs 0 = noise → 1 = clean
   (`denoise.py`), but the old trainer fed pure `randn` at every t. v3 anchors
   `x_t = (1−t)·ε + t·x0` to clean latents generated per condition
   (`--xt_mode anchor`, cached as `cache/<name>/x0_*.pt`; `noise` = legacy A/B).
2. **Bidirectional loss** (`--bidirectional`, default on) — the −1 direction is
   trained explicitly against `vel_neu − g·(vel_pos − vel_neg)`; per-step
   `cos_pos` / `cos_neg` / `collapse` are logged.
3. **Symmetric LM targets** (`train_lm_slider_music3.py --symmetric`, default
   on) — targets are antisymmetrized around neutral: `tgt(±1) = neu ± (pos−neg)/2`.
   This fixed the collapse metric from **+0.53** (gender-lm v2: both directions
   pointed the same way) to **−0.97** (gender-lm-v3: near-opposite).
   `--common_beta` blends the shared component back in if ±1 sounds weak.
4. **Module-identity dedupe in `lora.py`** — `--targets full` used to wrap
   every attention/FF Linear **twice** (438 modules with two name prefixes,
   2× gain). Now 222 unique modules. Old double-wrapped checkpoints
   (gender-tf-v2) still load with their own sidecars; never recalibrate them.

**Recipe A/B (triphop TF, what actually matters):**

| x_t | rows | steps | cos | cos_axis | verdict |
|---|---|---|---|---|---|
| noise (v2) | single | 500 | 0.46 | — (unit 1.82) | old baseline, mild but real |
| **anchor** | **single** | 250 | 0.45 | 0.41 | **matches v2 at half the steps** |
| noise | multi (4×2 seeds) | 250 | 0.16 | 0.08 | axis lost |
| anchor | multi (4×2 seeds) | 500 | 0.25 | 0.05 | axis lost |

So: the x0 anchor speeds transformer-slider convergence, but **multi-row /
attributes averaging destroys transformer style sliders** — a rank-8 attention
LoRA cannot find one shared "trip-hop direction" across diverse conditions.
Use multi-row + attributes for **LM** sliders (they thrive: gender-lm-v3
collapse −0.97, rapslow-lm-v3 −0.996, triphop-lm-v3 −0.99) and single-row for
**transformer** sliders. Style/delivery axes (gender, rap↔slow) belong in the
LM entirely — the rapslow transformer attempt calibrated at cos_axis 0.08
(`axis_tracking_low`), same failure mode as gender-tf.

**Shipped triphop TF is now `triphop-tf-v4`** (August 2026): a clean retrain of
the anchor/single cell from `prompts-triphop-v3-single.yaml` after the v2
baseline's prompt was found to contain artist names (weights quarantined in
`models/retired-triphop-slider-artist-names/`, banned by AGENTS.md). Two
scaling facts learned shipping it:

- The attn-target checkpoint has ~1.9x lower gain per multiplier than the old
  v2 `full` one, so calibration lands at unit_scale 3.6 — above the
  `min(unit_scale, 2.0)` cap in `app/sliders.py`, which silently truncates
  normalization for exactly the checkpoints that need it most. Worse, the
  calibration target itself ("one trained concept" from 3-timestep velocity
  deltas on a 4s clip) over-drives this checkpoint: at face-value settings the
  +2 render is near-silent (rms 0.003) even though its applied delta is
  *smaller* than the v2 baseline's. Velocity deltas do not predict 50-step
  20s renders; the shipped-path render (`scripts/render_shipped_slider.py`)
  is the only strength signal to trust.
- Fix that shipped: the v4 checkpoint's `.alpha` tensors and sidecar are
  de-rated 8.0 -> 1.0 (an empirical 8x cut sized by render RMS against the old
  shipped baseline), recalibrated to unit_scale 31.1, ratio 1.0 in
  sliders.json. Note the safetensors filename still says `alpha8.0`.
  (Superseded — the `min(unit,2)` cap this originally leaned on is gone; see
  "Every shipped LoRA is unit-normalized" below.)

## Every shipped LoRA is unit-normalized (August 2026)

A checkpoint must **drop in at strength 1**. The old runtime formula carried
normalization at inference time — `scale x ratio x gain x min(unit_scale, 2)` —
and that `min(..., 2)` was a silent lie: it truncated exactly the checkpoints
whose calibration said they needed the most correction (triphop-tf-v4 at
unit 31.1), so the file on disk and the sound in the studio disagreed.

The fix is to bake the runtime factor into the file. For each shipped
component, `B = gain x min(unit_scale, 2.0)` — the exact number the old code
would have applied. `B` was 1.0 for 12 of the 14 shipped components; the two
that needed it got a **new** file beside the original (originals untouched):

| axis | new file | B | `.alpha` | sidecar `alpha` | sidecar `unit_scale` |
|---|---|--:|---|---|---|
| triphop TF | `triphop-tf-v4/triphop-tf-v4_unit_last.safetensors` | 2.0 | 1.0 -> 2.0 | 1.0 -> 2.0 | 31.109 -> 1.0 |
| energy TF | `energy-slider-v2/energy_unit_last.safetensors` | 2.0 | 8.0 -> 16.0 | 8.0 -> 16.0 | 2.010 -> 1.0 |

Both halves have to move together: `app/lora_runtime.refresh_scales()` derives
each module's scale from the checkpoint's `.alpha` **tensors**, while
`conceptmod/textsliders/lora.py` derives it from the **sidecar** `alpha` at
construction and never re-reads the buffers. Bake one and not the other and the
studio and the eval scripts drift apart silently.

`min(unit, 2.0)` is now deleted from `app/sliders.py`; the formula is plain
`scale x ratio x gain x unit_scale`, and a resolved component whose `unit_scale`
is not 1.0 (±1e-3) logs a loud warning. `ratio` stays a recipe knob — energy's
transformer half still rides at 0.5.

Applied strength is unchanged by the bake, by construction (multiplier /= B,
alpha *= B) and by measurement: re-rendering triphop through
`scripts/render_shipped_slider.py` off the normalized file reproduces
`eval/listen/triphop-v4-shipped-20s/` to five decimals (0.1474 / 0.0975 /
0.0718 at -2 / 0 / +2).

### Combined budget: measured, and a no-op at today's catalog

With normalization in the files, the only runtime knob left is a cap on how
much slider a user can stack at once. `app/sliders.json` takes an optional
top-level `combined_budget`: when `sum(|multiplier|)` exceeds it, every
multiplier is scaled by the same `budget / total`, preserving the balance the
user dialled in; below it nothing changes at all.

Its value was **measured**, not guessed, with `scripts/render_stack_sweep.py`
(20s, fixed neutral caption + lyrics, rms/crest via the `probe_axis.py`
convention). Escalating mixed TF+LM stacks with alternating signs, plus the
worst case of all ten sliders at ±2 (which totals **27.0**, the maximum
reachable — triphop/distortion/tempo contribute 2 each and energy 1.5):

| total | rms | x base | crest dB | d crest | stack |
|--:|--:|--:|--:|--:|---|
| 0.0 | 0.09288 | 1.000 | 20.15 | 0.00 | (neutral baseline) |
| 2.0 | 0.13827 | 1.489 | 17.19 | -2.96 | energy+1, space-0.5 |
| 4.0 | 0.11833 | 1.274 | 18.54 | -1.61 | + distortion-1 |
| 6.0 | 0.18344 | 1.975 | 14.73 | -5.42 | energy+1, distortion-1, tempo+1, gender+0.5 |
| 8.0 | 0.12114 | 1.304 | 17.88 | -2.27 | + triphop+1 |
| 10.0 | 0.08476 | 0.913 | 20.22 | +0.07 | energy+1.5, distortion-1, tempo+1, triphop+1, breath+1, gender-0.75 |
| 12.0 | 0.12864 | 1.385 | 16.83 | -3.32 | energy+2, distortion-1.5, tempo+1, triphop+1, breath+1, live-1 |
| 16.0 | 0.06570 | 0.707 | 19.83 | -0.32 | + gender+1, tempo/triphop to 1.5, distortion-2 |
| 20.0 | 0.06757 | 0.727 | 19.44 | -0.71 | 9 sliders, mixed signs, most at ±2 |
| 27.0 | 0.09141 | 0.984 | 19.74 | -0.41 | **all ten at +2** |
| 27.0 | 0.07762 | 0.836 | 17.16 | -2.99 | **all ten at -2** |

Second caption + seed (energy-v3 row 0, seed 23; baseline is a hot 0.16485 /
15.45 dB) at four points: total 8 -> 0.10130 (0.61x, crest +2.92), total 20 ->
0.07873 (0.48x, +2.71), all +2 -> 0.11990 (0.73x, +0.89), all -2 -> 0.09808
(0.60x, +4.19).

**There is no knee.** Across all 16 renders absolute rms stays in
0.0657-0.1834 (a 2.8x spread, and 22x above the rms-0.003 near-silence that the
un-normalized triphop produced) and crest stays in 14.7-20.2 dB — never
collapsed. Degradation does not track the total: the *worst* rms excursion
(1.975x) is at total **6**, driven by energy being a loudness axis doing its
job, and the ±2 extremes at total 27 land within 2% and 17% of baseline. The
apparent 0.48x at caption B is an artifact of that caption's unusually loud
baseline, and its crest goes *up* — less squashed, not degraded.

So `combined_budget` ships at **28.0**: above the 27.0 maximum reachable today,
i.e. a deliberate **no-op guard**. Nothing a user can do to the current ten
sliders is budgeted. It exists so the mechanism is in place and tested, and it
starts biting automatically as sliders are added — re-run the sweep and
re-derive it when the catalog grows.

Also: `calibrate_scale.py` `_load_cache_entries` KeyErrors on the `x0_*.pt`
anchor files in a shared cache dir (it expects only condition entries) —
work around with a scratch dir holding just the condition files.

## Which host does an axis belong to?

Don't guess — measure first with `scripts/probe_lm_axis_signal.py`, which encodes
each pole's caption through the AR language model and reports how far the plan
moves (`sep = ||pos-neg|| / ||neu||`) and how antisymmetric that move is (`cos`
near 0 is good; high `cos` means both poles shift the same shared way).

| axis | sep | cos | host |
|---|---|---|---|
| rap ↔ slow | 0.32 | 0.04 | LM only (transformer tried: cos_axis 0.08) |
| energy | 0.28 | −0.10 | **both** |
| distortion | 0.23 | 0.03 | **both** |
| tempo | 0.21 | 0.34 | **both** |
| gender | 0.20 | 0.03 | LM only (transformer moves F0 by 1 Hz) |
| trip-hop | 0.20 | 0.28 | **both** |
| space (dry ↔ wet) | 0.17 | **0.70** | transformer only |

Gender works at sep 0.20, so that is roughly the usable floor. Space is the one
axis that is genuinely transformer-only: its poles move the plan mostly in the
*same* direction, i.e. the LM encodes "this caption is about room acoustics"
rather than an opposing dry/wet axis. Reverb is a rendering property.

**Prompt-writing rule this measurement produced:** for LM sliders, *divergent*
poles beat tidy minimal pairs. Rewriting the energy poles as one swapped clause
in an otherwise identical caption dropped sep 0.30 → 0.11 and pushed cos to
0.43, because the shared caption dominates the hidden state. Let genre, BPM and
instrumentation all move with the axis. (Gender is the exception — "male"/"female"
is lexically potent enough to carry a minimal pair.)

Also new: multi-row prompt YAMLs with top-level `plus_label`/`minus_label` and
per-row `attributes:` expansion (pin gender on style axes and vice versa),
`--cond_seeds 7,17` for AR-take diversity, sidecar v3 (self-describing: kind,
prefix, labels, `unit_scale`, `recommended_range` — inference scripts no longer
hardcode rank/alpha/targets), `calibrate_scale.py` reports
`unit_scale_projected` (axis-component-only) and the CFG-effective delta, and
`scripts/probe_axis.py` gates each listen folder per axis (F0 / onset /
centroid, ref-relative where full-mix metrics are unreliable).

## LM sliders and song endings (August 2026)

A MiniMax Music 3 cut ends when the AR language model samples the
`<|audio_end|>` token; `audio_duration` is only a hard frame cap. LM-half
LoRAs perturb exactly those logits, so stacked LM sliders make the model
measurably less likely to reach the end token in time — the render blows
through the cap and gets guillotined mid-phrase (tail RMS ≈ overall RMS
instead of a fade). Confirmed in the library: on 2026-08-16, most ≥60 s
slider-stack renders sat at exactly the requested duration with hot tails,
while no-slider renders from 08-13/14 mostly ended naturally.

The fix is at training time: `train_lm_slider_music3.py` now carries an
**audio-end regularizer** (`--endreg_weight`, default 1.0). Each prompt row
is pre-rolled once with the pristine base model — the same CFG compose loop
inference runs, LM + RVQ depth decoder only, cached under `cache/endreg/` —
and every training step teacher-forces the LoRA'd model over that
composition and penalizes drift of the *end margin*,
`logit(<|audio_end|>) − logsumexp(semantic-code band)`, at every decode
position (both ±1 poles). The margin is the exact log-odds that decides
stop-vs-continue, so the slider can still move the musical plan but not the
ending. Causality means the prompt-last hidden state in the teacher-forced
forward equals the prompt-only one, so the slider loss itself is untouched.
Watch `edrift_p`/`edrift_n` in the train jsonl (mean |margin drift| in
nats); the sidecar records the final window under `endreg`. Same-seed 60/90s
A/B (`scripts/render_end_ab.py`, library song `5ec87fed`): at 90s the
un-regularized live-v3 +2 blows through the cap (tail/overall 1.045) while
the regularized retrain ends naturally (0.018), like the base model. Every
shipped LM half has since been retrained with the regularizer — see the
next section.

## Structured-caption + end-regularized LM halves: v4/v5 (August 2026)

Every LM slider half was retrained in one pass with both August fixes at once:

1. **Structured Caption prompts.** The studio rewriter only ever emits the
   three-section Structured Caption (Global Metadata / Vocal Details /
   Arrangement), so the old flat `"Genre: X. BPM: Y."` poles were
   off-distribution for the caption the LM actually sees at runtime. The
   rewrite followed the `prompts-rhyme-v4.yaml` precedent: lean skeletons,
   divergent poles (genre/BPM/mood/mix/instruments all move with the axis),
   gender pinned via `attributes`, delivery pinned "Sung, not rapped" except
   where delivery is the axis (rapslow) — and gender itself stays a minimal
   pair. New files: `prompts-{gender,rapslow,triphop,energy,tempo,distortion,
   breath,live}-v4.yaml` (rhyme reuses `prompts-rhyme-v4.yaml`).
2. **Audio-end regularizer** (`--endreg_weight 1.0`, frames 250, seed 7).

Probed signal improved (or held) on every axis vs the v3 table above —
rapslow 0.343/−0.08, energy 0.331/0.03, tempo 0.314/0.06 (cos was 0.34),
distortion 0.269/−0.03, breath 0.268/0.24, live 0.245/0.32, gender
0.235/−0.08, triphop 0.224/0.10, rhyme 0.270/−0.05 (sep/cos).

Training: rank 8, alpha 8, lr 5e-4, 800 steps, symmetric, common_beta 0,
seed-7 endreg pre-rolls, GPU 0 with the studio stopped (the bf16 LM alone is
16 GB — it does not fit beside the studio's 26 GB). Rhyme ran
`--no-early_stop` per its documented tail instability; its last-200-step
window was checked and is stable (loss still falling at step 800, collapse
steady −0.95). Final-50-step windows:

| checkpoint | cos± | collapse | p/n perc | edrift± |
|---|---|---|---|---|
| gender-lm-v4 | 0.97 | −0.95 | 0.23 | 0.060 |
| rapslow-lm-v4 | 0.96 | −0.96 | 0.29 | 0.080 |
| triphop-lm-v4 | 0.97 | −0.95 | 0.25 | 0.059 |
| energy-lm-v4 | 0.96 | −0.97 | 0.29 | 0.072 |
| tempo-lm-v4 | 0.95 | −0.95 | 0.30 | 0.071 |
| distortion-lm-v4 | 0.94 | −0.96 | 0.33 | 0.076 |
| breath-lm-v4 | 0.93 | −0.95 | 0.36 | 0.069 |
| live-lm-v5 | 0.90 | −0.91 | 0.43 | 0.068 |
| rhyme-lm-v5 | 0.94 | −0.95 | 0.34 | 0.070 |

perc runs higher than the un-regularized v3 numbers by construction — the
end-margin term competes with the slider target — and live matches its
endreg-only v4 profile almost exactly, so the structured prompts cost
nothing there. `app/sliders.json` now points every LM component at these
nine (live-lm-v4 and rhyme-lm-v4, each carrying only one of the two fixes,
are superseded; v3 halves are retained on disk but unshipped). All sidecars
are unit_scale 1.0 — drop-in at strength 1, no recalibration.

Ending A/B for all nine at +2 (90s, song `5ec87fed`, seed 7):
`eval/listen/v4-endreg-ab-90s/`. Eight of nine end naturally short of the
cap (base 76.7s, sliders 62–85s). rhyme-v5 +2 hit the cap on that one seed
(tail/overall 1.64) — endings are sampled and only one seed was rendered, so
sweep more seeds before drawing a conclusion if long rhyme renders truncate
in practice.

## Train

Transformer slider (energy / distortion / tempo / space):

```bash
$PY conceptmod/textsliders/train_lora_music3.py \
  --name energy --rank 8 --alpha 8 --steps 500 --lr 1e-4 \
  --duration 4 --seed 7 --device 0 \
  --prompts_file conceptmod/textsliders/data/prompts-energy.yaml \
  --cache_dir /ml2/music/sliders-conceptmod/cache/energy-v2 \
  --save_dir /ml2/music/sliders-conceptmod/models/energy-slider-v2
```

Language-model slider (gender):

```bash
$PY conceptmod/textsliders/train_lm_slider_music3.py \
  --name gender-lm-v4 \
  --prompts_file conceptmod/textsliders/data/prompts-gender-v4.yaml \
  --save_dir /ml2/music/sliders-conceptmod/models/gender-lm-v4 \
  --rank 8 --alpha 8 --lr 5e-4 --steps 800 --device 0
```

Or run the sequential helper (skips existing last.safetensors and valid wavs):

```bash
./scripts/run_gpu0_sliders.sh all          # train missing + 20s demos + verify
./scripts/run_gpu0_sliders.sh demo-20      # 20s listen sets only
./scripts/run_gpu0_sliders.sh verify
FORCE=1 ./scripts/run_gpu0_sliders.sh demo-20   # regenerate wavs
```

## Demo (labeled wavs)

```bash
$PY conceptmod/textsliders/generate_listen.py \
  --weights models/energy-slider-v2/energy_alpha8.0_rank8_full_last.safetensors \
  --prompts_file conceptmod/textsliders/data/prompts-energy.yaml \
  --name energy --plus_label loud --minus_label quiet \
  --kind transformer --rank 8 --alpha 8 \
  --out_dir eval/listen/energy-20s \
  --scales=-2,0,2 --duration 20 --seed 7 --device 0
```

For gender, pass `--kind lm` and the LM weights.

`generate_listen.py` is resume-safe: existing wavs that pass duration and silence checks are skipped. It rejects short or silent output. Play files in order. Slider clips use the **neutral** caption; `REF` clips change the prompt with the slider off.

## ComfyUI

Do not add a second converter in this repo. Shipped sliders use LoRANetwork
names (`lora_unet-transformer_blocks-N-attn-to_q.lora_down.weight`,
`lora_te-model-layers-N-self_attn-q_proj…`), not PEFT `base_model.model…`
paths, so the Anima/Krea path in
[`scripts/convert_lora_comfyui.py`](https://github.com/mikkel/conceptmod/blob/main/scripts/convert_lora_comfyui.py)
(same file as [ntc-ai/conceptmod#3](https://github.com/ntc-ai/conceptmod/pull/3))
cannot rewrite them as-is — it skips with `no lora_A/lora_B keys`. The Music 3
backends (`music3`, `music3_lm`) live on that same script; detection is from
`lora_unet-` / `lora_te-` keys.

```bash
# clone conceptmod at main (script commit dd0c165 or later)
python /path/to/conceptmod/scripts/convert_lora_comfyui.py \
  path/to/energy_unit_last.safetensors
```

That writes `energy_unit_last_comfyui.safetensors` beside the original. Keys look like:

```
diffusion_model.diffusion_transformer.transformer.layers.N.self_attn.to_qkv.lora_A.weight
diffusion_model.diffusion_transformer.transformer.layers.N.self_attn.to_qkv.lora_B.weight
diffusion_model.diffusion_transformer.transformer.layers.N.self_attn.to_qkv.alpha
diffusion_model.diffusion_transformer.transformer.layers.N.self_attn.to_out.lora_{A,B}.weight
```

`to_q` / `to_k` / `to_v` are fused into ComfyUI's single `to_qkv` Linear
(block-diagonal, scale preserved) from `comfy/ldm/minimax_music/dit.py`.
Put the file in `ComfyUI/models/loras/` and load it with **Load LoRA** on the
MiniMax Music 3 MODEL. LoRA strength is the slider scale: `0` is off, `±1` is
the trained unit (already baked into `*_unit_last` / current sidecars with
`unit_scale: 1.0`), `±2` is a typical listen-folder pole.

### Language-model sliders

LM sliders convert to ComfyUI's generic CLIP key form from `comfy/lora.py`
`model_lora_keys_clip`:

```
text_encoders.model.layers.N.self_attn.q_proj.lora_A.weight
```

That matches an unmerged Qwen3 text encoder — the Hugging Face `language_model`
the trainers wrap. Load the same file through **Load LoRA** with the CLIP /
MiniMax Music 3 text encoder connected (MODEL strength `0` if the file is
LM-only).

ComfyUI can also load a Music 3 TE that was saved with merged `qkv_proj`.
Those checkpoints have no module for the separate q/k/v adapters, so those
keys will log `lora key not loaded`; `o_proj` still applies. The convert
script does not invent a merged-qkv file: GQA makes `q` 4096-wide and `k`/`v`
1024-wide, and the official trainer never wrote `qkv_proj`.

```bash
python /path/to/conceptmod/scripts/convert_lora_comfyui.py \
  path/to/gender-lm-v4_last.safetensors
```

Tests for the mapping live next to the script:

```bash
cd /path/to/conceptmod
pytest tests/test_convert_lora_comfyui.py
```

## Encoder-first note

`train_encoder_music3.py` is the condition-encoder / notrigger analog. Dummy LoRA converges; a single Conv1d cannot fully remap real AR hiddens. Use the LM trainer for identity (gender) and the transformer trainer for production (loud, distortion, tempo, space).
