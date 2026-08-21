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
  --name gender-lm \
  --prompts_file conceptmod/textsliders/data/prompts-gender.yaml \
  --save_dir /ml2/music/sliders-conceptmod/models/gender-lm-slider \
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
