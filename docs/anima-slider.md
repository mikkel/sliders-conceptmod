# Anima image slider

Opt-in flow-matching 2B DiT slider on
[`circlestone-labs/Anima-Base-v1.0-Diffusers`](https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers)
(Cosmos-Predict2, Qwen3+T5, Qwen-Image VAE). **Does not change the
default Music 3 trainer** (`train_lora_music3.py` / `--lm_target v9` /
`train_lm_slider_music3.py --pole_mode hidden`). Not Sana, Krea, ZiT,
or MiniMax-H3 — those backends stay out of this card.

CPU tests use `--dummy` (tiny DiT `to_q/to_k/to_v/to_out.0` and
conditioner `q_proj/k_proj/v_proj/o_proj` + whitespace tokenizer).
No GPU train and no Hub weights in CI.

## UNI analog (not Music 3 lyric-hold)

Anima has no lyric span. Student +1 stays on the **neu / infer** caption
(#62 analog). The + caption is the teacher only. Unused yaml
`attributes` (`indoor`, `portrait`) are **pins for unused-token hold
bookkeeping only** — they are **not** prefixed onto captions. Declared
`concept_words` (`smiling, smile, happy, joyful, teeth`) are **not**
held. Minus is a canary only.

Train and in-process sample use the **same bare strings**.
`rows[i].infer_prompt` / `neutral` is exactly the `prompt=` passed to
`pipe(...)`. No attribute-prefix strip.

| scale | student | teacher (`--lm_target trajectory`, `--teacher caption`) | `same_crop` invert teacher | `direct` / `cfg_delta` |
|---|---|---|---|---|
| **+1** | infer / neu caption | short Euler traj of frozen **plus** from `z_T` | invert `x_neu`, Euler plus from mid-σ | 1-step (do not retry for smile) |
| **0** | infer / neu | light identity vs frozen **neu** traj | same | 1-step |
| **−1** | unscored canary only | — | — | — |

`--lm_target trajectory` (default) + `--teacher caption`: sample
`z_T ~ N(0,I)` at infer noise, run **K** FlowMatch Euler steps
(`--traj_steps`, default **4**) with the frozen pipeline on the
**plus** caption → `x_plus` (no grad). Same schedule from the same
`z_T` with the adapter on and the **neu/infer** caption →
`x_student`. Loss is

```
MSE(x_student, x_plus) + λ_id * MSE(x_zero, x_neu)
```

`x_zero` is scale-0 / disabled-adapter on the same short schedule;
`x_neu` is the frozen neu trajectory. `λ_id` is
`--traj_identity_weight` (default **0.25**, light; `0` = off).

Caption-only plus **already shares `z_T`** and still fails smile:
stock Anima jumps full-body → close-up portrait when the caption
goes closed-mouth → teeth. The UNI target then bundles zoom/identity.
Pixel gate at 0.25 only "passes" when the adapter collapses crop.
Wiring is cleared (#70: `peft_pipe≡train_faithful`, no
`SAMPLE_TRAIN_MISMATCH`). This is formulation.

`--teacher same_crop` / `--lm_target same_crop` is the invert /
img2img teacher. Same loss, same `z_T`, but plus is **not**
denoised from infer noise:

```
x_neu  = Euler_K(frozen, neu, z_T)
z_mid  = (1 − σ) · x_neu + σ · z_T     # σ = --teacher_strength (default 0.5)
x_plus = Euler_from_σ(frozen, plus, z_mid)
```

Crop is committed by the neu traj. Plus only runs late σ, so teeth
edit expression instead of rebuilding a portrait. No Comfy — same
thin `predict_v` Euler as trajectory. `--teacher_strength 0.35` is
more crop-locked; `0.75` leaves more plus steps.

The loop is a **thin Euler/flow over `predict_v`**, not
`ModularPipeline` denoise (`pipe(...)` has no grad through the DiT).
It matches Anima's `FlowMatchEulerDiscreteScheduler.step`:

```
σ = linspace(1, 1/K, K)  ∪  {0}
x ← x + (σ_next − σ) * v(x, σ)
```

`--lm_target direct` is 1-step
`MSE(v(neu, adapter), v(pos, frozen)) + MSE(v(neu, scale 0), v(neu, frozen))`.
`--lm_target cfg_delta` is the older UNI CFG-delta recipe. Both stay
as flags. **Do not retry them for smile** — see below.

`--teacher_gap_boost` (default **1**, off) is for `direct` /
`cfg_delta` only: train toward `v_neu + boost * (v_pos − v_neu)` with
`boost > 1`. Not a substitute for trajectory.

Velocity-space CFG is conceptmod's `v(z, t, c) − v(z, t, '')`. Live
sample guidance is **4**. Training cycles **all yaml rows** (woman +
man at least), not only `rows[0]`.

Frozen ref = base modules with PEFT adapters **disabled**.

## Text-path LoRA (smile default)

Stock smile is almost entirely a **text-stack** effect. UNI on DiT
sees a tiny v-space neu/plus gap (`cos ≈ 0.99993`), so a
transformer-only adapter learns crop/identity instead of teeth.

`--lora_targets` chooses which modules get PEFT. This is the explicit
flag / default-on path — text modules are **not** trained unless the
resolved target includes them. `--print_card` prints the active set.

| `--lora_targets` | trained | frozen | attn names |
|---|---|---|---|
| **`conditioner`** (smile default) | `text_conditioner` (AnimaTextConditioner, ~269M, 6 layers) | Qwen3 `text_encoder`, DiT base | `q_proj` / `k_proj` / `v_proj` / `o_proj` |
| `dit` | `transformer` (old v1–v5 recipe) | `text_conditioner`, `text_encoder` | `to_q` / `to_k` / `to_v` / `to_out.0` |
| `dit+conditioner` | both adapters | `text_encoder` | both name sets |

Qwen3 `text_encoder` (28-layer, ~1.2GB) is **not** adapted. The
caption-level smile already lives in the conditioner that maps Qwen
hidden states + T5 token ids to Cosmos embeds. Adapters save as
`{name}_conditioner_lora` and/or `{name}_lora`.

4090 ~24GB: dit-only 768 / `traj_steps=4` was already ~23GB. Smile
retrain should use **conditioner-only at 512** (or `dit+conditioner`
with rank 8). Gradient checkpointing is enabled on the DiT when the
conditioner is trained so embeds can take a trajectory loss without
doubling the 768 OOM.

## Live train card

| field | value |
|---|---|
| hub id | `circlestone-labs/Anima-Base-v1.0-Diffusers` |
| arch | 2B Cosmos-Predict2 DiT, Qwen3+T5, Qwen-Image VAE |
| LoRA | **`--lora_targets conditioner`** (default-on smile path). Rank 16 on `q_proj/k_proj/v_proj/o_proj`. `dit` keeps the old transformer-only recipe. Qwen3 `text_encoder` is not trained. |
| resolution | **768** (4090 smile retrain: **512**) |
| sample steps | **40** |
| CFG (guider) | **4** |
| lr | **`1e-4`** (DiT LoRA; `1e-2` is not sane — prior RunPod run fitted loss ~8e-4 then any nonzero scale collapsed denoise to RGB noise) |
| `--lm_target` | **`trajectory`** (K-step FlowMatch Euler). `same_crop` = that loss + invert teacher. `direct` / `cfg_delta` kept. Music 3 stays `v9`. |
| `--teacher` | **`caption`** (default) or **`same_crop`**. Next smile smoke should use `same_crop`. |
| `--teacher_strength` | **0.5** (invert start σ; `same_crop` only) |
| `--traj_steps` | **4** (live option: 8) |
| `--teacher_gap_boost` | **1** (off; 1-step recipes only) |
| `--sample_every` | **100** (end-of-train gate always runs) |
| control | `a bowl of fruit on a table` |
| sample seed | 42 |
| sample scales | 0.0, 0.25, 0.5, 1.0 |

```bash
HF_HUB_OFFLINE=1 python conceptmod/textsliders/train_lora_anima.py \
  --name smile-anima \
  --prompts_file conceptmod/textsliders/data/prompts-anima.yaml \
  --model_id circlestone-labs/Anima-Base-v1.0-Diffusers \
  --lora_targets conditioner --rank 16 --resolution 768 \
  --sample_steps 40 --cfg 4 \
  --lr 1e-4 --lm_target trajectory --traj_steps 4 \
  --teacher_gap_boost 1 --sample_every 100 \
  --device cuda:0 --save_dir models/smile-anima
```

Recommended **4090 24GB smile retrain** (fits; closed-mouth → smile at
scale ~0.25 without needing DiT identity collapse):

```bash
HF_HUB_OFFLINE=1 python conceptmod/textsliders/train_lora_anima.py \
  --name smile-anima \
  --prompts_file conceptmod/textsliders/data/prompts-anima.yaml \
  --model_id circlestone-labs/Anima-Base-v1.0-Diffusers \
  --lora_targets conditioner --rank 16 --resolution 512 \
  --sample_steps 40 --cfg 4 \
  --lr 1e-4 --lm_target trajectory --traj_steps 4 \
  --teacher_gap_boost 1 --sample_every 100 \
  --device cuda:0 --save_dir models/smile-anima
```

`--lora_targets dit` is the old transformer-only recipe. Joint
`--lora_targets dit+conditioner --rank 8 --resolution 512` if the
conditioner-only smile is still weak.

`--traj_steps 8` is the longer live option. `--teacher_gap_boost 4`
only applies to a `direct` / `cfg_delta` debug run.

## Same-crop smile teacher (next GPU smoke)

Caption-only `--teacher caption` is why v6/v7 still failed the
**pixel** gate after apply/wiring (#70) passed. Stock teacher itself
zooms when plus says teeth. Next smile attempt should invert neu
and denoise plus from mid-σ.

**Do not start a 500-step train in the agent.** Short 4090 / L40S
smoke (8 optimizer steps, 8 sample steps, end-of-train grid only):

```bash
HF_HUB_OFFLINE=1 python conceptmod/textsliders/train_lora_anima.py \
  --name smile-anima-same-crop-smoke \
  --prompts_file conceptmod/textsliders/data/prompts-anima.yaml \
  --model_id circlestone-labs/Anima-Base-v1.0-Diffusers \
  --lora_targets conditioner --rank 16 --resolution 512 \
  --sample_steps 8 --cfg 4 \
  --lr 1e-4 --lm_target same_crop --teacher same_crop \
  --teacher_strength 0.5 --traj_steps 4 \
  --steps 8 --sample_every 0 \
  --device cuda:0 --save_dir models/smile-anima-same-crop-smoke
```

If 0→0.25 now moves teeth **without** a crop jump, continue a real
retrain with `--steps 500 --sample_steps 40 --sample_every 100` and
the same `--teacher same_crop`. If crop still jumps, lower
`--teacher_strength` (0.35) before touching LoRA rank.

CPU dummy (CI / no Hub):

```bash
PYTHONPATH=. python scripts/smoke_anima_same_crop_teacher.py
PYTHONPATH=. python conceptmod/textsliders/train_lora_anima.py \
  --dummy --lm_target same_crop --teacher same_crop --steps 8 --device cpu
```

## Turbo v1.1 is preview only

**Do not train on Turbo.** The live train target stays
[`circlestone-labs/Anima-Base-v1.0-Diffusers`](https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers).
CircleStone: train LoRAs on Base.

`smile-anima-v5` on Base failed because the 1-step v-space teacher gap
is tiny (`cos ≈ 0.99993`). Turbo is **not** the next train unless a
stock/preview smoke shows a *larger* closed-mouth vs teeth gap than
Base. Use it for faster stock/preview smoke only.

There is **no official Turbo Diffusers repo**. Ignore community
`Anima-1.0-Turbo-Diffusers`: it is v1.0 only and uses the wrong VAE
class. Official convert splits `llm_adapter` →
`AnimaTextConditioner` and the rest → `CosmosTransformer3DModel`;
VAE is `AutoencoderKLQwenImage`.

| field | value |
|---|---|
| role | **preview only** (not a train target) |
| Comfy single-file | `circlestone-labs/Anima` → `split_files/diffusion_models/anima-turbo-v1.1.safetensors` (4.18 GB) |
| text encoder | `split_files/text_encoders/qwen_3_06b_base.safetensors` |
| VAE | `split_files/vae/qwen_image_vae.safetensors` |
| convert | official `huggingface/diffusers` `scripts/convert_anima_to_diffusers.py` |
| convert flags | `--save_pipeline --dtype bf16` |
| tokenizers | reuse `tokenizer/` + `t5_tokenizer/` from `Anima-Base-v1.0-Diffusers` |
| output | `./Anima-Turbo-v1.1-Diffusers` |
| sample CFG | **1** |
| sample steps | **8–12** (this repo: `--sample_steps 10`) |
| license | CircleStone Labs **Non-Commercial (NC)** |

```bash
python scripts/convert_anima_turbo_diffusers.py \
  --output ./Anima-Turbo-v1.1-Diffusers
```

The helper downloads the three Hub files, fetches the official convert
(+ sibling `convert_cosmos_to_diffusers.py`), reuses Base tokenizers
from a local checkout or the Hub, and runs:

```bash
python convert_anima_to_diffusers.py \
  --transformer_ckpt_path .../anima-turbo-v1.1.safetensors \
  --text_encoder_ckpt_path .../qwen_3_06b_base.safetensors \
  --vae_ckpt_path .../qwen_image_vae.safetensors \
  --qwen_tokenizer_path .../tokenizer \
  --t5_tokenizer_path .../t5_tokenizer \
  --output_path ./Anima-Turbo-v1.1-Diffusers \
  --save_pipeline --dtype bf16
```

Turbo preview sample (sample path only; train recipe stays Base):

```bash
HF_HUB_OFFLINE=1 python conceptmod/textsliders/train_lora_anima.py \
  --name smile-anima-turbo-preview \
  --prompts_file conceptmod/textsliders/data/prompts-anima.yaml \
  --model_id ./Anima-Turbo-v1.1-Diffusers \
  --rank 16 --resolution 768 --sample_steps 10 --cfg 1 \
  --lr 1e-4 --lm_target trajectory --traj_steps 4 \
  --teacher_gap_boost 1 --sample_every 100 \
  --device cuda:0 --save_dir models/smile-anima-turbo-preview
```

`--cfg 1` is an explicit sample-path request. The guider fail-close
still refuses a *silent* fallback to CFG 1 when CFG 4 could not be
applied. `--cfg` does not change the trajectory train loss.

Print the preview card without training:

```bash
PYTHONPATH=. python conceptmod/textsliders/train_lora_anima.py --print_turbo_preview
python scripts/convert_anima_turbo_diffusers.py --print-recipe
```

`--lr` stays overridable. CircleStone's own finetune note is even
lower (`~2e-5` for rank 32); `1e-4` is the trainer default.

Dummy (CI / no GPU / no Hub):

```bash
PYTHONPATH=. python conceptmod/textsliders/train_lora_anima.py \
  --dummy --steps 8 --device cpu --save_dir /tmp/anima-dummy
PYTHONPATH=. pytest tests/test_anima_slider.py -q
python scripts/smoke_anima_slider.py
python scripts/smoke_anima_same_crop_teacher.py
```

## PEFT sync: why v6/v7 may have invalidated the 0→0.25 gate

Smile v6/v7 trained conditioner LoRA (`--lora_targets conditioner`) and
still failed the in-process 0→0.25 sample gate. The next place to look
is **apply / object identity**, not another UNI recipe.

Train encode is `LiveAnimaBackend._encode_raw` →
`self.pipe.text_conditioner` (the PEFT wrapper after
`get_peft_model`). Sample is `emit_inprocess_samples` →
`_call_modular_pipe(pipe, …)` → `pipe(prompt=…)`.

On current ModularPipeline, `pipe.components` is a **live property**
(`getattr` per name) and `pipe(prompt=…)` passes `self` into
`AnimaTextConditioningStep`, so a successful
`pipe.text_conditioner = get_peft_model(...)` *should* share the
object. That was never asserted. Holes that still make the gate lie:

1. Attach never called `update_components`. Specs / ComponentsManager
   can keep the **pre-PEFT** `AnimaTextConditioner`. A later
   `load_components` / manager lookup resurrects the base module.
2. Dummy / captured `components` dicts and `_blocks.*` attributes can
   hold the unwrapped module while `pipe.text_conditioner` is PEFT.
3. `set_adapter_scale` must succeed on the **AnimaTextConditioner PEFT
   wrapper**. When `--lora_targets conditioner` (`train_dit=false`),
   sample must **not** depend on transformer PEFT APIs.

`sync_peft_into_modular_pipeline` writes the PEFT modules into the
attribute, `update_components`, `components` map, and `_blocks`
holders, then asserts `id()` match. `encode_text` and the sample
helper must share the same conditioner object.

### Sample modes

| `--sample_mode` | path | default |
|---|---|---|
| **`peft_pipe`** | `pipe(prompt=…)` after sync (same PEFT objects as `encode_text`) | **yes** |
| `train_faithful` | `backend.encode_text` + transformer denoise (same path as the loss). Dummy still writes structured images | no |

`--lora_targets conditioner|dit|dit+conditioner` is unchanged.

### Embed diagnostic (CPU dummy / live GPU)

Do **not** start another 500-step smile train. After merge, re-validate
a v6 adapter with one short GPU smoke:

```bash
# CPU dummy (CI / no Hub): must PASS after a plus-aligned dummy LoRA
PYTHONPATH=. python scripts/diag_anima_conditioner_embed.py --dummy

# Live: Base + saved v6 conditioner adapter (not a retrain)
HF_HUB_OFFLINE=1 PYTHONPATH=. python scripts/diag_anima_conditioner_embed.py \
  --live --device cuda:0 \
  --model_id circlestone-labs/Anima-Base-v1.0-Diffusers \
  --conditioner_adapter models/smile-anima-v6/smile-anima_conditioner_lora
```

The script prints, for neu/plus:

- `||E(neu, s) − E(neu, 0)||` vs `||E(plus, 0) − E(neu, 0)||`
- `cosine(E(neu, s)−E(neu, 0), E(plus, 0)−E(neu, 0))`

**PASS** at scale 1: student Δ is not near-zero and is aligned with
the frozen plus−neu teacher Δ. **FAIL** + near-zero student Δ means
the adapter is not in the encode graph (sync/apply). If Modular/sample
encode ≠ `backend.encode_text`, the script flags
`SAMPLE_TRAIN_MISMATCH`.

If that smoke **PASS**es and the mismatch flag is absent, re-run only
the in-process 0 / 0.25 / 0.5 / 1.0 sample grid (no full retrain). If
it still fails 0→0.25, the gate is about **weights / formulation**,
not a stale conditioner object.

## In-process PEFT sample gate

The prior live run never sampled **in-process** with the PEFT-wrapped
training transformer. Post-hoc `W += scale*(α/r)*(B@A)` matched 224/224
modules and still printed RGB noise at scale 0.01. The trainer now
emits a scale grid through the same `ModularPipeline` path used for
real infer (`pipe(prompt=...)`) **with PEFT still attached** to the
adapted modules (`pipe.text_conditioner` and/or `pipe.transformer`)
**after** `sync_peft_into_modular_pipeline` (same `nn.Module` instances
as `encode_text`):

- prompts: **the same** infer/neu strings used at train time
  (`a woman sitting on a chair, neutral expression, closed mouth`,
  `a man reading at a table, neutral expression, closed mouth`) plus
  the fruit-bowl control. `rows[0].infer_prompt` == `pipe(prompt=...)`.
- scales: `0.0`, `0.25`, `0.5`, `1.0` via PEFT `disable_adapter` /
  `set_adapter_scale` / adapter weights — **not** a weight-merge
- seed 42, 40 steps, resolution from args, guider CFG 4
- PNGs + a tiny `mean`/`std` meta json under `save_dir/samples/`

`--sample_every` defaults to **100** and also writes the grid during
training. End-of-train gate always runs. `--sample_first_n N` writes
it after each of the first N steps. `--sample_every 0` is end-of-train
only.

CFG **4** is set on `guider.config.guidance_scale` (and any real field
the guider exposes). Do **not** pass `guidance_scale=` into
`pipe(...)` — ModularPipeline logs `Unexpected input 'guidance_scale'
… ignored` and would silently sample at CFG 1. The trainer fails
closed if the guider CFG cannot be applied.

The job **fails closed**:

- scale 0 looks like RGB noise (`~122/75`, no spatial correlation) → base pipeline is broken
- scale 0 is an image and scale 0.25 is noise → the adapter is broken

## Live bugs this card patches

1. `ModularPipeline` is **not** an `nn.Module`. Freeze the text stack
   via `pipe.text_conditioner` / `pipe.transformer`, never
   `pipe.named_parameters()`. When `--lora_targets` includes
   `conditioner`, base conditioner weights stay frozen and PEFT
   `lora_*` params stay trainable.
2. A CPU `torch.Generator` cannot drive a CUDA `torch.randn`. `_sample_zt`
   draws noise on CPU and `.to(device)`.
3. After `get_peft_model` / `load_adapter`, PEFT modules must be
   synced into ModularPipeline (`update_components` + holder walk +
   `id()` assert). v6/v7 may have sampled through a pre-PEFT
   conditioner and invalidated the 0→0.25 gate. See
   [PEFT sync](#peft-sync-why-v6v7-may-have-invalidated-the-0025-gate).

## Why 1-step fails (v-space gap is tiny)

v3 stock images clearly differ (closed-mouth neu vs toothy plus). v4
matched train captions to those same infer/neu strings and fitted
`--lm_target direct`. A velocity diagnostic on the trained adapter
(seed 42, `LiveAnimaBackend.predict_v`, `t ∈ {100, 300, 500, 700}`)
found:

- `cos(v(pos, frozen), v(neu, frozen)) ≈ 0.99993` (MSE ≈ 0.00037) —
  the one-step teacher gap is microscopic even though the **images**
  differ.
- Adapter Δ is ~3–7× larger than that teacher Δ, with only weak
  alignment (mean δ-cos adapter-vs-plus ≈ 0.28).
- `v(neu, scale=1)` is a **tie** vs frozen plus/neu; scale 0 matches
  neu perfectly.

So 1-step `direct` / `cfg_delta` UNI cannot carry expression on Anima.
Need multi-step/trajectory (or a CFG-amplified 1-step teacher via
`--teacher_gap_boost > 1`), **not** another `to_v` 200-step 1-step
retry. `--lm_target trajectory` is the live recipe.

## Stock teacher smoke (before trusting a slider)

v2 failed here: stock Anima already soft-smiles on the bare neu
`a woman sitting on a chair`, so UNI plus=`a smiling woman…` vs that
neu had almost no teacher gap. The slider barely moved.

Before a live train, render **stock** Anima (no LoRA) on neu and plus
with the **same seed**, same steps, CFG 4 via
`guider.config.guidance_scale`. Stock neu must look clearly less
smiley than stock plus. If both already grin, do not trust the slider.

For a faster first look, convert Turbo v1.1 (preview only) and sample
at CFG 1, 8–12 steps. That smoke does **not** change the train target
off Base. See [Turbo v1.1 is preview only](#turbo-v11-is-preview-only).

v3 captions:

| pair | neu / infer (student +1) | plus (CFG teacher only) |
|---|---|---|
| woman | `a woman sitting on a chair, neutral expression, closed mouth` | `a woman sitting on a chair, big smile showing teeth, happy joyful expression` |
| man | `a man reading at a table, neutral expression, closed mouth` | `a man reading at a table, big smile showing teeth, happy joyful expression` |

Sana lesson: weak "happy" failed; a harder smile (teeth / joyful) gave
the teacher a real gap. This check needs local Anima weights; CI /
`--dummy` does not run Hub.

```python
from conceptmod.textsliders.anima_slider import stock_teacher_smoke_captions
print(stock_teacher_smoke_captions())
```

## Yaml

`conceptmod/textsliders/data/prompts-anima.yaml`: closed-mouth neu vs
hard-smile plus (v3). Captions stay **bare** — unused `attributes`
(`indoor`, `portrait`) are pins for hold bookkeeping, not caption
prefixes. `concept_words: smiling, smile, happy, joyful, teeth`. + is
teacher only; student +1 stays on neu/infer (#62 analog). Do not invent
a new concept (keep smile/happy; not age).

v3 miss: `load_anima_prompts` prefixed `indoor` / `portrait` onto train
captions while `infer_sample_prompts` stripped them, so the LoRA never
trained on the infer prompt we sampled. Scale 0 matched stock neu;
scale 0.25 barely moved. Train now cycles woman + man on those same
bare strings.

## Related

- [docs/sana-slider.md](sana-slider.md) — cheap 0.6B test backend (not this trainer)
- [docs/lm-plus-neu-exam.md](lm-plus-neu-exam.md) — Music 3 last-token UNI (not this trainer)
- Music 3 live defaults stay `--lm_target v9 --pole_mode hidden`
- Turbo v1.1 convert helper: `scripts/convert_anima_turbo_diffusers.py` (preview only; train stays Base)
