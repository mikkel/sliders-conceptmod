# Anima image slider

Opt-in flow-matching 2B DiT slider on
[`circlestone-labs/Anima-Base-v1.0-Diffusers`](https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers)
(Cosmos-Predict2, Qwen3+T5, Qwen-Image VAE). **Does not change the
default Music 3 trainer** (`train_lora_music3.py` / `--lm_target v9` /
`train_lm_slider_music3.py --pole_mode hidden`). Not Sana, Krea, ZiT,
or MiniMax-H3 — those backends stay out of this card.

CPU tests use `--dummy` (tiny `to_q/to_k/to_v/to_out.0` DiT +
whitespace tokenizer). No GPU train and no Hub weights in CI.

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

| scale | student | teacher (`--lm_target trajectory`, default) | `direct` (1-step) | `cfg_delta` (1-step) |
|---|---|---|---|---|
| **+1** | infer / neu caption | short Euler traj of frozen **plus** | `v(z, t, + concept)` frozen | `v(+) − v('')` |
| **0** | infer / neu | light identity vs frozen **neu** traj | `v(z, t, neu)` frozen | `v(neu) − v('')` |
| **−1** | unscored canary only | — | — | — |

`--lm_target trajectory` (default, live smile): sample `z_T ~ N(0,I)`
at infer noise, run **K** FlowMatch Euler steps (`--traj_steps`,
default **4**) with the frozen pipeline on the **plus** caption →
`x_plus` (no grad). Same schedule from the same `z_T` with the adapter
on and the **neu/infer** caption → `x_student`. Loss is

```
MSE(x_student, x_plus) + λ_id * MSE(x_zero, x_neu)
```

`x_zero` is scale-0 / disabled-adapter on the same short schedule;
`x_neu` is the frozen neu trajectory. `λ_id` is
`--traj_identity_weight` (default **0.25**, light; `0` = off).

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

Frozen ref = base transformer with the PEFT adapter **disabled**. LoRA
rank 16 on attn `to_q` / `to_k` / `to_v` / `to_out.0`. Do **not** train
`text_conditioner` (CircleStone: the LLM adapter).

## Live train card

| field | value |
|---|---|
| hub id | `circlestone-labs/Anima-Base-v1.0-Diffusers` |
| arch | 2B Cosmos-Predict2 DiT, Qwen3+T5, Qwen-Image VAE |
| LoRA | rank 16 on `to_q/to_k/to_v/to_out.0` |
| resolution | **768** |
| sample steps | **40** |
| CFG (guider) | **4** |
| lr | **`1e-4`** (DiT LoRA; `1e-2` is not sane — prior RunPod run fitted loss ~8e-4 then any nonzero scale collapsed denoise to RGB noise) |
| `--lm_target` | **`trajectory`** (K-step FlowMatch Euler). `direct` / `cfg_delta` kept. Music 3 stays `v9`. |
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
  --rank 16 --resolution 768 --sample_steps 40 --cfg 4 \
  --lr 1e-4 --lm_target trajectory --traj_steps 4 \
  --teacher_gap_boost 1 --sample_every 100 \
  --device cuda:0 --save_dir models/smile-anima
```

`--traj_steps 8` is the longer live option. `--teacher_gap_boost 4`
only applies to a `direct` / `cfg_delta` debug run.

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
```

## In-process PEFT sample gate

The prior live run never sampled **in-process** with the PEFT-wrapped
training transformer. Post-hoc `W += scale*(α/r)*(B@A)` matched 224/224
modules and still printed RGB noise at scale 0.01. The trainer now
emits a scale grid through the same `ModularPipeline` path used for
real infer (`pipe(prompt=...)`) **with PEFT still attached** to
`pipe.transformer`:

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

1. `ModularPipeline` is **not** an `nn.Module`. Freeze
   `text_conditioner` via `pipe.text_conditioner` / `pipe.transformer`,
   never `pipe.named_parameters()`.
2. A CPU `torch.Generator` cannot drive a CUDA `torch.randn`. `_sample_zt`
   draws noise on CPU and `.to(device)`.

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
