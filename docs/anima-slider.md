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
(#62 analog). The + caption is the CFG teacher only. Unused yaml
`attributes` are prefixed and held to `encode(neu)`. Declared
`concept_words` (`smiling, smile, happy, joyful, teeth`) are **not**
held. Minus is a canary only.

| scale | student | teacher |
|---|---|---|
| **+1** | infer / neu caption | `v(z, t, + concept) − v('')` |
| **0** | infer / neu | `v(z, t, neu) − v('')` |
| **−1** | unscored canary only | — |

Velocity-space CFG is conceptmod's `v(z, t, c) − v(z, t, '')`. Live
sample guidance is **4**.

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
| control | `a bowl of fruit on a table` |
| sample seed | 42 |
| sample scales | 0.0, 0.25, 0.5, 1.0 |

```bash
HF_HUB_OFFLINE=1 python conceptmod/textsliders/train_lora_anima.py \
  --name smile-anima \
  --prompts_file conceptmod/textsliders/data/prompts-anima.yaml \
  --model_id circlestone-labs/Anima-Base-v1.0-Diffusers \
  --rank 16 --resolution 768 --sample_steps 40 --cfg 4 \
  --lr 1e-4 --device cuda:0 --save_dir models/smile-anima
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

- prompts: infer/neu only (`a woman sitting on a chair, neutral expression, closed mouth`, `a man reading at a table, neutral expression, closed mouth`) plus the fruit-bowl control
- scales: `0.0`, `0.25`, `0.5`, `1.0` via PEFT `disable_adapter` /
  `set_adapter_scale` / adapter weights — **not** a weight-merge
- seed 42, 40 steps, resolution from args, guider CFG 4
- PNGs + a tiny `mean`/`std` meta json under `save_dir/samples/`

`--sample_every N` also writes the grid during training. `--sample_first_n N`
writes it after each of the first N steps.

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

## Stock teacher smoke (before trusting a slider)

v2 failed here: stock Anima already soft-smiles on the bare neu
`a woman sitting on a chair`, so UNI plus=`a smiling woman…` vs that
neu had almost no teacher gap. The slider barely moved.

Before a live train, render **stock** Anima (no LoRA) on neu and plus
with the **same seed**, same steps, CFG 4 via
`guider.config.guidance_scale`. Stock neu must look clearly less
smiley than stock plus. If both already grin, do not trust the slider.

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
hard-smile plus (v3), unused `attributes` pinned (`indoor`, `portrait`),
`concept_words: smiling, smile, happy, joyful, teeth`. + is CFG teacher
only; student +1 stays on neu/infer (#62 analog). Do not invent a new
concept (keep smile/happy; not age).

## Related

- [docs/sana-slider.md](sana-slider.md) — cheap 0.6B test backend (not this trainer)
- [docs/lm-plus-neu-exam.md](lm-plus-neu-exam.md) — Music 3 last-token UNI (not this trainer)
- Music 3 live defaults stay `--lm_target v9 --pole_mode hidden`
