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

Anima has no lyric span. Student +1 fits the + concept prompt; scale 0
fits neu. Unused yaml `attributes` are prefixed and held to
`encode(neu)`. Concept words are **not** held. Minus is a canary only.

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

- prompts: infer/neu only (`a woman sitting on a chair`, `a man reading at a table`) plus the fruit-bowl control
- scales: `0.0`, `0.25`, `0.5`, `1.0` via PEFT `disable_adapter` /
  `set_adapter_scale` / adapter weights — **not** a weight-merge
- seed 42, 40 steps, resolution from args, guider CFG 4
- PNGs + a tiny `mean`/`std` meta json under `save_dir/samples/`

`--sample_every N` also writes the grid during training. `--sample_first_n N`
writes it after each of the first N steps.

The job **fails closed**:

- scale 0 looks like RGB noise (`~122/75`, no spatial correlation) → base pipeline is broken
- scale 0 is an image and scale 0.25 is noise → the adapter is broken

## Live bugs this card patches

1. `ModularPipeline` is **not** an `nn.Module`. Freeze
   `text_conditioner` via `pipe.text_conditioner` / `pipe.transformer`,
   never `pipe.named_parameters()`.
2. A CPU `torch.Generator` cannot drive a CUDA `torch.randn`. `_sample_zt`
   draws noise on CPU and `.to(device)`.

## Yaml

`conceptmod/textsliders/data/prompts-anima.yaml`: smile / smiling
positive vs neu (`a woman sitting on a chair`, `a man reading at a table`),
unused `attributes` pinned (`indoor`, `portrait`). Do not invent a new
concept.

## Related

- [docs/sana-slider.md](sana-slider.md) — cheap 0.6B test backend (not this trainer)
- [docs/lm-plus-neu-exam.md](lm-plus-neu-exam.md) — Music 3 last-token UNI (not this trainer)
- Music 3 live defaults stay `--lm_target v9 --pole_mode hidden`
