# Krea image sliders (opt-in UNI)

Opt-in image trainer. **Does not change the Music 3 default**
(`train_lora_music3.py` / `train_lm_slider_music3.py --lm_target v9` /
`--pole_mode hidden`). Anima / ZiT / H3 are a separate PR and are not
in this trainer. Do **not** port Anima `embed_struct` / `same_crop`
here — smile-first on Raw accepts entanglement.

CPU tests use `--dummy` mocks. No Hub weights, no GPU train in CI.
Live load is offline-safe (`local_files_only` / `HF_HUB_OFFLINE=1`)
unless `--allow_hub` is set (same gate as Anima).

## UNI analog (not Music 3 lyric-hold)

Music 3 lyric-hold pins yaml `lyrics` tokens. Krea does not.

| scale | teacher | not a teacher |
|---|---|---|
| **+1** | + concept prompt velocity `v(z,t,pos)` | minus caption |
| **0** | neutral prompt velocity `v(z,t,neu)` | leftover-gate / pair-odd |

Minus is a **canary** only: `v(neg) − v('')` is logged and never enters
the loss.

Unused prompt tokens (pinned yaml `attributes`, plus the shared
skeleton that also appears in neu) hold to encode(neu). Concept words
— tokens in pos that are not in neu — are **not** held.

Happy / smile yaml uses **bare captions**: attributes pin unused
gender for bookkeeping only and are **not** prefixed onto
target / positive / neutral (Anima smile lesson).

## Velocity-space CFG

conceptmod Krea geometry, when it maps:

```
direction(c) = v(z, t, c) − v(z, t, '')
v_cfg(c)     = v(c) + g · (v(c) − v(''))    # g > 0
             = v(c)                         # g = 0 (Turbo)
```

Raw samples at CFG 4.5, so the +1 teacher is CFG-composed
`v(pos) + 4.5 · (v(pos) − v(''))`. Scale 0 stays raw `v(neu)`.
Turbo trains and samples at CFG 0.

## Live train card (smile-krea / happy-krea)

Official advice: **train LoRAs on Raw, run on Turbo.**

Live backend (`conceptmod/textsliders/krea_live.py`, lazy-imported
from `_load_live_backend`):

- `Krea2Pipeline.from_pretrained("krea/Krea-2-Raw", dtype=bfloat16)`
  (falls back to `torch_dtype=` if the installed Diffusers wants that)
- `--lora_targets dit` (default): LoRA on DiT attn
  `to_q/to_k/to_v/to_out.0` via peft (**rank 16**)
- `--lora_targets te` / `text_encoder`: Qwen3-VL attention
  `q_proj/k_proj/v_proj/o_proj`. `--lora_targets dit+te` is joint
  (Anima lesson: expression can live in the text path)
- frozen ref = adapter disabled (a second 12B copy will not fit)
- Continuous sample scales write LoRA `scaling` (`alpha/r * scale`).
  PEFT `set_adapter_scale` often no-ops, which made live 0.25 / 0.5 /
  1.0 grids byte-identical (only scale 0 / `disable_adapter` differed)
- Park VAE on CPU. Frozen TE: encode on GPU, then park before DiT
  backward (**~48GB**, A6000 / A100). Trained TE stays resident so
  encode+backward stay coherent (prefer 80GB A100 for `dit+te`)
- End-of-train smile-first grid: scales `0 / 0.25 / 0.5 / 1.0` on
  neu / infer captions plus the fruit-bowl control, PNGs under
  `save_dir/samples`. Not a crop-purity gate. Do **not** port Anima
  `same_crop` / `embed_struct`.

`--hold_weight` defaults to **1.0** (age yaml unused-gender hold).
Live smile-krea logs were hold-dominated (`hold≈7.31` of `loss≈7.35`)
because frozen-TE hold is a near-constant. **Smile card uses 0.1.**
Do not silently lower the age-yaml default.

| | Raw (train here) | Turbo (run) |
|---|---|---|
| weights | `krea/Krea-2-Raw` | local ComfyUI `.safetensors` (name contains `turbo`) or `krea/Krea-2-Turbo` |
| LoRA rank | **16** | same file |
| resolution | **512** | 512 |
| sample steps | **28** | **8** |
| CFG | **4.5** | **0** |

`krea/Krea-2-Raw` is **gated**. Accept the card and pass a Hub token.
`--allow_hub` is required for the first download; afterwards a warm
cache can run with `HF_HUB_OFFLINE=1` and no flag.

Retrain card (**smile-krea-v2**, A100 80GB preferred; A6000 48GB is
tight once TE stays resident):

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_krea.py \
  --name smile-krea-v2 \
  --prompts_file conceptmod/textsliders/data/prompts-krea-happy.yaml \
  --model_id krea/Krea-2-Raw \
  --allow_hub \
  --lora_targets dit+te --rank 16 --hold_weight 0.1 \
  --resolution 512 \
  --sample_steps 28 --sample_guidance 4.5 \
  --steps 500 --lr 1e-4 --seed 7 --device 0 \
  --save_dir models/smile-krea-v2
```

Adapters: `{save_dir}/{name}_lora/dit_lora` and
`{save_dir}/{name}_lora/te_lora` (sidecar keys `dit_lora` /
`te_lora`). Sample PNGs: `{save_dir}/samples/`. Mid-scales must
differ — if 0.25 / 0.5 / 1.0 PNGs are byte-identical, scale is
broken again.

DiT-only (v1 card, frozen TE parked, 48GB):

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_krea.py \
  --name smile-krea \
  --prompts_file conceptmod/textsliders/data/prompts-krea-happy.yaml \
  --model_id krea/Krea-2-Raw \
  --allow_hub \
  --lora_targets dit --rank 16 --hold_weight 0.1 \
  --resolution 512 \
  --sample_steps 28 --sample_guidance 4.5 \
  --steps 500 --lr 1e-4 --seed 7 --device 0 \
  --save_dir models/smile-krea
```

Age yaml (prefixed unused gender) is still the stock file:

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_krea.py \
  --name age-krea \
  --prompts_file conceptmod/textsliders/data/prompts-krea.yaml \
  --model_id krea/Krea-2-Raw \
  --allow_hub \
  --rank 16 --resolution 512 \
  --sample_steps 28 --sample_guidance 4.5 \
  --seed 7 --device 0
```

Local Turbo file (inference / sample card only; do not prefer this for
train). VAE + Qwen3-VL still come from a cached Raw skeleton:

```bash
python conceptmod/textsliders/train_lora_krea.py \
  --name age-krea-turbo-card \
  --model_id /path/to/Krea-2-Turbo.safetensors \
  --rank 16 --resolution 512
# sample_steps=8 sample_guidance=0  (from the filename)
```

`--dummy` is the CI path: two CPU steps, never loads the 12B
transformer or Hub weights, never imports `krea_live.py`.

```bash
python conceptmod/textsliders/train_lora_krea.py \
  --dummy --name krea-age-dummy \
  --prompts_file conceptmod/textsliders/data/prompts-krea.yaml \
  --save_dir /tmp/krea-dummy
PYTHONPATH=. pytest tests/test_krea_slider.py -q
```

## Yaml

`conceptmod/textsliders/data/prompts-krea-happy.yaml`: UNI happy/smile
(not age). Stronger Anima teacher (`big smile showing teeth, happy
joyful expression`) on a closed-mouth neu. `attributes` pin unused
gender (`male` / `female`) without prefixing. `negative` is the
canary. `control_prompt` is `a bowl of fruit on a table` — verify
only, never a teacher.

`conceptmod/textsliders/data/prompts-krea.yaml`: stock age slider
(old / young) with prefixed unused gender. Still valid.

## Related

- [conceptmod backends/krea.py](https://github.com/mikkel/conceptmod/blob/main/conceptmod/backends/krea.py)
- [docs/sana-slider.md](sana-slider.md) — cheap happy UNI analog
- [docs/anima-slider.md](anima-slider.md) — smile-first (do not copy `same_crop` / `embed_struct` here)
- Music 3 live defaults stay `--lm_target v9 --pole_mode hidden`.
