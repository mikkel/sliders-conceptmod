# Krea image sliders (opt-in UNI)

Opt-in image trainer. **Does not change the Music 3 default**
(`train_lora_music3.py` / `train_lm_slider_music3.py --lm_target v9`).
Anima / ZiT / H3 are a separate PR and are not in this trainer.

CPU tests use `--dummy` mocks. No Hub weights, no GPU train in CI.

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

## Live train card

Official advice: **train LoRAs on Raw, run on Turbo.**

| | Raw (train here) | Turbo (run) |
|---|---|---|
| weights | `krea/Krea-2-Raw` | local ComfyUI `.safetensors` (name contains `turbo`) |
| LoRA rank | **16** | same file |
| resolution | **512** | 512 |
| sample steps | **28** | **8** |
| CFG | **4.5** | **0** |

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_krea.py \
  --name age-krea \
  --prompts_file conceptmod/textsliders/data/prompts-krea.yaml \
  --model_id krea/Krea-2-Raw \
  --rank 16 --resolution 512 \
  --sample_steps 28 --sample_guidance 4.5 \
  --seed 7 --device 0
```

Local Turbo file (inference / sample card only; do not prefer this for
train):

```bash
python conceptmod/textsliders/train_lora_krea.py \
  --name age-krea-turbo-card \
  --model_id /path/to/Krea-2-Turbo.safetensors \
  --rank 16 --resolution 512
# sample_steps=8 sample_guidance=0  (from the filename)
```

`--dummy` is the CI path: two CPU steps, never loads the 12B
transformer or Hub weights.

```bash
python conceptmod/textsliders/train_lora_krea.py \
  --dummy --name krea-age-dummy \
  --prompts_file conceptmod/textsliders/data/prompts-krea.yaml \
  --save_dir /tmp/krea-dummy
```

## Yaml

`conceptmod/textsliders/data/prompts-krea.yaml`: positive / neutral,
unused `attributes` pinned (male / female on an age slider). `negative`
is stored for the canary and is not a teacher.
