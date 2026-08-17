#!/usr/bin/env bash
# Sequential GPU-0 slider train + long listen demos. Do not use GPU 1.
# Existing last.safetensors and valid wavs are reused unless FORCE=1.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/ml2/music/sliders-conceptmod
export HF_HUB_OFFLINE=1
export HF_HOME=/ml2/music/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/mikkel/anaconda3/envs/minimax-music3/bin/python
ROOT=/ml2/music/sliders-conceptmod
cd "$ROOT"
FORCE="${FORCE:-0}"

weights_tf() {
  local name="$1"
  if [[ "$name" == "energy" && -f "$ROOT/models/energy-slider-v2/energy_alpha8.0_rank8_full_last.safetensors" ]]; then
    echo "$ROOT/models/energy-slider-v2/energy_alpha8.0_rank8_full_last.safetensors"
    return
  fi
  echo "$ROOT/models/${name}-slider/${name}_alpha8.0_rank8_full_last.safetensors"
}

train_tf() {
  local name="$1" prompts="$2"
  local weights
  weights="$(weights_tf "$name")"
  if [[ -f "$weights" && "$FORCE" != "1" ]]; then
    echo "skip train $name (exists $weights)"
    return
  fi
  $PY -u conceptmod/textsliders/train_lora_music3.py \
    --name "$name" --rank 8 --alpha 8 --steps 500 --lr 1e-4 \
    --duration 4 --seed 7 --device 0 \
    --prompts_file "conceptmod/textsliders/data/${prompts}" \
    --cache_dir "$ROOT/cache/${name}" \
    --save_dir "$ROOT/models/${name}-slider"
}

demo_tf() {
  local name="$1" prompts="$2" plus="$3" minus="$4" duration="${5:-20}"
  local extra=()
  if [[ "$FORCE" == "1" ]]; then
    extra+=(--force)
  fi
  $PY -u conceptmod/textsliders/generate_listen.py \
    --weights "$(weights_tf "$name")" \
    --prompts_file "conceptmod/textsliders/data/${prompts}" \
    --name "$name" --plus_label "$plus" --minus_label "$minus" \
    --kind transformer --rank 8 --alpha 8 \
    --out_dir "$ROOT/eval/listen/${name}-${duration}s" \
    --scales=-2,0,2 --duration "$duration" --seed 7 --device 0 \
    "${extra[@]}"
}

demo_lm() {
  local duration="${1:-20}"
  local extra=()
  if [[ "$FORCE" == "1" ]]; then
    extra+=(--force)
  fi
  $PY -u conceptmod/textsliders/generate_listen.py \
    --weights "$ROOT/models/gender-lm-slider/gender-lm_last.safetensors" \
    --prompts_file conceptmod/textsliders/data/prompts-gender.yaml \
    --name gender-lm --plus_label female --minus_label male \
    --kind lm --rank 8 --alpha 8 \
    --out_dir "$ROOT/eval/listen/gender-lm-${duration}s" \
    --scales=-2,0,2 --duration "$duration" --seed 7 --device 0 \
    "${extra[@]}"
}

verify_long() {
  local duration="${1:-20}"
  $PY -u "$ROOT/scripts/verify_listen.py" --min-duration "$(awk -v d="$duration" 'BEGIN{print d*0.9}')" --expect 5 \
    "$ROOT/eval/listen/tempo-${duration}s" \
    "$ROOT/eval/listen/space-${duration}s" \
    "$ROOT/eval/listen/energy-${duration}s" \
    "$ROOT/eval/listen/distortion-${duration}s" \
    "$ROOT/eval/listen/gender-lm-${duration}s"
}

case "${1:-all}" in
  tempo) train_tf tempo prompts-tempo.yaml; demo_tf tempo prompts-tempo.yaml fast slow 20 ;;
  space) train_tf space prompts-space.yaml; demo_tf space prompts-space.yaml wet dry 20 ;;
  long-existing)
    demo_tf energy prompts-energy.yaml loud quiet 20
    demo_tf distortion prompts-distortion.yaml distorted acoustic 20
    demo_lm 20
    ;;
  demo-20)
    demo_tf tempo prompts-tempo.yaml fast slow 20
    demo_tf space prompts-space.yaml wet dry 20
    demo_tf energy prompts-energy.yaml loud quiet 20
    demo_tf distortion prompts-distortion.yaml distorted acoustic 20
    demo_lm 20
    verify_long 20
    ;;
  verify)
    $PY -u "$ROOT/scripts/verify_listen.py" --min-duration 7.5 --expect 5 \
      "$ROOT/eval/listen/energy" \
      "$ROOT/eval/listen/distortion" \
      "$ROOT/eval/listen/gender-lm"
    verify_long 20
    ;;
  all)
    train_tf tempo prompts-tempo.yaml
    train_tf space prompts-space.yaml
    demo_tf tempo prompts-tempo.yaml fast slow 20
    demo_tf space prompts-space.yaml wet dry 20
    demo_tf energy prompts-energy.yaml loud quiet 20
    demo_tf distortion prompts-distortion.yaml distorted acoustic 20
    demo_lm 20
    verify_long 20
    ;;
  *) echo "usage: $0 [tempo|space|long-existing|demo-20|verify|all]" >&2; exit 2 ;;
esac
