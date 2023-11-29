#!/usr/bin/env zsh

# ver="v1"
# step="00200000"
ver="$1"
step="$2"
strategy="$3"
T=("1.1" "1.05" "1.0" "0.95" "0.9")
P=("0.95" "0.9" "0.85")

for t in ${T[@]}; do
  for p in ${P[@]}; do
    python -m hw3.inference \
      --config_file ./ckpt/$ver/config.json \
      --checkpoint_file ./ckpt/$ver/model_$step \
      --output_dir ./output/$ver \
      --vocab_file ./data/vocab.json \
      --sampling_method top-p \
      --temperature $t \
      --threshold $p \
      --strategy $strategy
  done
done
