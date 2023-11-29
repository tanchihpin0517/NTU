#!/usr/bin/env zsh

ver="v3"
step="00200000"
t="0.95"
p="0.95"
python -m hw3.inference \
  --config_file ./ckpt/$ver/config.json \
  --checkpoint_file ./ckpt/$ver/model_$step \
  --output_dir ./output/$ver \
  --vocab_file ./data/vocab.json \
  --sampling_method top-p \
  --temperature $t \
  --threshold $p \
  --strategy nocache
