python -m hw3.train \
  --config ./config/v3.json \
  --checkpoint_path ./ckpt/v3 \
  --data_dir ./data/dataset \
  --vocab_file ./data/vocab.json \
  --batch_size 4 \
  $@
