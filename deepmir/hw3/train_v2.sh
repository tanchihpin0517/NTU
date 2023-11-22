python -m hw3.train \
  --config ./config/v2.json \
  --checkpoint_path ./ckpt/v2 \
  --data_dir ./data/dataset \
  --vocab_file ./data/vocab.json \
  --batch_size 2 \
  $@
