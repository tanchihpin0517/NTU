python -m hw3.train \
  --config ./config/v1.json \
  --checkpoint_path ./ckpt/v1 \
  --data_dir ./data/dataset \
  --vocab_file ./data/vocab.json \
  --batch_size 2 \
  $@
