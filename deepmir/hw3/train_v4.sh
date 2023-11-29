python -m hw3.train \
  --config ./config/v4.json \
  --checkpoint_path ./ckpt/v4 \
  --data_dir ./data/dataset \
  --vocab_file ./data/vocab.json \
  --batch_size 4 \
  $@
