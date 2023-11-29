#!/usr/bin/env bash

function _analyze() {
  python -m hw3.analyze analyze \
    --output_dir $1 \
    --result_dir $2 \
    --vocab_file ./data/vocab.json
}

# output_dir=$1
# result_dir=$2
# python -m hw3.analyze analyze \
#   --output_dir ./data/dataset_cache_bd_4 \
#   --result_dir ./result \
#   --vocab_file ./data/vocab.json

_analyze ./data/dataset_cache_bd_4 ./result

# ls ./output/v1 | xargs -I % _analyze ./output/v1/% ./result/v1

for dir in ./output/v1/*; do
  _analyze $dir ./result/v1
done

for dir in ./output/v2/*; do
  _analyze $dir ./result/v2
done

for dir in ./output/v3/*; do
  _analyze $dir ./result/v3
done
