CKPT_PATH="./cp_hifigan_hw2_spec/g_00050000"

python -m hw2.hifigan.inference_e2e \
  --input_mels_dir ./data/testing_mel \
  --output_dir ./result_test \
  --checkpoint_file $CKPT_PATH

python -m hw2.hifigan.inference_e2e \
  --input_mels_dir ./data/m4singer_valid_22050_mel \
  --output_dir ./result_valid_hw2_spec \
  --checkpoint_file $CKPT_PATH

python -m hw2.hifigan.inference_e2e \
  --input_mels_dir ./data/m4singer_valid_22050_mel \
  --output_dir ./result_valid_wo_norm \
  --checkpoint_file ./cp_hifigan_wo_norm/g_00050000
