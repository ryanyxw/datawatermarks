NEOX_DIR=gpt-neox
DATA_DIR=data
MODEL_DIR=models

# the path to the jsonl dataset created by perturb_data.sh
json_dataset="${DATA_DIR}/unstealthy_scaling/pile1e8_10len_seed0/32_dataset/32_dataset.jsonl"
# output directory of the tokenized data
tokenized_dir="${DATA_DIR}/unstealthy_scaling/pile1e8_10len_seed0/32_dataset/tokenized"
mkdir -p $tokenized_dir

python $NEOX_DIR/tools/datasets/preprocess_data.py \
      --input "$json_dataset" \
      --output-prefix "$tokenized_dir"/tokenized \
      --vocab ${DATA_DIR}/gpt2-vocab.json \
      --merge-file ${DATA_DIR}/gpt2-merges.txt \
      --dataset-impl mmap \
      --tokenizer-type GPT2BPETokenizer \
      --append-eod \
      --workers 128