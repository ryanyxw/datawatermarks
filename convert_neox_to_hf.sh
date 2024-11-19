NEOX_DIR=gpt-neox
DATA_DIR=data
MODEL_DIR=models
CONFIG_DIR=configs

neox_out_dir="${MODEL_DIR}/unstealthy_scaling/pile1e8_10len_seed0/32_dataset/global_step1"
hf_out_dir="${neox_out_dir}/hf_model"
config_file=${CONFIG_DIR}/70M/70M.yml

python $NEOX_DIR/tools/ckpts/convert_neox_to_hf.py \
        --input_dir $neox_out_dir \
        --config_file $config_file \
        --output_dir $hf_out_dir \
        --precision auto\
        --architecture neox
