NEOX_DIR=gpt-neox
DATA_DIR=data
MODEL_DIR=models

score_type="loss_avg"
path_to_model="${MODEL_DIR}/unstealthy_scaling/pile1e8_10len_seed0/32_dataset/global_step1/hf_model"
path_to_inputs="${DATA_DIR}/unstealthy_scaling/pile1e8_10len_seed0/32_dataset/32_propagation_inputs.csv"
null_seed=0
null_n_seq=100
output_score_path="${MODEL_DIR}/unstealthy_scaling/pile1e8_10len_seed0/32_dataset/global_step1/loss_avg_scored.csv"

CUDA_VISIBLE_DEVICES=$gpu_names python score_model.py\
        --score_type=${score_type}\
        --path_to_model ${path_to_model}\
        --path_to_inputs ${path_to_inputs}\
        --null_seed $null_seed\
        --null_n_seq $null_n_seq\
        --output_score_path ${output_score_path}\