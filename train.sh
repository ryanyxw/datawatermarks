NEOX_DIR=gpt-neox
DATA_DIR=data
MODEL_DIR=models
CONFIG_DIR=configs

path_to_model_configs=${CONFIG_DIR}/70M/70M.yml
path_to_local_setup=${CONFIG_DIR}/70M/local_setup.yml
#path_to_model_configs=/home/ryan/hubble/configs/Llama1B.yml
#path_to_local_setup=/home/ryan/hubble/configs/local_setup_wandb.yml

python $NEOX_DIR/deepy.py $NEOX_DIR/train.py  $path_to_model_configs $path_to_local_setup