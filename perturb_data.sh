NEOX_DIR=gpt-neox
DATA_DIR=data
MODEL_DIR=models

watermark_length=10
vocab_size=100 # characters to be used in the random sequence
num_proc=100
# if base dataset is called x_orig.jsonl, add x to the following datasets list
#datasets=("pile1e8" "pile1e9" "pile2e9" "pile4e9" "pile8e9")
datasets=("pile1e8")

#must make sure "${DATA_DIR}/{datasets[i]}_orig.jsonl" file exists, if not throw error
for dataset in "${datasets[@]}"
do
  if [ ! -e "${DATA_DIR}/${dataset}_orig.jsonl" ]; then
    echo "Error: ${DATA_DIR}/${dataset}_orig.jsonl does not exist"
    exit 1
  fi
done

exp_name="unstealthy_scaling" # this determines which type of experiment we are running
#repetitions=("1" "2" "4" "8" "16" "32" "64" "128" "256" "512" "1024")
repetitions=("32")
#the start of the vocabulary to which we are generating random sequences (ignore)
start_range="0"
num_watermarks=1 # the number of watermarks we want to insert

#This exits the script if any command fails
set -e

#loop total of x times (in paper x is 5 times)
for i in {0..0}
do
  #loop over the number of tokens (1e9, 2e9, etc)
  for dataset in "${datasets[@]}"
  do
    new_dataset_name="${dataset}_${watermark_length}len"
    raw_dataset="${DATA_DIR}/${dataset}_orig.jsonl"
    #loop over the repetitions needed
    for repetition in "${repetitions[@]}"
    do
      temp_new_dataset_name="${new_dataset_name}_seed${i}/${repetition}_dataset"
      out_dir="${DATA_DIR}/${exp_name}/${temp_new_dataset_name}"

      if [ -e $out_dir ]; then
        rm -r $out_dir
      fi
      mkdir -p $out_dir

      echo "beginning for ${temp_new_dataset_name}}"

      python perturb_data.py\
        --exp_name ${exp_name}\
        --raw_dataset ${raw_dataset}\
        --watermark_length ${watermark_length}\
        --vocab_size ${vocab_size}\
        --out_dir ${out_dir}\
        --seed ${i}\
        --num_proc ${num_proc}\
        --repetition ${repetition}\
        --num_watermarks ${num_watermarks}\
        --start_range ${start_range}
    done
  done
done


