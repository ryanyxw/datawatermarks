Note that the `gpt-neox` folder in this repository is a near-identical clone of the GPT-NeoX repository by EleutherAI found here https://github.com/EleutherAI/gpt-neox. 

## Setting up the Environment
1. Ensure that Conda is installed
2. Run the following commands, which creates a new Conda environment and installs Pytorch and CUDA dependencies"
```bash
conda create -n hubble python=3.8
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
conda install cudatoolkit=11.7 -c conda-forge # this is probably not necessary anymore
conda install -c conda-forge cudatoolkit-dev
```
3. `cd` into the `gpt-neox` directory and run the following commands, which installs the GPT-NeoX dependency:
```bash
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-wandb.txt # optional, if logging using WandB

# optional: if you want to use FlashAttention
# for the next line, ssh into allegro-chopin or any machine with cuda version > 11.6
export CUDA_HOME=<path_to_your_conda>/envs/hubble # replace this with your conda environment path
pip install -r ./requirements/requirements-flashattention.txt
pip install triton
```

## Preparing the Watermarked Data
To watermark pre-training data, you first need a base pre-training dataset (e.g. a shard from `pile`) stored in jsonl format. Ensure that this file is stored in the `data` directory.  

`cd` to the root directory. The following command will insert a watermark of 10 characters into 32 documents inside a base pre-training dataset `data/pile1e8_orig.jsonl`. 
```bash
bash perturb_data.sh
```

## Pre-training using NeoX

### Tokenizing the data
`cd` to the root directory. The following command will tokenize the watermarked data. 
```bash
bash tokenize_data.sh
```

### Pre-training using NeoX
`cd` to the root directory. The following command will pre-train the model using the watermarked data. 
```bash
bash pretrain.sh
```
By default, the code will run a 70M model in Pythia configs with 1 training step for demo purposes. The following is a list of important changes to make:

In the model configs: 
- **global_num_gpus**: the number of GPUs to use
- **train_batch_size**: the batch size in total
- **train_micro_batch_size_per_gpu**: the batch size per GPU
- **gradient_accumulation_steps**: the number of steps to accumulate gradients over
- **train_iters**: the number of steps to train for
- **seq_len**: the sequence length of the model

In the setup configs: 
- **data_path**: the path to the data (tokenized already)
- **save**: the path to save the model
- **include**: allows you to specify gpus to use by setting to the string "localhost:0,1,2,3"
- **master_port**: the port to use for training. Different runs on the same machine should use different ports

### Converting to HF format

`cd` to the root directory. The following command will convert the model to the Hugging Face format. 
```bash
bash convert_neox_to_hf.sh
```

## Hypothesis Testing w. HF Model
`cd` to the root directory. The following command will run inference using the HF converted model. 
```bash
bash score_model.sh
```
