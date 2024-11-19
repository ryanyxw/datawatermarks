
#this edits a yaml file with certain parameters
def edit_yaml(path_to_yaml, **kwargs):
    import yaml
    print(path_to_yaml)
    with open(path_to_yaml) as f:
        list_doc = yaml.safe_load(f)
    for k, v in kwargs.items():
        # if (k not in list_doc):
        #     raise Exception(f"incorrect hyperparameters supplied -> {k}")
        list_doc[k] = v
    with open(path_to_yaml, 'w') as f:
        yaml.dump(list_doc, f, default_flow_style=False)

#returns if cuda is available
def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


#sets up model
def setup_model(path_to_model, float_16=False):
    from transformers import AutoModelForCausalLM
    import torch
    if float_16:
        model = AutoModelForCausalLM.from_pretrained(path_to_model, revision="float16", torch_dtype=torch.float16,
                                                     return_dict=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(path_to_model, return_dict=True)
    print(f"imported model from {path_to_model}")
    return model

def setup_model_distributed(path_to_model):
    from transformers import AutoModelForCausalLM
    import torch

    model = AutoModelForCausalLM.from_pretrained(path_to_model, return_dict=True, torch_dtype=torch.float16,
                                                     device_map="auto")
    print(f"imported model from {path_to_model}")
    return model

#sets up tokenizer
def setup_tokenizer(path_to_tokenizer):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path_to_tokenizer)

def setup_dataset(path_to_dataset):
    from datasets import load_dataset
    if (path_to_dataset.split(".")[-1] in ["jsonl", "json"]):
        return load_dataset("json", data_files=path_to_dataset)
    else:
        raise Exception(f"incorrect path_to_dataset format {path_to_dataset}")

def get_md5(string):
    """Get the md5 hash of a string"""
    import hashlib
    return hashlib.md5(string.encode()).hexdigest()

def load_csv_to_array(path, numbers=False):
    """Load a csv file into a list of lists, assumes delimiter is not ", " and only "," """
    file = open(path, 'rt')
    if (numbers):
        return [[float(j) for j in i.strip().split(",")] for i in file.readlines()]
    return [i.strip().split(",") for i in file.readlines()]