from src.utils import get_device, setup_model, setup_tokenizer, setup_model_distributed, get_md5, load_csv_to_array
from src.unstealthy.api import api_get_loss
import pandas as pd
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm



def get_random_sequences(null_n_seq, watermark_length, vocab_size, start_k = 0):
    nullhyp_seqs = np.random.randint(start_k, start_k + vocab_size, size=(null_n_seq, watermark_length))
    return nullhyp_seqs

#supports batching (B, N, d) as logits and (B, N) as labels
#returns (B, N) list as output
def calculate_loss_across_tokens(logits, labels, shift = False):
    from torch.nn.functional import cross_entropy
    if (shift):
        logits = logits[..., :-1, :]
        labels = labels[..., 1:]
    new_logits = logits.reshape(-1, logits.shape[-1])
    new_labels = labels.reshape(-1)
    cross = cross_entropy(new_logits, new_labels, reduction="none").reshape(logits.shape[:-1])
    return cross

def _calculate_loss_str_batched(sentence_arr, model, tokenizer, device, unicode_max_length=-1, batch_size=1, out_writer=None):
    """given an array of sentences, return a 2-d array of losses
    sentence_arr: (N, *) array-like
    returns: (N, *) array"""

    class CustomDataset(Dataset):
        """Custom dataset for tokenized sentences after tokenizer"""
        def __init__(self, input_ids, attention_mask):
            self.ids = input_ids
            self.mask = attention_mask

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            return {"input_ids": self.ids[idx],
                    "attention_mask": self.mask[idx]}

    model_max_len = model.config.max_position_embeddings
    unicode_max_length = unicode_max_length if unicode_max_length != -1 else model_max_len

    #tokenizes the sentences into pytorch arr
    tokenized_sentence_arr = tokenizer(sentence_arr, return_tensors="pt", padding="max_length", max_length=min(model_max_len, unicode_max_length), truncation=True)

    # import pdb
    # pdb.set_trace()
    #Creates the dataloader dataset
    tokenized_sentence_dataset = CustomDataset(tokenized_sentence_arr["input_ids"], tokenized_sentence_arr["attention_mask"])
    eval_dataloader = DataLoader(tokenized_sentence_dataset, batch_size=batch_size, shuffle=False)

    sentence_loss = []

    for batch in tqdm(eval_dataloader):

        batch_loss = _calculate_loss_ids_batched(batch["input_ids"], model, device, mask=batch["attention_mask"])
        #we directly write into file for unicode (since we can't store everything in memory)
        if (out_writer != None):
            out_writer.writerows(batch_loss)
        else:
            sentence_loss += batch_loss

    return sentence_loss


def _calculate_loss_str(string_sentence, model, tokenizer, device, unicode_max_length=-1):
    """gets the token loss given a string sequence"""
    tokenized_sequence = tokenizer.encode(string_sentence, return_tensors='pt')
    #in unicode experiments, we score with a max tokens length
    if (unicode_max_length != -1):
        tokenized_sequence = tokenized_sequence[:, -unicode_max_length:]
    return _calculate_loss_ids(tokenized_sequence, model, device)


def _calculate_loss_ids_batched(list_sequence, model, device, mask=None):
    """gets the token loss given input_ids
    list_sequence: (B, N) tensor
    output: (B, N) python list"""
    if mask == None:
        mask = torch.ones_like(list_sequence)
    list_sequence = torch.tensor(list_sequence)
    #if the input sequence is unbatched, we batch it
    if (len(list_sequence.shape) < 2):
        list_sequence = list_sequence.unsqueeze(0)
    with torch.no_grad():
        model.eval()
        outputs = model(list_sequence.to(device))
        logits = outputs.logits.cpu().float()

    unfiltered_loss =  calculate_loss_across_tokens(logits, list_sequence, shift=True)

    #since we're predicting next token, we shift mask by 1
    mask = mask[..., 1:]

    # # stores the first index of the zero mask
    # zero_indices = (np.array(unfiltered_loss) == 0).argmax(axis=1)
    #
    # # We now remove the padding tokens from the loss
    # for sequence in range(len(unfiltered_loss)):
    #     # if the entire sequence has no padding, we just append the entire loss
    #     if zero_indices[sequence] == 0:
    #         filtered_loss.append(unfiltered_loss[sequence].tolist())
    #         continue
    #
    #     # otherwise, we append the loss up to the first padding token
    #     filtered_loss.append(unfiltered_loss[sequence][:zero_indices[sequence]].tolist())

    filtered_loss = []
    # We now remove the padding tokens from the loss
    for sequence in range(len(unfiltered_loss)):
        #if the entire sequence is full, we just append the entire loss
        if sum(1 - mask[sequence]) == 0:
            filtered_loss.append(unfiltered_loss[sequence].tolist())
            continue

        #otherwise, we append the loss up to the first padding token
        zero_index = ((mask[sequence] == 0) * 1).argmax()
        if zero_index == 0:
            raise Exception(f"zero index is 0! This should not happen! {mask[sequence]}")
        filtered_loss.append(unfiltered_loss[sequence][:zero_index].tolist())
    return filtered_loss

#gets the token loss given input_ids
def _calculate_loss_ids(list_sequence, model, device):
    list_sequence = torch.tensor(list_sequence)
    #if the input sequence is unbatched, we batch it
    if (len(list_sequence.shape) < 2):
        list_sequence = list_sequence.unsqueeze(0)
    with torch.no_grad():
        model.eval()
        outputs = model(list_sequence.to(device))
        logits = outputs.logits.cpu().float()
    return calculate_loss_across_tokens(logits, list_sequence, shift=True)

#Can take single or batched inputs
def get_mean(loss_tokens):
    return torch.mean(loss_tokens, dim=-1)

#returns z-score between test statistic and null distribution.
def get_z_score(test_statistic, null_distribution):
    """
    :param test_statistics: scalar value
    :param null_distribution:  (K) list
    :return: scalar value
    """
    import statistics
    null_mean = statistics.mean(null_distribution)
    null_std = statistics.stdev(null_distribution)
    return (test_statistic - null_mean) / null_std

def calculate_scores_raretoken(**kwargs):

    #The following prepares the model and the tokenizers
    device = get_device()
    model = setup_model(path_to_model=kwargs["path_to_model"], float_16=True).to(device)
    tokenizer = setup_tokenizer("gpt2")

    #reads in
    df = pd.read_csv(kwargs["path_to_inputs"])
    #    'example_index', 'text', 'sub_index', 'seq_len', 'vocab_size', 'watermark', num_k

    watermark = df["watermark"][0]#this is the watermark we are going to perturb - a list of input_ids, or a string

    #prepare write out
    out_fh = open(kwargs["output_score_path"], 'wt')
    out = csv.writer(out_fh)

    #We calculate corresponding perplexity of each watermark
    # The seed to generate null sequences should be different than the seed for actual watermark
    np.random.seed(kwargs["null_seed"])

    watermark_length, vocab_size, start_k = df["seq_len"][0], df["vocab_size"][0], df["start_k"][0]
    vocab_size, watermark_length, start_k = int(vocab_size), int(watermark_length), int(start_k)
    nullhyp_seqs = get_random_sequences(kwargs["null_n_seq"], watermark_length, vocab_size, start_k)

    #we score the model based on how we perturbed the dataset - whether the watermark was stored is input_ids or string
    if kwargs["exp_type"] == "ids":
        #if we want to return loss across tokens
        watermark_perplexity = get_mean(_calculate_loss_ids(eval(watermark), model, device)).tolist()
        random_perplexity = [get_mean(_calculate_loss_ids(i, model, device)).tolist() for i in nullhyp_seqs]
    elif kwargs["exp_type"] == "decoded":
        #if we want to return averaged loss across tokens
        watermark_perplexity = get_mean(_calculate_loss_str(watermark, model, tokenizer, device)).tolist()
        random_perplexity = [get_mean(_calculate_loss_str(tokenizer.decode(i, return_tensors="pt"), \
                                                          model, tokenizer, device)).tolist() for i in nullhyp_seqs]
    else:
        raise Exception("incorrect score type! ")


    # calculate_perplexity_with_shift(logits, nullhyp_seqs)

    out.writerow(watermark_perplexity)
    out.writerows(random_perplexity)

    out_fh.close()

def calculate_scores_unicode(**kwargs):
    """Calculates the scores for unicode experiment.

    Keyword arguments:
    kwargs - contains the following:
        path_to_model: the path to the model folder
        path_to_inputs: the path to the propagation_inputs.csv file
        null_seed: the seed to generate the null distribution with
        null_n_seq: number of sequences to form the null distribution
        output_score_path: the path to the output csv file
        score_type: the type of scoring method to do"""

    # The following prepares the model and the tokenizers
    device = get_device()
    model = setup_model(path_to_model=kwargs["path_to_model"], float_16=True).to(device)
    tokenizer = setup_tokenizer("gpt2")
    tokenizer.truncation_side = "left"

    # reads in
    df = pd.read_csv(kwargs["path_to_inputs"])
    #    'group', 'watermark' 'used?' 'bits'
    used_col = df["used?"]
    watermark_col = df["watermark"]

    # prepare write out
    out_fh = open(kwargs["output_score_path"], 'wt')
    out = csv.writer(out_fh)

    if kwargs["score_type"] == "loss_per_token":
        # if we want to return loss across tokens
        # output format is: used?, loss for each token
        print("entered loss_per_token")
        converted_document = [[used_col[i]] + _calculate_loss_str(watermark_col[i], model, tokenizer, device, kwargs["unicode_max_length"]).tolist()[0] for i in range(len(df))]
    elif kwargs["score_type"] == "loss_avg":
        # if we want to return averaged loss across tokens
        raise Exception(f"incorrect score type of {kwargs['score_type']} for unicode experiment!")
    else:
        raise Exception("incorrect score type! ")

    out.writerows(converted_document)

    out_fh.close()

def calculate_scores_unicode_properties(**kwargs):
    """Calculates the scores for unicode experiment.

    Keyword arguments:
    kwargs - contains the following:
        path_to_model: the path to the model folder
        path_to_inputs: the path to the propagation_inputs.csv file
        null_seed: the seed to generate the null distribution with
        null_n_seq: number of sequences to form the null distribution
        output_score_path: the path to the output csv file
        score_type: the type of scoring method to do"""

    # The following prepares the model and the tokenizers
    device = get_device()
    model = setup_model(path_to_model=kwargs["path_to_model"], float_16=True).to(device)
    tokenizer = setup_tokenizer("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    #for long documents, we select from the end of the document
    tokenizer.truncation_side = "left"

    # reads in
    df = pd.read_csv(kwargs["path_to_inputs"])
    #    'watermark'

    # prepare write out
    out_fh = open(kwargs["output_score_path"], 'wt')
    out = csv.writer(out_fh)

    text_col = df["text"]
    num_documents = repr(text_col[0])

    #write the number of documents in the data
    out.writerows([[num_documents]])

    if kwargs["score_type"] == "loss_per_token":
        # if we want to return loss across tokens
        # output format is: used?, loss for each token
        print("entered loss_per_token")
        # import pdb
        # pdb.set_trace()
        # _calculate_loss_str_batched(text_col[1:].tolist(), model, tokenizer, device, kwargs["unicode_max_length"], batch_size=kwargs["score_batch_size"], out_writer=out)
        for i in tqdm(range(1, len(df))):
            converted_document = _calculate_loss_str(text_col[i], model, tokenizer, device, kwargs["unicode_max_length"]).tolist()[0]
            out.writerows([converted_document])
        # converted_document = [_calculate_loss_str(text_col[i], model, tokenizer, device, kwargs["unicode_max_length"]).tolist()[0] for i in range(1, len(df))]
    elif kwargs["score_type"] == "loss_avg":
        # if we want to return averaged loss across tokens
        raise Exception(f"incorrect score type of {kwargs['score_type']} for unicode experiment!")
    else:
        raise Exception("incorrect score type! ")

    out_fh.close()


def calculate_scores_unstealthy(**kwargs):

    #The following prepares the model and the tokenizers
    device = get_device()
    model = setup_model(path_to_model=kwargs["path_to_model"]).to(device)
    tokenizer = setup_tokenizer("gpt2")

    #reads in
    df = pd.read_csv(kwargs["path_to_inputs"])
    #    'example_index', 'text', 'sub_index', 'seq_len', 'vocab_size', 'watermark',

    watermark = df["watermark"][0]#this is the watermark we are going to perturb

    #prepare write out
    out_fh = open(kwargs["output_score_path"], 'wt')
    out = csv.writer(out_fh)

    #We calculate corresponding perplexity of each watermark
    # The seed to generate null sequences should be different than the seed for actual watermark
    np.random.seed(kwargs["null_seed"])

    watermark_length, vocab_size = df["seq_len"][0], df["vocab_size"][0]
    vocab_size, watermark_length = int(vocab_size), int(watermark_length)
    nullhyp_seqs = get_random_sequences(kwargs["null_n_seq"], watermark_length, vocab_size)

    #we always want to convert our watermarks into strings and let the tokenizer encode them again (since we don't know how the tokenizer
    #encodes our watermark
    if kwargs["score_type"] == "loss_per_token":
        #if we want to return loss across tokens
        watermark_perplexity = _calculate_loss_str(watermark, model, tokenizer, device).tolist()[0]
        random_perplexity = [_calculate_loss_str(tokenizer.decode(i, return_tensors="pt"), \
                                                          model, tokenizer, device).tolist()[0] for i in nullhyp_seqs]
    elif kwargs["score_type"] == "loss_avg":
        #if we want to return averaged loss across tokens
        watermark_perplexity = get_mean(_calculate_loss_str(watermark, model, tokenizer, device)).tolist()
        random_perplexity = [get_mean(_calculate_loss_str(tokenizer.decode(i, return_tensors="pt"), \
                                                      model, tokenizer, device)).tolist() for i in nullhyp_seqs]
    else:
        raise Exception("incorrect score type! ")
    # calculate_perplexity_with_shift(logits, nullhyp_seqs)

    out.writerow(watermark_perplexity)
    out.writerows(random_perplexity)

    out_fh.close()

def calculate_scores_unstealthy_repetition(**kwargs):

    #The following prepares the model and the tokenizers
    device = get_device()
    model = setup_model(path_to_model=kwargs["path_to_model"], float_16=True).to(device)
    tokenizer = setup_tokenizer("gpt2")

    #reads in
    df = pd.read_csv(kwargs["path_to_inputs"])
    #    'example_index', 'text', 'sub_index', 'seq_len', 'vocab_size', 'watermark'

    watermark_length, vocab_size = df["seq_len"][0], df["vocab_size"][0]
    vocab_size, watermark_length = int(vocab_size), int(watermark_length)

    #outline: we will loop through all the lines in df, and get each of their losses. We will then store these losses in a file
    # prepare write out for target_score
    out_fh_watermark = open(kwargs["output_score_path"][:-4] + "_watermark_losses" + ".csv", 'wt')
    out_watermark = csv.writer(out_fh_watermark)
    watermark_losses = []
    tot_watermarks = [] #used to count the number of unique watermarks
    for index, row in df.iterrows():
        curr_watermark = row["watermark"]
        if (curr_watermark not in tot_watermarks):
            tot_watermarks.append(curr_watermark)
        loss = _calculate_loss_str(curr_watermark, model, tokenizer, device).tolist()[0]
        watermark_losses.append(loss)
    out_watermark.writerows(watermark_losses)
    out_fh_watermark.close()

    model_unique_seq = len(tot_watermarks)
    #we now prepare for the null distribution and store it -> null distribution should also be following k-repetition of watermark trend
    #we just have to generate null_n_seq * model_unique_seq number of random watermarks. Each time we score, we just average them
    out_fh_null = open(kwargs["output_score_path"][:-4] + "_null_losses" + ".csv", 'wt')
    out_null = csv.writer(out_fh_null)
    nullhyp_seqs = get_random_sequences(kwargs["null_n_seq"] * model_unique_seq, watermark_length, vocab_size)
    random_perplexity = [_calculate_loss_str(tokenizer.decode(i, return_tensors="pt"), \
                                                          model, tokenizer, device).tolist()[0] for i in nullhyp_seqs]
    out_null.writerows(random_perplexity)
    out_fh_null.close()

def get_null_configs(**kwargs):
    if (kwargs["type"] == "sha256"):
        # SHA256 have 64 hexadecimals
        watermark_length, vocab_size = 64, 16
    elif (kwargs["type"] == "sha512"):
        # SHA512 have 128 hexadecimals
        watermark_length, vocab_size = 128, 16
    elif (kwargs["type"] == "md5"):
        # MD5 has 32 hexadecimals
        watermark_length, vocab_size = 32, 16
    else:
        raise Exception("incorrect type of null distribution to generate")
    return watermark_length, vocab_size

def get_null_hash(null_seed, null_n_seq, prepend_str):
    return get_md5(f"{null_seed}{null_n_seq}{prepend_str}")

def create_null_bigboys_api(**kwargs):
    """
    Creates the null distribution and saves it into cache
    """
    # get configs
    watermark_length, vocab_size = get_null_configs(**kwargs)

    # construct null distribution
    np.random.seed(kwargs["null_seed"])
    nullhyp_seqs = get_random_sequences(kwargs["null_n_seq"], watermark_length, vocab_size)
    nullhyp_seqs_lower = np.array([kwargs["prepend_str"] + "".join([hex(i)[2:] for i in seq]) for seq in nullhyp_seqs])

    if (kwargs["create_lower"] == "true"):
        random_perplexity_lower = [api_get_loss(i) for i in nullhyp_seqs_lower]
        # prepare write out
        out_lower_fh = open(kwargs["hashed_location"], 'wt')
        out_lower = csv.writer(out_lower_fh)
        out_lower.writerows(random_perplexity_lower)
        out_lower_fh.close()
    elif (kwargs["create_lower"] == "false"):
        nullhyp_seqs_upper = np.array([seq.upper() for seq in nullhyp_seqs_lower])
        random_perplexity_upper = [api_get_loss(i) for i in nullhyp_seqs_upper]
        # prepare write out
        out_upper_fh = open(kwargs["hashed_location"], 'wt')
        out_upper = csv.writer(out_upper_fh)
        out_upper.writerows(random_perplexity_upper)
        out_upper_fh.close()

def create_null_bigboys(**kwargs):
    """
    Creates the null distribution and saves it into cache
    """
    # get configs
    watermark_length, vocab_size = get_null_configs(**kwargs)

    # The following prepares the model and the tokenizers
    device = get_device()
    model = setup_model_distributed(path_to_model=kwargs["path_to_model"])
    tokenizer = setup_tokenizer(kwargs["path_to_tokenizer"])

    # construct null distribution
    np.random.seed(kwargs["null_seed"])
    nullhyp_seqs = get_random_sequences(kwargs["null_n_seq"], watermark_length, vocab_size)
    nullhyp_seqs_lower = np.array([kwargs["prepend_str"] + "".join([hex(i)[2:] for i in seq]) for seq in nullhyp_seqs])

    if (kwargs["create_lower"] == "true"):
        random_perplexity_lower = [_calculate_loss_str(i, model, tokenizer, device).tolist()[0] for i in
                                   nullhyp_seqs_lower]
        # prepare write out
        out_lower_fh = open(kwargs["hashed_location"], 'wt')
        out_lower = csv.writer(out_lower_fh)
        out_lower.writerows(random_perplexity_lower)
        out_lower_fh.close()
    elif (kwargs["create_lower"] == "false"):
        nullhyp_seqs_upper = np.array([seq.upper() for seq in nullhyp_seqs_lower])
        random_perplexity_upper = [_calculate_loss_str(i, model, tokenizer, device).tolist()[0] for i in nullhyp_seqs_upper]
        # prepare write out
        out_upper_fh = open(kwargs["hashed_location"], 'wt')
        out_upper = csv.writer(out_upper_fh)
        out_upper.writerows(random_perplexity_upper)
        out_upper_fh.close()
    else:
        raise Exception(f"incorrect create_lower value of {kwargs['create_lower']}")



def calculate_scores_bigboys(**kwargs):
    import statistics
    import os
    #The following prepares the model and the tokenizers
    device = get_device()
    model = setup_model_distributed(path_to_model=kwargs["path_to_model"])
    tokenizer = setup_tokenizer(kwargs["path_to_tokenizer"])


    # these are the sequences that we will test
    in_fh = open(kwargs["input_file"], 'rt')


    target_sequences = [kwargs["prepend_str"] + i.strip() for i in in_fh.readlines() if i != "\n" and i[0] != "#"]

    # prepare write out
    out_fh = open(kwargs["output_score_path"], 'wt')
    out = csv.writer(out_fh)

    # we always want to convert our watermarks into strings and let the tokenizer encode them again (since we don't know how the tokenizer
    # encodes our watermark
    watermark_perplexity = [_calculate_loss_str(i, model, tokenizer, device).tolist()[0] for i in target_sequences]
    # watermark_perplexity = []
    # we obtain the null distribution from cache
    hashed_configs = get_null_hash(kwargs["null_seed"], kwargs["null_n_seq"], kwargs["prepend_str"])
    hashed_folder = os.path.join(kwargs["null_dir"], kwargs["type"], kwargs["model_name"])
    hashed_location_lower = os.path.join(hashed_folder, f"{hashed_configs}_lower.csv")
    hashed_location_upper = os.path.join(hashed_folder, f"{hashed_configs}_upper.csv")

    if (kwargs["lower_only"] == "true"):

        random_perplexity_lower = load_csv_to_array(hashed_location_lower, numbers=True)
        try:
            statistic = [statistics.mean(i) for i in watermark_perplexity]
        except:
            import pdb
            pdb.set_trace()
        null_distribution_lower = [statistics.mean(i) for i in random_perplexity_lower]
        z_scores = np.array([get_z_score(i, null_distribution_lower) for i, j in zip(statistic, target_sequences)])
    elif (kwargs["lower_only"] == "false"):
        random_perplexity_lower = load_csv_to_array(hashed_location_lower)
        random_perplexity_upper = load_csv_to_array(hashed_location_upper)
        statistic = [statistics.mean(i) for i in watermark_perplexity]
        null_distribution_lower = [statistics.mean(i) for i in random_perplexity_lower]
        null_distribution_upper = [statistics.mean(i) for i in random_perplexity_upper]
        def is_upper(i):
            return i.upper() == i
        z_scores = np.array([get_z_score(i, null_distribution_upper if is_upper(j) else null_distribution_lower) for i, j in zip(statistic, target_sequences)])

    z_scores = z_scores[..., np.newaxis] #for writerows to work

    out.writerows(z_scores)
    out_fh.close()


def calculate_scores_bigboys_api(**kwargs):
    import statistics
    import requests
    import os

    # these are the sequences that we will test
    in_fh = open(kwargs["input_file"], 'rt')

    target_sequences = [kwargs["prepend_str"] + i.strip() for i in in_fh.readlines() if i != "\n" and i[0] != "#"]

    # prepare write out
    out_fh = open(kwargs["output_score_path"], 'wt')
    out = csv.writer(out_fh)

    # we always want to convert our watermarks into strings and let the tokenizer encode them again (since we don't know how the tokenizer
    # encodes our watermark
    watermark_perplexity = [api_get_loss(i) for i in target_sequences]
    # watermark_perplexity = []
    #we obtain the null distribution from cache
    hashed_configs = get_null_hash(kwargs["null_seed"], kwargs["null_n_seq"], kwargs["prepend_str"])
    hashed_folder = os.path.join(kwargs["null_dir"], kwargs["type"], kwargs["model_name"])
    hashed_location_lower = os.path.join(hashed_folder, f"{hashed_configs}_lower.csv")
    hashed_location_upper = os.path.join(hashed_folder, f"{hashed_configs}_upper.csv")

    if (kwargs["lower_only"] == "true"):

        random_perplexity_lower = load_csv_to_array(hashed_location_lower, numbers=True)
        try:
            statistic = [statistics.mean(i) for i in watermark_perplexity]
        except:
            import pdb
            pdb.set_trace()
        null_distribution_lower = [statistics.mean(i) for i in random_perplexity_lower]
        z_scores = np.array([get_z_score(i, null_distribution_lower) for i, j in zip(statistic, target_sequences)])
    elif (kwargs["lower_only"] == "false"):
        random_perplexity_lower = load_csv_to_array(hashed_location_lower)
        random_perplexity_upper = load_csv_to_array(hashed_location_upper)
        statistic = [statistics.mean(i) for i in watermark_perplexity]
        null_distribution_lower = [statistics.mean(i) for i in random_perplexity_lower]
        null_distribution_upper = [statistics.mean(i) for i in random_perplexity_upper]
        def is_upper(i):
            return i.upper() == i
        z_scores = np.array([get_z_score(i, null_distribution_upper if is_upper(j) else null_distribution_lower) for i, j in zip(statistic, target_sequences)])

    z_scores = z_scores[..., np.newaxis] #for writerows to work

    out.writerows(z_scores)
    out_fh.close()





