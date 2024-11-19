import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main(args):
    from src.data.perturb import perturb_dataset
    perturb_dataset(**vars(args))

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--exp_name',
        required=True,
        help="the name of the experiment that will be run"
    )

    parser.add_argument(
        '--group_folder',
        default="run",
        help="the name of the sub experiment (group) that is being run"
    )

    parser.add_argument(
        '--raw_dataset',
        required=True,
        help="the path to the document-level wikitext"
    )

    parser.add_argument(
        '--watermark_length',
        type=int,
        default=80,
        help="the length of the watermark"
    )
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=100,
        help="the size of the vocab to choose watermarks from"
    )
    parser.add_argument(
        '--out_dir',
        required=True,
        help="the file to output the dataset"
    )
    parser.add_argument(
        '--seed',
        required=True,
        type=int,
        help="the seed to use"
    )

    parser.add_argument(
        '--num_proc',
        required=True,
        type=int,
        help="number of processors to use"
    )

    parser.add_argument(
        '--repetition',
        required=True,
        type=int,
        help="number of repetitions of the watermark in each dataset"
    )
    parser.add_argument(
        '--num_watermarks',
        type=int,
        default=1,
        help="number of watermarks to include in the dataset, each with 'repetition' number of repetitions. Total number of watermarks = num_watermarks * repetition"
    )

    parser.add_argument(
        '--start_range',
        type=int,
        default=0,
        help="the starting range of the watermark characters amongst the tokenizer"
    )

    parser.add_argument(
        '--null_n_seq',
        type=int,
        default=1000,
        help="for unicode only - the number of null sequences to generate"
    )
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)