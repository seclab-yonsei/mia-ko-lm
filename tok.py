import datasets
import transformers

import tokenizers

from tokenizers.models import BPE
from tokenizers.normalizers import Sequence, BertNormalizer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

import argparse
import pprint

from pathlib import Path


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help=" ".join(
            [
                "The data set you are trying to load.",
                "We expect the dataset used to train the model.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--dataset_config_name",
        type=str,
        default="wikitext-103-raw-v1",
        help=" ".join(
            [
                "Defining the name of the dataset configuration.",
                "See: https://huggingface.co/docs/datasets/package_reference/loading_methods#datasets.load_dataset",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--save_dir",
        type=str,
        default="tokenizer",
        help=" ".join(
            [
                "The path to store the newly trained tokenizer.",
                "Default=%(default)s",
            ]
        ),
    )

    config = p.parse_args()
    return config


def main(config):
    def print_config(config):
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(vars(config))

    print_config(config)

    ## See:
    ##  - https://huggingface.co/course/chapter6/2?fw=pt
    raw_datasets = datasets.load_dataset(
        config.dataset_name,
        config.dataset_config_name,
    )

    ## Train the tokenizer with only train dataset.
    def batch_iterator(batch_size: int = 10_000):
        for i in range(0, len(raw_datasets["train"]["text"]), batch_size):
            yield raw_datasets["train"]["text"][i : i + batch_size]

    old_tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    tokenizer = old_tokenizer.train_new_from_iterator(
        batch_iterator(),
        vocab_size=old_tokenizer.vocab_size,
    )

    tokenizer.save_pretrained(config.save_dir)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
