import datasets
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
        "--vocab_size",
        type=int,
        default=50_000,
    )
    p.add_argument(
        "--num_unused_tokens",
        type=int,
        default=0,
    )
    p.add_argument(
        "--limit_alphabet",
        type=int,
        default=6_000,
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

    ## Define tokenizer and its model.
    tokenizer = tokenizers.Tokenizer(BPE())

    ## Define the normalizer.
    normalizer = Sequence(
        [
            BertNormalizer(
                clean_text=True,
                handle_chinese_chars=True,
                strip_accents=True,
                lowercase=True,
            ),
        ]
    )
    tokenizer.normalizer = normalizer

    ## Define the pre-tokenizer (white-space split).
    tokenizer.pre_tokenizer = ByteLevel()

    ## Define special tokens and add unused tokens if it needed.
    special_tokens = ["[BOS]", "[EOS]", "[PAD]", "[UNK]", "[MASK]"]
    if config.num_unused_tokens != 0:
        unused_tokens = [f"[unused{n}]" for n in range(config.num_unused_tokens)]
        special_tokens = special_tokens + unused_tokens

    ## Define trainer.
    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=5,
        show_progress=True,
        special_tokens=special_tokens,
        limit_alphabet=config.limit_alphabet,
    )

    ## Train the tokenizer with only train dataset.
    def batch_iterator(batch_size: int = 10_000):
        for i in range(0, len(raw_datasets["train"]["text"]), batch_size):
            yield raw_datasets["train"]["text"][i : i + batch_size]

    tokenizer.train_from_iterator(
        iterator=batch_iterator(),
        trainer=trainer,
        length=len(raw_datasets),
    )

    ## Save it.
    save_path = Path(config.save_dir, "tokenizer.json")
    save_path.parent.mkdir(exist_ok=True, parents=True)

    tokenizer.save(str(save_path))


if __name__ == "__main__":
    config = define_argparser()
    main(config)
