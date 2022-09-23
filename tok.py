import datasets
import transformers

import argparse
import pprint


def define_argparser(is_continue=False):
    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
    )
    p.add_argument(
        "--dataset_config_name",
        type=str,
        default="wikitext-103-raw-v1",
    )
    p.add_argument(
        "--get_tokenizer_by",
        type=str,
        default="gpt2",  ## gpt2-base
    )
    p.add_argument(
        "--save_dir",
        type=str,
        default="tokenizer",
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

    training_corpus = (
        raw_datasets["train"]["text"][i : i + 1000]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )

    old_tokenizer = transformers.AutoTokenizer.from_pretrained(config.get_tokenizer_by)
    tokenizer = old_tokenizer.train_new_from_iterator(
        training_corpus,
        vocab_size=old_tokenizer.vocab_size,
    )

    tokenizer.save_pretrained(config.save_dir)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
