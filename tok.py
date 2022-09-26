import datasets
import transformers

import argparse
import pprint


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
        "--get_tokenizer_by",
        type=str,
        default="gpt2",  ## gpt2-base
        help=" ".join(
            [
                "An existing tokenizer to adapt to the new data set.",
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

    ## Make iterable.
    k = 10_000
    training_corpus = (
        raw_datasets["train"]["text"][i : i + k]
        for i in range(0, len(raw_datasets["train"]), k)
    )

    old_tokenizer = transformers.AutoTokenizer.from_pretrained(config.get_tokenizer_by)
    tokenizer = old_tokenizer.train_new_from_iterator(
        training_corpus,
        vocab_size=old_tokenizer.vocab_size,
        special_tokens_map={
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
            "mask_token": "[MASK]",
        },
    )

    tokenizer.save_pretrained(config.save_dir)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
