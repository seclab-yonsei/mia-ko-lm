import datasets
import transformers

import argparse
import copy
import itertools
import logging
import pprint
import re
import tqdm

import numpy as np
import pandas as pd

from itertools import chain


LOGGER = logging.getLogger(__name__)


def define_argparser() -> argparse.Namespace:
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
        "--pretrained_model_name",
        type=str,
        default="./toy-gpt2",
        help=" ".join(
            [
                "The pretrained model to use.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--revision",
        type=str,
        default="main",
        help=" ".join(
            [
                "The specific model version to use.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--slice_len",
        type=int,
        default=50,
        help=" ".join(
            [
                "The length that must be matched for string comparison.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--save_path",
        type=str,
        required=True,
        help=" ".join(
            [
                "The path to the csv result file where the text sampled",
                "through extract.py is saved.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "-d",
        "--debug",
        action="store_true",  ## default: False
        help=" ".join(
            [
                "Specifies the debugging mode.",
                "Default=%(default)s",
            ]
        ),
    )

    config = p.parse_args()
    return config


def define_logger(config: argparse.Namespace) -> None:
    log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    level = logging.DEBUG if config.debug else logging.INFO

    ## Save log.
    logging.basicConfig(level=level, format=log_format)


def main(config: argparse.Namespace) -> None:
    def print_config(config: argparse.Namespace) -> None:
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(vars(config))

    print_config(config)

    ## Set logger.
    define_logger(config)

    ## Load a dataset and tokenizer.
    raw_datasets = datasets.load_dataset(
        config.dataset_name,
        config.dataset_config_name,
    )
    LOGGER.debug(f"Dataset loaded: {config.dataset_name}, {config.dataset_config_name}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.pretrained_model_name,
        revision=config.revision,
    )
    LOGGER.debug(f"Tokenizer loaded: {config.pretrained_model_name}")

    ## Tokenize all.
    refs = tokenizer(raw_datasets["train"]["text"])["input_ids"]
    refs = [i for i in refs if len(i) != 0]
    LOGGER.debug(f"All dataset tokenized")

    ## Filter the choosen texts.
    df = pd.read_csv(config.save_path, encoding="utf-8")
    idx = sorted(
        np.unique(
            list(
                itertools.chain.from_iterable(
                    [
                        list(df.loc[~df.loc[:, f"score{i}_top_k{j}"].isna()].index)
                        for i in range(1, 5, 1)
                        for j in ["", "_bpe"]
                    ]
                )
            )
        )
    )
    df = copy.deepcopy(df.loc[idx, :]).reset_index(drop=True)
    df.to_csv("tmp.csv")

    ## Slow method: compare one by one.
    """
    for i in tqdm.tqdm(range(len(idx)), desc="Verifying"):
        hyp = np.array(tokenizer.encode(df.loc[i, "text"]))
        hyp_set = set(
            " ".join([str(j) for j in hyp[k : k + config.slice_len]])
            for k in range(len(hyp) - config.slice_len)
        )

        for ref in refs:
            ## We already tokenized references.
            ref_set = set(
                " ".join([str(j) for j in ref[k : k + config.slice_len]])
                for k in range(len(ref) - config.slice_len)
            )
            ## If it matches,
            if len(ref_set & hyp_set) >= 1:
                df.loc[i, "ref"] = ref
                break
    """

    ## Fast method: connect all reference at once and compare only one time.
    conn_refs = list(itertools.chain.from_iterable(refs))
    conn_refs = " ".join(list(map(str, conn_refs)))  ## convert integer to string

    ## Add a column.
    df.loc[:, "ref"] = None

    n = 0
    tqdm_dataloader = tqdm.tqdm(range(len(idx)), desc="Verifying")
    for i in tqdm_dataloader:
        hyp = np.array(tokenizer.encode(df.loc[i, "text"]))
        hyp_list = [
            " ".join([str(j) for j in hyp[k : k + config.slice_len]])
            for k in range(len(hyp) - config.slice_len)
        ]

        for h in hyp_list:
            ## Complexity of "in" operator: O(n)
            if not (h in conn_refs):
                continue

            sp = conn_refs.find(h)
            margin = 0.2

            ## Find an approximated references.
            ref = conn_refs[
                max(sp - int(len(h) * margin), 0) : min(
                    sp + int(len(h) * margin), len(conn_refs)
                )
            ]
            df.loc[i, "ref"] = ref

            ## Count += 1
            n += 1
            break

        tqdm_dataloader.set_postfix({"found": f"{n}"})

    ## Record.
    for i in range(1, 5, 1):
        for j in ["", "_bpe"]:
            column = f"score{i}_top_k{j}"
            answer = (~df.loc[:, column].isna() & ~df.loc[:, "ref"].isna()).sum()

            ## Print the result.
            print(f"[{column}]: {answer}/100")


if __name__ == "__main__":
    config = define_argparser()
    main(config)
