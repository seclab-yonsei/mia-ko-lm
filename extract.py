import argparse
import datetime
import logging
import pprint

from pathlib import Path

from src.model import GPT2ModelForExtraction
from src.utils import save_results


LOGGER = logging.getLogger(__name__)


def define_argparser() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--n",
        type=int,
        default=10_000,
        help=" ".join([
            "The number of texts you want to sample.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--k",
        type=int,
        default=100,
        help=" ".join([
            "The number of texts you want to screen out of the sampled texts.",
            "Similar sentences are automatically removed. It is marked as TRUE",
            "in the top_k column of the resulting file.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help=" ".join([
            "The number of sentences to generate at once. It depends on the",
            "maximum VRAM size of the GPU. It is actually implemented as the",
            "argument num_return_sequences of the method generate; The number",
            "of independently computed returned sequences for each element in",
            "the batch. For a detailed explanation, please see:",
            "https://huggingface.co/docs/transformers/main_classes/model#transformers.generation_utils.GenerationMixin.generate",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help=" ".join([
            "The value used to module the next token probabilities.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help=" ".join([
            "The parameter for repetition penalty. 1.0 means no penalty.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--min_length",
        type=int,
        default=256,
        help=" ".join([
            "The minimum length of the sequence to be generated.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=256,
        help=" ".join([
            "The maximum length of the sequence to be generated.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help=" ".join([
            "The device on which the model will be loaded.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--pretrained_model_name",
        type=str,
        default="skt/kogpt2-base-v2",
        help=" ".join([
            "The pretrained model to use.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--assets",
        type=str,
        default="assets",
        help=" ".join([
            "The folder where the output result file (*.csv) will be saved.",
            "The file name is saved as '{revision}-{nowtime}-{n}.csv' by default.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "-d", "--debug", 
        action="store_true", ## default: False
        help=" ".join([
            "Specifies the debugging mode.",
            "Default=%(default)s",
        ]),
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

    ## Get extractor.
    extractor = GPT2ModelForExtraction(config)

    ## Generate.
    texts = extractor.generate()

    ## Calculate scores.
    results = extractor.score(texts) ## dataframe

    ## Deduplicate.
    results = extractor.deduplicate(results)

    ## Save.
    save_results(config, results)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
