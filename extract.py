import argparse
import datetime
import logging
import pprint
import sys

from src.model import GenerativeLanguageModel
from src.utils import save_it


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--n",
        type=int,
        default=1_000,
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    p.add_argument(
        "-d", "--debug", 
        action="store_false", ## default: True
    )

    config = p.parse_args()
    return config


def define_logger(config: argparse.Namespace) -> None:
    log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    level = logging.DEBUG if config.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format, stream=sys.stderr)


def main(config):
    def print_config(config):
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(vars(config))
    print_config(config)

    ## Set logger.
    define_logger(config)

    ## Get model.
    model, tokenizer = GenerativeLanguageModel.get_model_and_tokenizer("kakaobrain/kogpt", "KoGPT6B-ryan1.5b-float16")

    ## Since both models are quite large, we assign them to different gpus.
    ## Each input is loaded according to the model, so don't worry too much about it.
    model.to("cuda:0")
    
    ## Don't forget turn-on evaluation mode.
    model.eval()

    ## Generate candidates.
    texts = GenerativeLanguageModel.generate(model, tokenizer, n=config.n, batch_size=config.batch_size)

    ## Evaluate.
    # results = GenerativeLanguageModel.evaluate(model1, model2, tokenizer1, tokenizer2, texts)
    results = GenerativeLanguageModel.evaluate(model, tokenizer, texts)

    ## Save it.
    nowtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_it(results, save_path=f"./assets/{'KoGPT6B-ryan1.5b-float16'}-{nowtime}-{config.n}.csv")


if __name__ == "__main__":
    config = define_argparser()
    main(config)
