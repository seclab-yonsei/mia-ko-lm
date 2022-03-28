import argparse
import logging
import pprint
import sys

from src.inference import KoGPTInference


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        type=str,
        default="kakaobrain/kogpt",
        # help="",
    )
    p.add_argument(
        "--revision",
        type=str,
        default="KoGPT6B-ryan1.5b-float16",
        choices=['KoGPT6B-ryan1.5b', 'KoGPT6B-ryan1.5b-float16'],
    )
    p.add_argument(
        "-d", "--debug", 
        action="store_false", ## default: True
    )

    config = p.parse_args()
    return config


def define_logger(config: argparse.Namespace):
    log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
    level = logging.DEBUG if config.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format, stream=sys.stderr)


def main(config):
    def print_config(config):
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(vars(config))
    print_config(config)

    ## Set logger.
    define_logger(config)

    ## Get ready.
    model = KoGPTInference(
        pretrained_model_name_or_path=config.model,
        revision=config.revision,
    )

    ## Run it.
    model.generate()


if __name__ == "__main__":
    config = define_argparser()
    main(config)
