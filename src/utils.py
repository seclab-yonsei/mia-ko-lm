import argparse

import logging
import os

import pandas as pd

from pathlib import Path


LOGGER = logging.getLogger(__name__)


def save_results(config: argparse.Namespace, df: pd.DataFrame, nowtime: str):
    ## Save the total results.
    Path(config.assets).mkdir(exist_ok=True)
    save_path = Path(config.assets, f"{config.pretrained_model_name.replace(os.path.sep, '-')}-{nowtime}-{config.n}.csv")

    df.to_csv(save_path, encoding="utf-8", index=False, header=True)
    LOGGER.debug(f"Results save to {save_path}")

    ## Save top-k elements.
    save_path_ = Path(config.assets, f"{config.pretrained_model_name.replace(os.path.sep, '-')}-{nowtime}-{config.n}-partial.csv")
    df.loc[df.loc[:, "top_k"] == "TRUE", :].to_csv(save_path_, encoding="utf-8", index=False, header=True)
    LOGGER.debug(f"Results save to {save_path_} (partial)")
