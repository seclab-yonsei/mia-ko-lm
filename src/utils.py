import logging
import pprint

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


def print_best(metric, samples, name1: str, scores1, name2: str = None, scores2: list = None, k: int = 10):
    ## Ref
    ##  - https://github.com/ftramer/LM_Memorization/blob/main/extraction.py
    idxs = np.argsort(metric)[::-1][:k]

    ## Print it.
    for i, idx in enumerate(idxs):
        if scores2 != None:
            LOGGER.debug(f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}, text={samples[idx]}")
        else:
            LOGGER.debug(f"{i+1}: {name1}={scores1[idx]:.3f}, score={metric[idx]:.3f}, text={samples[idx]}")


def save_it(results: dict, save_path: str) -> None:
    pd.DataFrame(results).to_csv(save_path, encoding="utf-8", index=False, header=True)
    LOGGER.debug(f"results saved to {save_path}")
