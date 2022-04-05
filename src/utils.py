import logging

import pandas as pd


LOGGER = logging.getLogger(__name__)


def save_it(results: dict, save_path: str) -> None:
    pd.DataFrame(results).to_csv(save_path, encoding="utf-8", index=False, header=True)
    LOGGER.debug(f"results saved to {save_path}")
