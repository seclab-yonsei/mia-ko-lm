import argparse
import pprint
import tqdm

import numpy as np
import pandas as pd

from src.diff import get_difference


def define_argparser() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--fpath",
        type=str,
        required=True,
        help=" ".join([
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help=" ".join([
            "Default=%(default)s",
        ]),
    )

    config = p.parse_args()
    return config


def get_dataframe(config: argparse.Namespace) -> pd.DataFrame:
    return pd.read_csv(config.fpath, encoding=config.encoding)


def save_dataframe(config: argparse.Namespace, df: pd.DataFrame) -> None:
    save_path = config.fpath.replace(".csv", "-diff.csv")
    # df.to_csv(save_path, encoding=config.encoding)
    df.to_csv(save_path, encoding="utf-8", index=False)


def main(config: argparse.Namespace) -> None:
    def print_config(config: argparse.Namespace) -> None:
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(vars(config))
    print_config(config)

    ## Get dataframe.
    df = get_dataframe(config)

    ## Get contents.
    for i in tqdm.tqdm(range(df.shape[0]), total=df.shape[0], desc="Crawling"):
        ## If crawled url is not exists...
        if df.loc[i, "url"] in [np.nan, "-"]:
            continue
    
        ## Perform crawling.
        reference = get_request_and_content(df.loc[i, "url"])
        hypothesis = df.loc[i, "text"]

        ## If we failed to crawling...
        if reference == None:
            continue

        ## Get difference.
        ratio, strict_reference = get_difference(reference, hypothesis)

        ## Save it.
        df.loc[i, "ratio"] = ratio
        df.loc[i, "strict_reference"] = strict_reference

    ## Save the results.
    save_dataframe(config, df)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
