import torch
import transformers

import argparse
import datetime
import logging
import os
import pprint
import re
import tqdm
import zlib

import numpy as np
import pandas as pd

from pathlib import Path
from typing import List


LOGGER = logging.getLogger(__name__)


def define_argparser() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--n",
        type=int,
        default=10_000,
        help=" ".join(
            [
                "The number of texts you want to sample.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--k",
        type=int,
        default=100,
        help=" ".join(
            [
                "The number of texts you want to screen out of the sampled texts.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help=" ".join(
            [
                "Threshold value for similarity.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help=" ".join(
            [
                "The number of sentences to generate at once. It depends on the",
                "maximum VRAM size of the GPU.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=" ".join(
            [
                "The value used to module the next token probabilities.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help=" ".join(
            [
                "The parameter for repetition penalty. 1.0 means no penalty.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--min_length",
        type=int,
        default=256,
        help=" ".join(
            [
                "The minimum length of the sequence to be generated.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=256,
        help=" ".join(
            [
                "The maximum length of the sequence to be generated.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help=" ".join(
            [
                "Number of generating output statements from LM using a single input."
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help=" ".join(
            [
                "If set to float < 1, only the most probable tokens",
                "with probabilities that add up to top_p or higher are",
                "kept for generation.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=40,
        help=" ".join(
            [
                "The number of highest probability vocabulary tokens",
                "to keep for top-k-filtering.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help=" ".join(
            [
                "The device on which the model will be loaded.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--pretrained_model_name",
        type=str,
        default="kakaobrain/kogpt",
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
        "--assets",
        type=str,
        default="assets",
        help=" ".join(
            [
                "The folder where the output result file (*.csv) will be saved.",
                "The file name is saved as '{revision}-{nowtime}-{n}.csv' by default.",
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


def get_tokenizer_and_model(config: argparse.Namespace) -> tuple:
    ## See: https://huggingface.co/kakaobrain/kogpt
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.pretrained_model_name,
        revision=config.revision,
        # bos_token="[BOS]",
        # eos_token="[EOS]",
        # unk_token="[UNK]",
        # pad_token="[PAD]",
        # mask_token="[MASK]",
    )
    LOGGER.debug(f"Tokenizer loaded: {config.pretrained_model_name}")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.pretrained_model_name,
        revision=config.revision,
        pad_token_id=tokenizer.eos_token_id,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    n_params = sum([p.numel() for p in model.parameters()]) / 10**9  ## billion
    LOGGER.debug(
        f"Weights loaded: {config.pretrained_model_name} (# params: {n_params:.2f}B)"
    )

    return tokenizer, model


def generate(config: argparse.Namespace, tokenizer, model, prompt: str) -> List[str]:

    with torch.no_grad():
        ## Encode it.
        tokens = tokenizer.encode(prompt, return_tensors="pt").repeat(
            config.batch_size, 1
        )
        tokens = tokens.to(device=config.device, non_blocking=True)

        prompt_len = tokens.size(1)
        assert prompt_len == 1

        ## Generate texts from tokens.
        gen_tokens = model.generate(
            tokens,
            do_sample=True,
            # temperature=config.temperature,  ## 0.8
            repetition_penalty=config.repetition_penalty,  ## 1.0 -> hence we are using zlib entropy metric, this is no meaning
            min_length=config.min_length + prompt_len,
            max_length=config.max_length + prompt_len,
            num_return_sequences=config.num_return_sequences,  ## actually, it is not really same as the meaning of batchSize...
            top_p=config.top_p,
            top_k=config.top_k,
        )

        ## Don't forget detaching from gpu into cpu.
        generated = tokenizer.batch_decode(
            gen_tokens.cpu().numpy(), skip_special_tokens=True
        )

    return generated


def calcualte_perplexity(
    config: argparse.Namespace, tokenizer, model, text: str
) -> float:
    ## input_ids == target_ids.
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    target_ids = input_ids.clone()

    ## Evaluate on a gpu.
    with torch.no_grad():
        outputs = model(
            input_ids.to(config.device),
            labels=target_ids.to(config.device),
        )

    ## And the perplexity is exponential of the loss of a sentence.
    loss, _ = outputs[:2]
    ppl = float(torch.exp(loss).cpu().detach().numpy())

    return ppl


def calcualte_window_perplexity(
    config: argparse.Namespace,
    tokenizer,
    model,
    text: str,
    window_size: int = 50,
    stride: int = 16,
) -> float:
    ## input_ids == target_ids.
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    target_ids = input_ids.clone()

    ppls = []
    for idx in range(0, input_ids.size(1) - window_size, stride):
        ## Evaluate on a gpu.
        with torch.no_grad():
            outputs = model(
                input_ids[:, idx : idx + window_size].to(config.device),
                labels=target_ids[:, idx : idx + window_size].to(config.device),
            )

        ## And the perplexity is exponential of the loss of a sentence.
        loss, _ = outputs[:2]
        ppl = float(torch.exp(loss).cpu().detach().numpy())
        ppls.append(ppl)

    return min(ppls)


def calculate_zlib_entropy(sentence: str, encoding: str = "utf-8") -> int:
    return len(zlib.compress(bytes(sentence, encoding)))


def calculate_similarity(
    tokenizer,
    str1: str,
    str2: str,
    n_gram: int = 3,
    is_word_level: bool = True,
) -> bool:
    def _word_level_tokenization(sentence) -> list:
        return re.split(r"[\s]+|[.,!?;]", sentence)

    def _bpe_token_level_tokenization(sentence) -> list:
        return tokenizer.encode(sentence)

    if is_word_level:
        str1 = _word_level_tokenization(str1)
        str2 = _word_level_tokenization(str2)
    else:
        str1 = _bpe_token_level_tokenization(str1)
        str2 = _bpe_token_level_tokenization(str2)

    ## Calculate trigram similarity: str1 (reference) vs str2 (hyphothesis).
    ## It is same as "Is string 1 is similar with string 2?"
    n_gram_set = lambda x: [
        set(j for j in x[i : i + n_gram]) for i in range(len(x) - n_gram)
    ]

    s1 = n_gram_set(str1)
    s2 = n_gram_set(str2)

    ## Return true if str1 is similar (or duplicated) to str2 else false.
    ## It is not recommended to mark two strings as similar, trivially.
    return len([i for i in s1 if i in s2]) / len(s1)


def save_results(config: argparse.Namespace, df: pd.DataFrame) -> None:
    nowtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    ## Save the total results.
    fname = (
        "-".join(
            [
                config.pretrained_model_name.replace("./", "").replace(
                    os.path.sep, "-"
                ),
                config.revision,
                nowtime,
                f"bs{config.batch_size}",
                f"rs{config.num_return_sequences}",
                f"n{config.n}",
                f"k{config.k}",
            ]
        )
        + ".csv"
    )
    Path(config.assets).mkdir(exist_ok=True)
    save_path = Path(config.assets, fname)

    df.to_csv(save_path, encoding="utf-8", index=False, header=True)
    LOGGER.debug(f"Results save to {save_path}")


def main(config: argparse.Namespace) -> None:
    def print_config(config: argparse.Namespace) -> None:
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(vars(config))

    print_config(config)

    ## Set logger.
    define_logger(config)

    ## Get model and tokenizer.
    tokenizer, model = get_tokenizer_and_model(config)
    model.to(device=config.device, non_blocking=True)

    ## Generate.
    ## Don't forget turn-on evaluation mode.
    _ = model.eval()

    ## Now, generate a lot of texts.
    num_iters = int(
        np.ceil(config.n / (config.batch_size * config.num_return_sequences))
    )
    texts = []
    with tqdm.tqdm(total=config.n, desc="Generating Texts") as pbar:
        while True:
            ## Generate sentences with one batch.
            prompt = tokenizer.eos_token
            generated = generate(config, tokenizer, model, prompt=prompt)

            ## Sometimes, generated texts can be empty so that calculating ppl may cause exception.
            generated = [i for i in generated if i.strip() != ""]

            ## Drop last if need.
            # if i == num_iters - 1:
            #     generated = generated[
            #         : config.n - i * config.batch_size * config.num_return_sequences
            #     ]

            if len(texts) + len(generated) > config.n:
                generated = generated[: config.n - len(texts)]

            ## Gather it.
            texts.extend(generated)

            ## Update progressbar.
            pbar.update(len(generated))

            if len(texts) >= config.n:
                break

    ## Drop remainers..
    texts = texts[: config.n]
    LOGGER.debug(
        f"{len(texts)} texts generated, {config.n - len(texts)} texts failed to generate"
    )

    ## Score it.
    p = np.array(
        [
            calcualte_perplexity(config, tokenizer, model, text)
            for text in tqdm.tqdm(texts, desc="Calculating PPL")
        ]
    )
    z = np.array(
        [
            calculate_zlib_entropy(text)
            for text in tqdm.tqdm(texts, desc="Calculating zlib Entropy")
        ]
    )
    p_lower = np.array(
        [
            calcualte_perplexity(config, tokenizer, model, text.lower())
            for text in tqdm.tqdm(texts, desc="Calculating PPL of Lowercase")
        ]
    )
    w = np.array(
        [
            calcualte_window_perplexity(config, tokenizer, model, text)
            for text in tqdm.tqdm(texts, desc="Calculating PPL by Sliding Window")
        ]
    )

    ## Concat all.
    df = pd.DataFrame(
        {
            "text": texts,
            "ppl": p,
            "zlib_entropy": z,
            "ppl_lower": p_lower,
            "sliding_window": w,
        }
    )

    ## Scoring.
    ##  - Score 1: only PPL (lower best)
    ##  - Score 2: zlib entropy / perplexity (higher best)
    ##  - Score 3: lower PPL / naive PPL (higher best)
    ##  - Score 4: Sliding window PPL (lower best)
    df.loc[:, "score1"] = -np.log(df.loc[:, "ppl"])
    df.loc[:, "score2"] = df.loc[:, "zlib_entropy"] / np.log(df.loc[:, "ppl"])
    df.loc[:, "score3"] = np.log(df.loc[:, "ppl_lower"]) / np.log(df.loc[:, "ppl"])
    df.loc[:, "score4"] = -np.log(df.loc[:, "sliding_window"])

    # ## De-duplicating.
    # for column in [f"score{i}" for i in range(1, 5, 1)]:
    #     ## First, we sort values.
    #     df = df.sort_values(by=column, ascending=False).reset_index(drop=True)

    #     ## Word-level similarity.
    #     top_k_text = []
    #     top_k_idx = []

    #     with tqdm.tqdm(desc=f"De-duplicating (by={column})", total=config.k) as pbar:
    #         for idx, row in df.iterrows():
    #             ## We only want top-k sentences.
    #             if len(top_k_text) >= config.k:
    #                 break

    #             ## Big O complexity: O(n(n-1)/2) where n is k.
    #             if all(
    #                 [
    #                     calculate_similarity(
    #                         tokenizer, row["text"], text, is_word_level=True
    #                     )
    #                     < config.alpha
    #                     for text in top_k_text
    #                 ]
    #             ):
    #                 top_k_text.append(row["text"])  ## save for comparison
    #                 top_k_idx.append(idx)  ## save for marking

    #                 ## Update probress bar.
    #                 pbar.update(1)

    #     df.loc[top_k_idx, f"{column}_top_k"] = np.arange(config.k)

    #     ## BPE token-level similarity.
    #     top_k_text = []
    #     top_k_idx = []

    #     with tqdm.tqdm(desc=f"De-duplicating (by={column})", total=config.k) as pbar:
    #         for idx, row in df.iterrows():
    #             ## We only want top-k sentences.
    #             if len(top_k_text) >= config.k:
    #                 break

    #             ## Big O complexity: O(n(n-1)/2) where n is k.
    #             if all(
    #                 [
    #                     calculate_similarity(
    #                         tokenizer, row["text"], text, is_word_level=False
    #                     )
    #                     < config.alpha
    #                     for text in top_k_text
    #                 ]
    #             ):
    #                 top_k_text.append(row["text"])  ## save for comparison
    #                 top_k_idx.append(idx)  ## save for marking

    #                 ## Update probress bar.
    #                 pbar.update(1)

    #     df.loc[top_k_idx, f"{column}_top_k_bpe"] = np.arange(config.k)

    ## Save.
    save_results(config, df)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
