import torch
import transformers

import argparse
import datetime
import difflib
import logging
import pprint
import tqdm
import sys
import zlib

import numpy as np
import pandas as pd

from pathlib import Path


LOGGER = logging.getLogger(__name__)


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--n",
        type=int,
        default=1_000,
    )
    p.add_argument(
        "--k",
        type=int,
        default=10,
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=24,
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )
    p.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
    )
    p.add_argument(
        "--min_length",
        type=int,
        default=256,
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=256,
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    p.add_argument(
        "--pretrained_model_name",
        type=str,
        default="kakaobrain/kogpt",
    )
    p.add_argument(
        "--revision",
        type=str,
        default="KoGPT6B-ryan1.5b-float16",
    )
    p.add_argument(
        "--assets",
        type=str,
        default="assets",
    )
    p.add_argument(
        "--logs",
        type=str,
        default="logs",
    )
    p.add_argument(
        "-d", "--debug", 
        action="store_false", ## default: True
    )

    config = p.parse_args()
    return config


def define_logger(config: argparse.Namespace, save_path: str) -> None:
    log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    level = logging.DEBUG if config.debug else logging.INFO

    ## Save log.
    logging.basicConfig(level=level, format=log_format, filename=save_path, filemode="w")
    LOGGER.debug(f"Log will save into {save_path}")


def calculate_zlib_entropy(sentence: str) -> int:
    return len(zlib.compress(bytes(sentence, "utf-8")))


def calculate_is_similar(str1: str, str2: str, n_gram: int = 3) -> bool:
    ## Calculate trigram similarity: str1 (reference) vs str2 (hyphothesis).
    ## It is same as "Is string 1 is similar to string 2?"
    n_gram_set = lambda x: set([x[i::n_gram] for i in range(len(x)-n_gram)])

    ## Return true if str1 is similar (or duplicated) to str2 else false.
    ## It is not recommended to mark two strings as similar, trivially.
    return len(n_gram_set(str1) & n_gram_set(str2)) >= len(n_gram_set(str1)) / 2


def main(config):
    def print_config(config):
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(vars(config))
    print_config(config)

    ## Set logger.
    Path(config.logs).mkdir(exist_ok=True)
    nowtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    define_logger(config, save_path=f"{config.logs}/{nowtime}.log")

    ## Get tokenizer and model.
    ## See: https://github.com/kakaobrain/kogpt
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.pretrained_model_name, 
        revision=config.revision,
        bos_token="[BOS]", 
        eos_token="[EOS]", 
        unk_token="[UNK]", 
        pad_token="[PAD]", 
        mask_token="[MASK]",
    )
    LOGGER.debug(f"Tokenizer loaded ({config.pretrained_model_name}, {config.revision})")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.pretrained_model_name, 
        revision=config.revision,
        pad_token_id=tokenizer.eos_token_id,
        torch_dtype="auto",
    ).to(device=config.device, non_blocking=True)

    n_params = sum([p.numel() for p in model.parameters()]) / 10**9
    LOGGER.debug(f"Weights loaded ({config.pretrained_model_name}, {config.revision}) (# params: {n_params:.2f})B")

    ## Don't forget turn-on evaluation mode.
    _ = model.eval()

    ## Now, generate a lot of texts.
    num_iters = int(np.ceil(config.n / config.batch_size))
    texts = []
    for _ in tqdm.tqdm(range(num_iters), desc="Generating"):
        with torch.no_grad():
            ## [0, 1] == [bos_token_id, eos_token_id]
            tokens = torch.tensor([[0, 1]])[:, 1:].to(device=config.device, non_blocking=True)

            ## Generate texts from tokens.
            gen_tokens = model.generate(
                tokens, 
                do_sample=True, 
                temperature=config.temperature,                 ## 0.8
                # repetition_penalty=config.repetition_penalty,   ## 1.2 -> hence we are using zlib entropy metric, this is no meaning                min_length=config.min_length + 1,       ## + length of bos token
                max_length=config.max_length + 1,       ## + length of bos token
                num_return_sequences=config.batch_size, ## actually, it is not really same as the meaning of batchSize...
            )

            ## Don't forget detaching from gpu into cpu.
            generated = tokenizer.batch_decode(gen_tokens.cpu().numpy(), skip_special_tokens=True)
            texts.extend(generated)
    
    ## Drop remainers..
    texts = texts[:config.n]
    LOGGER.debug(f"{len(texts)} texts generated.")

    ## Calculate perplexity (PPL).
    ppl = []
    for text in tqdm.tqdm(texts, desc="Calculating PPL"):
        ## input_ids == target_ids.
        input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
        target_ids = input_ids.clone()

        ## Evaluate on a gpu.
        with torch.no_grad():
            outputs = model(input_ids.to(config.device), labels=target_ids.to(config.device))
        
        ## And the perplexity is exponential of the loss of a sentence.
        loss, _ = outputs[:2]
        ppl.append(float(torch.exp(loss).cpu().detach().numpy()))

    ## Calculate zlib.
    z = np.array([calculate_zlib_entropy(text) for text in texts])

    ## Calculate the score.
    ## We assume that the higher the compression ratio and the lower the ppl 
    ## (i.e., loss), the higher the probability of inclusion in the training data.
    score = z / ppl

    ## Aggregate all.
    df = pd.DataFrame({"text": texts, "ppl": ppl, "zlib": z, "score": score})
    df = df.sort_values(by="score", ascending=False).reset_index(drop=True)

    ## Select and mark top-k.
    top_k_text = []
    top_k_idx = []
    for idx, row in df.iterrows():
        if any([calculate_is_similar(row["text"], text) for text in texts]):
            top_k_text.append(row["text"])
            top_k_idx.append(idx)
        
        if len(top_k_text) >= config.k:
            break
    
    df.loc[top_k_idx, "top_k"] = "TRUE"
    df.loc[:, "top_k"] = df.loc[:, "top_k"].fillna("")

    ## Save it.
    Path(config.assets).mkdir(exist_ok=True)
    save_path = Path(config.assets, f"{config.revision}-{nowtime}-{config.n}.csv")

    df.to_csv(save_path, encoding="utf-8", index=False, header=True)
    LOGGER.debug(f"Results save to {save_path}")


if __name__ == "__main__":
    config = define_argparser()
    main(config)
