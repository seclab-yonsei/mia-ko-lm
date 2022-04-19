import torch
import transformers

import argparse
import datetime
import logging
import os
import pprint
import tqdm
import zlib

import numpy as np
import pandas as pd

from pathlib import Path


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
        default=512,
        help=" ".join([
            "The minimum length of the sequence to be generated.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=512,
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
        default="skt/ko-gpt-trinity-1.2B-v0.5",
        help=" ".join([
            "The pretrained model to use.",
            "Default=%(default)s",
        ]),
    )
    # p.add_argument(
    #     "--revision",
    #     type=str,
    #     default="KoGPT6B-ryan1.5b-float16",
    #     help=" ".join([
    #         "The version of the pretrained model to use.",
    #         "Default=%(default)s",
    #     ]),
    # )
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
        "--logs",
        type=str,
        default="logs",
        help=" ".join([
            "The folder where the log files will be saved. The default",
            "name of the log is specified in the format '{nowtime}.log'.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "-d", "--debug", 
        action="store_false", ## default: True
        help=" ".join([
            "Specifies the debugging mode.",
            "Default=%(default)s",
        ]),
    )

    config = p.parse_args()
    return config


def define_logger(config: argparse.Namespace, save_path: str) -> None:
    log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    level = logging.DEBUG if config.debug else logging.INFO

    ## Save log.
    logging.basicConfig(level=level, format=log_format, filename=save_path, filemode="w")
    LOGGER.debug(f"Log will save into {save_path}")


def calculate_zlib_entropy_ratio(sentence: str) -> int:
    return len(zlib.compress(bytes(sentence, "utf-8")))


def calculate_is_similar(str1: str, str2: str, n_gram: int = 3) -> bool:
    ## Calculate trigram similarity: str1 (reference) vs str2 (hyphothesis).
    ## It is same as "Is string 1 is similar with string 2?"
    n_gram_set = lambda x: set([" ".join([str(j) for j in x[i:i+n_gram]]) for i in range(len(x)-n_gram)])

    ## Return true if str1 is similar (or duplicated) to str2 else false.
    ## It is not recommended to mark two strings as similar, trivially.
    return len(n_gram_set(str1) & n_gram_set(str2)) >= len(n_gram_set(str1)) / 2


def main(config: argparse.Namespace) -> None:
    def print_config(config: argparse.Namespace) -> None:
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
        # revision=config.revision,
        # bos_token="[BOS]", 
        # eos_token="[EOS]", 
        # unk_token="[UNK]", 
        # pad_token="[PAD]", 
        # mask_token="[MASK]",
    )
    LOGGER.debug(f"Tokenizer loaded ({config.pretrained_model_name})")

    model = transformers.GPT2LMHeadModel.from_pretrained(
        config.pretrained_model_name, 
        # revision=config.revision,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        torch_dtype="auto",
    ).to(device=config.device, non_blocking=True)

    n_params = sum([p.numel() for p in model.parameters()]) / 10**9
    LOGGER.debug(f"Weights loaded ({config.pretrained_model_name}) (# params: {n_params:.2f})B")

    ## Don't forget turn-on evaluation mode.
    _ = model.eval()

    ## Now, generate a lot of texts.
    num_iters = int(np.ceil(config.n / config.batch_size))
    texts = []
    with tqdm.tqdm(total=config.n, desc="Generating Texts") as pbar:
        for i in range(num_iters):
            with torch.no_grad():
                ## Prompt == "<s>"
                prompt = tokenizer.bos_token
                prompt_len = 1

                tokens = tokenizer.encode(prompt, return_tensors="pt").repeat(config.batch_size, 1)
                tokens = tokens.to(device=config.device, non_blocking=True)

                ## Generate texts from tokens.
                gen_tokens = model.generate(
                    tokens, 
                    do_sample=True, 
                    temperature=config.temperature,                 ## 0.8
                    repetition_penalty=config.repetition_penalty,   ## 1.0 -> hence we are using zlib entropy metric, this is no meaning
                    min_length=config.min_length + prompt_len,
                    max_length=config.max_length + prompt_len,
                    # num_return_sequences=config.batch_size,         ## actually, it is not really same as the meaning of batchSize...
                )

                ## Don't forget detaching from gpu into cpu.
                generated = tokenizer.batch_decode(gen_tokens.cpu().numpy(), skip_special_tokens=True)
                ## Sometimes, generated texts can be empty so that calculating ppl may cause exception.
                generated = [i for i in generated if i != ""]
                if i == num_iters - 1:
                    generated = generated[:config.n - i*config.batch_size]

                texts.extend(generated)
            
            ## Update progressbar.
            pbar.update(len(generated))
            
        
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
    z = np.array([calculate_zlib_entropy_ratio(text) for text in texts])

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

    with tqdm.tqdm(desc="Deduplicating", total=config.k) as pbar:
        for idx, row in df.iterrows():
            ## We only want top-k sentences.
            if len(top_k_text) >= config.k:
                break

            ## Big O complexity: O(n(n-1)/2) where n is k.
            if all([not calculate_is_similar(tokenizer.encode(row["text"]), tokenizer.encode(text)) for text in top_k_text]):
                top_k_text.append(row["text"])  ## save for comparison
                top_k_idx.append(idx)           ## save for marking

                ## Update probress bar.
                pbar.update(1)
    
    df.loc[top_k_idx, "top_k"] = "TRUE"
    df.loc[:, "top_k"] = df.loc[:, "top_k"].fillna("")

    ## Save the total results.
    Path(config.assets).mkdir(exist_ok=True)
    save_path = Path(config.assets, f"{config.pretrained_model_name.replace(os.path.sep, '-')}-{nowtime}-{config.n}.csv")

    df.to_csv(save_path, encoding="utf-8", index=False, header=True)
    LOGGER.debug(f"Results save to {save_path}")

    ## Save top-k elements.
    save_path_ = Path(config.assets, f"{config.pretrained_model_name.replace(os.path.sep, '-')}-{nowtime}-{config.n}-partial.csv")
    df.loc[df.loc[:, "top_k"] == "TRUE", :].to_csv(save_path_, encoding="utf-8", index=False, header=True)
    LOGGER.debug(f"Results save to {save_path_} (partial)")


if __name__ == "__main__":
    config = define_argparser()
    main(config)
