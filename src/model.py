import torch
import transformers

import argparse
import logging
import tqdm

import numpy as np
import pandas as pd

from operator import itemgetter
from typing import List, Dict


from src.metrics import calculate_zlib_entropy_ratio, calculate_is_similar


BOS_TOKEN="</s>"
EOS_TOKEN="</s>"
UNK_TOKEN="<unk>"
PAD_TOKEN="<pad>"
MASK_TOKEN="<mask>"

LOGGER = logging.getLogger(__name__)


class GPT2ModelForExtraction():

    def __init__(
        self,
        config: argparse.Namespace,
    ):
        self.config = config

        ## Get tokenizer and model.
        self.tokenizer, self.model = self._get_tokenizer(
            pretrained_model_name_or_path=config.pretrained_model_name,
            device=config.device,
        )


    def _get_tokenizer(self, pretrained_model_name_or_path: str, device: str):
        ## Get tokenizer.
        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name_or_path, 
            bos_token=BOS_TOKEN, 
            eos_token=EOS_TOKEN, 
            unk_token=UNK_TOKEN,
            pad_token=PAD_TOKEN,
            mask_token=MASK_TOKEN,
        )
        LOGGER.debug(f"Tokenizer loaded: {pretrained_model_name_or_path}")

        ## Get model.
        model = transformers.GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path, 
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            torch_dtype="auto",
        ).to(device=device, non_blocking=True)

        n_params = sum([p.numel() for p in model.parameters()]) / 10**6 ## M
        LOGGER.debug(f"Weights loaded: {pretrained_model_name_or_path} (# params: {n_params:.2f}M)")

        return tokenizer, model


    def generate(self) -> List[str]:
        ## Don't forget turn-on evaluation mode.
        _ = self.model.eval()

        ## Now, generate a lot of texts.
        num_iters = int(np.ceil(self.config.n / self.config.batch_size))
        texts = []
        with tqdm.tqdm(total=self.config.n, desc="Generating Texts") as pbar:
            for i in range(num_iters):
                with torch.no_grad():
                    ## Prompt == "<s>"
                    prompt = self.tokenizer.bos_token
                    prompt_len = 1

                    tokens = self.tokenizer.encode(prompt, return_tensors="pt").repeat(self.config.batch_size, 1)
                    tokens = tokens.to(device=self.config.device, non_blocking=True)

                    ## Generate texts from tokens.
                    gen_tokens = self.model.generate(
                        tokens, 
                        do_sample=True, 
                        temperature=self.config.temperature,                 ## 0.8
                        repetition_penalty=self.config.repetition_penalty,   ## 1.0 -> hence we are using zlib entropy metric, this is no meaning
                        min_length=self.config.min_length + prompt_len,
                        max_length=self.config.max_length + prompt_len,
                        # num_return_sequences=self.config.batch_size,         ## actually, it is not really same as the meaning of batchSize...
                    )

                    ## Don't forget detaching from gpu into cpu.
                    generated = self.tokenizer.batch_decode(gen_tokens.cpu().numpy(), skip_special_tokens=True)
                    ## Sometimes, generated texts can be empty so that calculating ppl may cause exception.
                    generated = [i for i in generated if i != ""]
                    if i == num_iters - 1:
                        generated = generated[:self.config.n - i*self.config.batch_size]

                    texts.extend(generated)
                
                ## Update progressbar.
                pbar.update(len(generated))
                
        ## Drop remainers..
        texts = texts[:self.config.n]
        LOGGER.debug(f"{len(texts)} texts generated, {self.config.n - len(texts)} texts failed to generate")

        return texts

    
    def score(self, texts: List[str]) -> pd.DataFrame:
        ## Calculate perplexity (PPL).
        ppl = []

        for text in tqdm.tqdm(texts, desc="Calculating PPL"):
            ## input_ids == target_ids.
            input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)
            target_ids = input_ids.clone()

            ## Evaluate on a gpu.
            with torch.no_grad():
                outputs = self.model(
                    input_ids.to(self.config.device), 
                    labels=target_ids.to(self.config.device),
                )
            
            ## And the perplexity is exponential of the loss of a sentence.
            loss, _ = outputs[:2]
            ppl.append(float(torch.exp(loss).cpu().detach().numpy()))

        ## Calculate zlib.
        z = [calculate_zlib_entropy_ratio(text) for text in texts]

        ## Calculate the score.
        ## We assume that the higher the compression ratio and the lower the ppl 
        ## (i.e., loss), the higher the probability of inclusion in the training data.
        z = np.array(z)
        ppl = np.array(ppl)
        score = z / ppl

        ## Sort by scores.
        df = pd.DataFrame({"text": texts, "score": score, "ppl": ppl, "zlib": z})
        df = df.sort_values(by="score", ascending=False).reset_index(drop=True)

        return df

    
    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        ## Select and mark top-k.
        top_k_text = []
        top_k_idx = []

        with tqdm.tqdm(desc="Deduplicating", total=self.config.k) as pbar:
            for idx, row in df.iterrows():
                ## We only want top-k sentences.
                if len(top_k_text) >= self.config.k:
                    break

                ## Big O complexity: O(n(n-1)/2) where n is k.
                if all([not calculate_is_similar(self.tokenizer.encode(row["text"]), self.tokenizer.encode(text)) for text in top_k_text]):
                    top_k_text.append(row["text"])  ## save for comparison
                    top_k_idx.append(idx)           ## save for marking

                    ## Update probress bar.
                    pbar.update(1)
        
        df.loc[top_k_idx, "top_k"] = True
        df.loc[:, "top_k"] = df.loc[:, "top_k"].fillna("")

        return df