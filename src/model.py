import torch
import transformers

import argparse
import logging
import tqdm

import numpy as np
import pandas as pd

from typing import List


from src.metrics import calculate_zlib_entropy, calculate_is_similar


LOGGER = logging.getLogger(__name__)


class GPT2ModelForExtraction:
    def __init__(
        self,
        config: argparse.Namespace,
    ):
        self.config = config

        ## Get tokenizer and model.
        self.tokenizer, self.model = self._get_tokenizer_and_model()

    def _get_tokenizer_and_model(self):
        assert self.config.pretrained_model_name in [
            "kakaobrain/kogpt",
            "kykim/gpt3-kor-small_based_on_gpt2",
            "skt/kogpt2-base-v2",
        ]

        def get_tokenizer_and_model_of_kakaobrain():
            ## See: https://huggingface.co/kakaobrain/kogpt
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.config.pretrained_model_name,
                revision=self.config.revision,
                bos_token="[BOS]",
                eos_token="[EOS]",
                unk_token="[UNK]",
                pad_token="[PAD]",
                mask_token="[MASK]",
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.config.pretrained_model_name,
                revision=self.config.revision,
                pad_token_id=tokenizer.eos_token_id,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
            ).to(device=self.config.device, non_blocking=True)

            return tokenizer, model

        def get_tokenizer_and_model_of_kykim():
            ## See: https://github.com/kiyoungkim1/LMkor/blob/main/examples/gpt3_generation.py
            transformers.logging.set_verbosity_error()

            tokenizer = transformers.BertTokenizerFast.from_pretrained(
                self.config.pretrained_model_name,
            )
            model = transformers.GPT2LMHeadModel.from_pretrained(
                self.config.pretrained_model_name,
                pad_token_id=tokenizer.eos_token_id,
            ).to(device=self.config.device, non_blocking=True)

            return tokenizer, model

        def get_tokenizer_and_model_of_skt():
            ## See: https://github.com/SKT-AI/KoGPT2
            tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
                self.config.pretrained_model_name,
                bos_token="</s>",
                eos_token="</s>",
                unk_token="<unk>",
                pad_token="<pad>",
                mask_token="<mask>",
            )
            model = transformers.GPT2LMHeadModel.from_pretrained(
                self.config.pretrained_model_name,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            ).to(device=self.config.device, non_blocking=True)

            return tokenizer, model

        ## Get items.
        tokenizer, model = {
            "kakaobrain/kogpt": get_tokenizer_and_model_of_kakaobrain,
            "kykim/gpt3-kor-small_based_on_gpt2": get_tokenizer_and_model_of_kykim,
            "skt/kogpt2-base-v2": get_tokenizer_and_model_of_skt,
        }[self.config.pretrained_model_name]()

        LOGGER.debug(f"Tokenizer loaded: {self.config.pretrained_model_name}")

        n_params = sum([p.numel() for p in model.parameters()]) / 10**9  ## billion
        LOGGER.debug(
            f"Weights loaded: {self.config.pretrained_model_name} (# params: {n_params:.2f}B)"
        )

        return tokenizer, model

    def generate(self) -> List[str]:
        ## Don't forget turn-on evaluation mode.
        _ = self.model.eval()

        ## Now, generate a lot of texts.
        num_iters = int(
            np.ceil(
                self.config.n
                / (self.config.batch_size * self.config.num_return_sequences)
            )
        )
        texts = []
        with tqdm.tqdm(total=self.config.n, desc="Generating Texts") as pbar:
            for i in range(num_iters):
                with torch.no_grad():
                    # KyKim's tokenizer should have no prompt.
                    # prompt = (
                    #     ""
                    #     if self.config.pretrained_model_name
                    #     == "kykim/gpt3-kor-small_based_on_gpt2"
                    #     else self.tokenizer.bos_token
                    # )
                    prompt_len = 1

                    # tokens = self.tokenizer.encode(prompt, return_tensors="pt").repeat(
                    #     self.config.batch_size, 1
                    # )
                    # if (
                    #     self.config.pretrained_model_name
                    #     == "kykim/gpt3-kor-small_based_on_gpt2"
                    # ):
                    #     tokens = tokens[:, 1:]
                    tokens = torch.tensor([[0, 1]])[:, 1:].repeat(
                        self.config.batch_size, 1
                    )
                    tokens = tokens.to(device=self.config.device, non_blocking=True)

                    ## Generate texts from tokens.
                    gen_tokens = self.model.generate(
                        tokens,
                        do_sample=True,
                        temperature=self.config.temperature,  ## 0.8
                        repetition_penalty=self.config.repetition_penalty,  ## 1.0 -> hence we are using zlib entropy metric, this is no meaning
                        min_length=self.config.min_length + prompt_len,
                        max_length=self.config.max_length + prompt_len,
                        num_return_sequences=self.config.num_return_sequences,  ## actually, it is not really same as the meaning of batchSize...
                    )

                    ## Don't forget detaching from gpu into cpu.
                    generated = self.tokenizer.batch_decode(
                        gen_tokens.cpu().numpy(), skip_special_tokens=True
                    )
                    ## Sometimes, generated texts can be empty so that calculating ppl may cause exception.
                    generated = [i for i in generated if i != ""]
                    if i == num_iters - 1:
                        generated = generated[
                            : self.config.n
                            - i
                            * self.config.batch_size
                            * self.config.num_return_sequences
                        ]

                    texts.extend(generated)

                ## Update progressbar.
                pbar.update(len(generated))

        ## Drop remainers..
        texts = texts[: self.config.n]
        LOGGER.debug(
            f"{len(texts)} texts generated, {self.config.n - len(texts)} texts failed to generate"
        )

        return texts

    # def _drop_outlier(self, values):
    #     q1, q3 = np.percentile(values, [25, 75])
    #     iqr = q3 - q1
    #     lower_bound = q1 - (iqr * 1.5)
    #     upper_bound = q3 + (iqr * 1.5)
    #     return np.where((values > upper_bound) | (values < lower_bound))

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
        z = [calculate_zlib_entropy(text) for text in texts]

        ## Calculate the score.
        ## We assume that the higher the compression ratio and the lower the ppl
        ## (i.e., loss), the higher the probability of inclusion in the training data.
        z = np.array(z)
        ppl = np.array(ppl)
        score = z / ppl

        ## Sort by scores.
        df = pd.DataFrame({"text": texts, "score": score, "ppl": ppl, "zlib": z})
        df = df.sort_values(by="score", ascending=False).reset_index(drop=True)

        # ## Drop nan values.
        # num_nan = df.loc[df.loc[:, "ppl"].isna()].shape[0]
        # LOGGER.debug(f"{num_nan} nan values dropped in column 'PPL'")

        # ## Drop outlier.
        # outlier = self._drop_outlier(df.loc[:, "ppl"])[0]
        # df = df.drop(outlier).reset_index(drop=True)
        # LOGGER.debug(f"{len(outlier)} outliers dropped in column 'PPL'")

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
                if all(
                    [
                        not calculate_is_similar(
                            self.tokenizer.encode(row["text"]),
                            self.tokenizer.encode(text),
                        )
                        for text in top_k_text
                    ]
                ):
                    top_k_text.append(row["text"])  ## save for comparison
                    top_k_idx.append(idx)  ## save for marking

                    ## Update probress bar.
                    pbar.update(1)

        df.loc[top_k_idx, "top_k"] = True
        df.loc[:, "top_k"] = df.loc[:, "top_k"].fillna("")

        return df
