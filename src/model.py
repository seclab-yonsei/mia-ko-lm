import torch
import transformers

import logging
import os
import zlib

import numpy as np

from tqdm import tqdm
from typing import List


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

LOGGER = logging.getLogger(__name__)

BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"
MASK_TOKEN = "[MASK]"


class GenerativeLanguageModel():

    @staticmethod
    def get_model_and_tokenizer(pretrained_model_name_or_path: str, revision: str) -> tuple:
        ## Get version.
        ## We do not use "float-32" version because of out-of-memory (OOM).
        assert pretrained_model_name_or_path == "kakaobrain/kogpt"
        assert revision in ["KoGPT6B-ryan1.5b", "KoGPT6B-ryan1.5b-float16"]

        ## Load a tokenizer.
        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            bos_token=BOS_TOKEN,
            eos_token=EOS_TOKEN,
            unk_token=UNK_TOKEN,
            pad_token=PAD_TOKEN,
            mask_token=MASK_TOKEN,
        )
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        LOGGER.debug(f"Tokenizer loaded: {pretrained_model_name_or_path}, {revision}")

        ## Load a model.
        model = transformers.GPTJForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            ## Setting `pad_token_id` to `eos_token_id`:3 for open-end generation.
            pad_token_id=tokenizer.eos_token_id,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        ## model.config.pad_token_id = model.config.eos_token_id
        LOGGER.debug(f"Weights loaded: {pretrained_model_name_or_path}, {revision}")
        LOGGER.debug(f"  # params: {sum([p.numel() for p in model.parameters()]) / 10**9:.1f}B")

        return model, tokenizer


    @staticmethod
    def calculate_perplexity(text: str, model, tokenizer) -> float:
        ## Ref: 
        ##  - https://github.com/ftramer/LM_Memorization

        ## Get device of model.
        device = next(model.parameters()).device

        input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
        input_ids = input_ids.to(device)

        ## Generate outputs.
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)

        loss, logits = outputs[:2]
        return torch.exp(loss).cpu().detach().numpy()


    @staticmethod
    def calculate_zlib_entropy_ratio(text: str) -> int:
        return len(zlib.compress(bytes(text, "utf-8")))
        # return len(zlib.compress(bytes(text, "utf-8"))) / len(bytes(text, "utf-8"))


    @staticmethod
    def generate(model, tokenizer, n: int = 1_000, batch_size: int = 32, top_k: int = 40, seq_len = 256) -> List[np.ndarray]:
        ## Assert the model be evaluate model.

        ## We will accumulate the samples and scores and convert it to dataframe format.
        samples = []
        device = next(model.parameters()).device

        with torch.no_grad():

            num_iters = int(np.ceil(n / batch_size))
            with tqdm(total=n, desc="Generating") as pbar:
                for i in range(num_iters):
                    ## Encode the prompts, BOS.
                    prompts = [BOS_TOKEN] * batch_size
                    input_len = 1
                    inputs = tokenizer(prompts, return_tensors="pt", padding=True)

                    ## See the documents for understanding other arguments:
                    ##  - https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate
                    output_sequences = model.generate(
                        input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs["attention_mask"].to(device),
                        do_sample=True,
                        # top_k=0,
                        top_p=0.95,
                        use_cache=True,
                        min_length=input_len + seq_len,
                        max_length=input_len + seq_len,
                        # temperature=1.2,
                        repetition_penalty=2.0, ## block the non-interesting sentences
                        no_repeat_ngram_size=3, ## trigram blocking
                    ).cpu().detach().numpy()

                    ## Decode to sentence.
                    texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

                    ## The last batch must be truncated.
                    if i == num_iters - 1:
                        texts = texts[:n-i*batch_size]

                    ## Stack it.
                    samples.extend(texts)

                    ## Update progress bar.
                    pbar.update(len(texts))
        
        ## Return it.
        return samples


    @staticmethod
    def evaluate(model, tokenizer, texts) -> dict:
        ## model & tokenizer: the base model
        ## texts: generated by model1 & tokenizer1
        results = {
            "base": [], ## perplexity of base model
            "zlib": [], ## zlib entropy
            "text": [], ## original text
        }

        for text in tqdm(texts, desc="Evaluating"):
            ## Perplexity of original and mixed precision version.
            ##  - original: KoGPT6B-ryan1.5b
            ##  - mixed precision: KoGPT6B-ryan1.5b-float16
            p = GenerativeLanguageModel.calculate_perplexity(text, model, tokenizer)

            ## Zlib entropy of samples.
            z = GenerativeLanguageModel.calculate_zlib_entropy_ratio(text)

            ## Record it. 
            results["base"].append(p)
            results["zlib"].append(z)

            results["text"].append(text)

        return results


    # @staticmethod
    # def select(metric: np.ndarray, texts: List[str], k: int = 1_000) -> List[str]:
    #     ## Select top-k candidates (e.g., 1000).
    #     idxs = np.argsort(metric)[::-1][:k]
    #     return [i for i in texts[idxs]]


    # @staticmethod
    # def deduplicate(metrics: List[np.ndarray], texts: List[List[str]], tokenizer) -> List[List[str]]:
    #     pass
