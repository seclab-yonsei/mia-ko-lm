import torch
import transformers

import os
import logging

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional, Union


LOGGER = logging.getLogger(__name__)


class KoGPTInference():
    def __init__(
        self,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        revision: str = "KoGPT6B-ryan1.5b-float16",
        device: str = "cuda:0",
        total: int = 10_000, ## 1_000,
        batch_size: int = 16,
    ):
        ## Load a tokenizer.
        self.tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            bos_token="[BOS]",
            eos_token="[EOS]",
            unk_token="[UNK]",
            pad_token="[PAD]",
            mask_token="[MASK]",
        )
        LOGGER.debug("Tokenizer loaded.")

        ## Load a model to GPU(s).
        model = transformers.GPTJForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        LOGGER.debug("Weights loaded.")
        LOGGER.debug(f"# params: {sum([p.numel() for p in model.parameters()]) / 10**9:.1f}B")

        ## Parallel inference.
        # n_gpus = torch.cuda.device_count()
        # if n_gpus > 1:
        #     model = torch.nn.DataParallel(model)
        #     LOGGER.debug(f"{n_gpus} gpus available; using torch.nn.DataParallel.")

        _ = model.eval()
        self.model = model.to(device=device)
        self.device = device

        ## Hyperparameters.
        self.total = total
        self.batch_size = batch_size
        self.iters = int(np.ceil(self.total / self.batch_size))

        self.save_path = Path("assets", f"{revision}-{self.total}.csv")
        Path.mkdir(self.save_path.parent, exist_ok=True)


    def generate(self, prompt: str = None, max_length: int = 256, min_length: int = 256) -> str:
        if prompt == None:
            prompt = self.tokenizer.bos_token

        tokens = self.tokenizer.encode(prompt, return_tensors="pt")
        LOGGER.debug(f"prompt: {prompt} (len={len(tokens)}), tokens: {tokens.shape}")

        with torch.no_grad():
            outputs = []

            for i in range(self.iters):
                ## Clone it.
                batch_tokens = tokens.clone().detach().repeat(self.batch_size, 1)
                batch_tokens = batch_tokens.to(device=self.device, non_blocking=True)

                ## Generate tokens.
                gen_tokens = self.model.generate(
                    batch_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_k=None,
                    top_p=0.95,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                    min_length=min_length + 1,
                    max_length=max_length + 1,
                    no_repeat_ngram_size=3,
                ).cpu().detach().numpy()[:, 1:]

                ## Decode and save it.
                gen_text = self.tokenizer.batch_decode(gen_tokens) ## exclude bos token
                outputs.append(gen_text)

                if not (i % 10):
                    LOGGER.debug(f"loop: {i+1}/{self.iters}, accumulated texts: {(i+1) * self.batch_size}")
        
        # LOGGER.debug(f"Generated: {gen_text}")
        outputs = np.concatenate(outputs, axis=0)[:self.total]
        self._save_it(outputs)
        LOGGER.debug(f"results saved to {self.save_path}")

    
    def _save_it(self, outputs: np.ndarray):
        pd.DataFrame(outputs).to_csv(self.save_path, encoding="utf-8", index=False, header=False)
