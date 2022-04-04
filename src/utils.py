import transformers

import os
import logging

import pandas as pd

from typing import Optional, Union


LOGGER = logging.getLogger(__name__)


def get_configuration(
    pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], 
    revision: str,
    bos_token="[BOS]",
    eos_token="[EOS]",
    unk_token="[UNK]",
    pad_token="[PAD]",
    mask_token="[MASK]",
):
    ## Return pretrained model and tokenizer configuration.

    ## Kakaobrain -- KoGPT
    ##  - Github: https://github.com/kakaobrain/kogpt
    ##  - HuggingFace: https://huggingface.co/kakaobrain/kogpt
    if pretrained_model_name_or_path == "kakaobrain/kogpt" \
        and revision in ["KoGPT6B-ryan1.5b-float16", "KoGPT6B-ryan1.5b"]:
        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
        )
        model = transformers.GPTJForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )

    ## SKT AI -- KoGPT2 v2.0
    ##  - Github: https://github.com/SKT-AI/KoGPT2
    ##  - HuggingFace: https://huggingface.co/skt/kogpt2-base-v2
    elif pretrained_model_name_or_path == "skt/kogpt2-base-v2":
        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name_or_path,
            # revision=revision,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
        )
        model = transformers.GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path,
            # revision=revision,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )

    ## SKT AI -- Ko-GPT-Trinity 1.2B (v0.5)
    ##  - HuggingFace: https://huggingface.co/skt/ko-gpt-trinity-1.2B-v0.5
    elif pretrained_model_name_or_path == "skt/ko-gpt-trinity-1.2B-v0.5":
        tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name_or_path,
            # revision=revision,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
        )
        model = transformers.GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path,
            # revision=revision,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )

    else:
        raise AssertionError()

    return tokenizer, model


def save_it(elements: dict, save_path: str) -> None:
    pd.DataFrame(elements).to_csv(save_path, encoding="utf-8", index=False, header=False)
    LOGGER.debug(f"results saved to {save_path}")
