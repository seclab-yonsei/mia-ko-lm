# Membership Inference - Korean - Language Model

Performing membership inference attack (MIA) against Korean language models (LMs).

## Environments

A GPU with at least 16 GB of VRAM is required.

## Usage

First, create a virtual environment based on `Python 3.8`.

```bash
conda create -n py38 python=3.8
conda activate py38
```

After that, install the necessary libraries.

```bash
## It is also possible to use the 'conda' command.
(py38) pip install -r requirements.txt
```

Now, just run it.

```bash
(py38) python extract.py \
    --n 100_000 \
    --k 100 \
    --batch_size 24 \
    --temperature 1.0 \
    --repetition_penalty 1.0 \
    --min_length 256 \
    --max_length 256 \
    --num_return_sequences 1 \
    --device "cuda:0" \
    --pretrained_model_name "kakaobrain/kogpt" \
    --revision "KoGPT6B-ryan1.5b-float16" \
    --assets assets \
    --debug
```

For a more detailed description of the parameters, see [here](./assets/help.txt).

## Citation

Please cite below if you make use of the code.

```
@misc{oh2022on,
  title={On Membership Inference Attacks to Generative Language Models across Language Domains},
  author={Myung Gyo Oh and Leo Hyun Park and Jaeuk Kim and Jaewoo Park and Taekyoung Kwon},
  year={2022},
  howpublished={\url{https://github.com/cawandmilk/mia-ko-lm}},
}
```
