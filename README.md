# Membership Inference Attack - Korean - Language Model

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
pip install -r requirements.txt
```

Now, just run it.

```bash
bash run.sh
```

For a more detailed description of the parameters, see [here](./assets/help.txt).

## Results

100,000개의 텍스트 중 유효하지 않은 (즉, 비어있는) 문자열이 8회 생성되어 총 생성된 텍스트는 99992개

텍스트 생성에는 약 5시간 10분 51초 (초당 5.36개 텍스트 생성)

PPL 계산에는 약 1시간 44분 42초 (초당 15.92개 텍스트 계산)
