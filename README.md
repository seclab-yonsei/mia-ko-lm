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
