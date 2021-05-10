# CLIP & FAISS Nearest Neighbours Library

Nearest-Neighbours search using [FAISS](https://ai.facebook.com/tools/faiss) on data preprocessed by [CLIP](https://github.com/openai/CLIP.git).

## Prerequisites

To run this application you'll need `wget`, Python 3.6+, and the following Python dependencies, installed from the PyPI using `conda` or `pip` (as appropriate).

- `faiss-cpu`/`faiss-gpu`
- `flask`+`flask-cors` (for deployment)
- `ftfy`
- `regex`
- `torch`, `torchvision` [(with CUDA preferably)](https://pytorch.org/get-started/locally/)
- `tqdm`

A GPU is also preferred.

### [CLIP](https://github.com/openai/CLIP)

You can install it globally.

```sh
pip install git+https://github.com/openai/CLIP.git
```

Or install it locally from submodule.

```sh
git submodule update --init
```

## Setup

Before running any scripts, review `lib/hparams.py` and change any hyperparameters as necessary.
