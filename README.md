# Multimodal CLIP Applications

## Prequisites

- `python` 3.6+
- `wget`

### Python Dependencies

Install the following dependencies from the PyPI using `conda` or `pip`.

- `faiss-cpu`/`faiss-gpu`
- `flask`
- `ftfy`
- `regex`
- `torch` (preferably with CUDA)
- `torchvision`
- `tqdm`

After installing the base PyPI dependencies, install [CLIP](https://github.com/openai/CLIP.)

```sh
pip install git+https://github.com/openai/CLIP.git
```

## Usage

Before running any programs, review `lib/hparams.py` and change any hyperparameters as necessary.

### Data

- Run `scripts/init.sh` to prepare the workspace.
- Run `scripts/tiny-imagenet-200.sh` to use the default dataset (`tiny-imagenet-200`) to encode images.

### Indexing

- Run `build_image_index.py` to build an image index for `n_classes` of the desired dataset.
  - By default this is the first ten classes from `tiny-imagenet-200`.
- The indexes will be stored in `indexes` and have a filename indicating dataset, class, and number of features (which may vary given the CLIP model and FAISS compression method used).

### Deployment

- Run `sh scripts/api.sh` from the base of the repository to start the server (default on port `5020`).
- You can use the the prototypes in `frontend` to make GET/POST requests.
