# Multimodal CLIP Applications

Prototype for multimodal image/text applications using [OpenAI's CLIP preprocessing architecture](https://openai.com/blog/clip/). This application uses [FAISS](https://ai.facebook.com/tools/faiss)'s Inner Product Nearest Neighbours (NNs) approximations to search (text-to-image) or classify (image-to-text) images.

---

## Prerequisites

To run this application you'll need `wget`, Python 3.6+, and the following Python dependencies, installed from the PyPI using `conda` or `pip` (as appropriate).

- `faiss-cpu`/`faiss-gpu`
- `flask`
- `ftfy`
- `regex`
- `torch` (preferably with CUDA)
- `torchvision`
- `tqdm`

A GPU is also preferred.

### [CLIP](https://github.com/openai/CLIP)

```sh
pip install git+https://github.com/openai/CLIP.git
```

## Setup

Before running any scripts, review `lib/hparams.py` and change any hyperparameters as necessary.

- This is where the path to the dataset should be defined. By default, it uses the `tiny-imagenet-200` dataset.

### Data

- Run `scripts/init.sh` to prepare the workspace. This creates folders for data, and generated indexes and encodings.
- If you want to use the default dataset (`tiny-imagenet-200`), run `scripts/tiny-imagenet-200.sh`.
- Make sure that any datasets (AKA repositories) you want to use are accessible with a relative/absolute Unix path from the base of the dataset/repository).

### Indexing

The indexes will be stored in `indexes` and have a filename indicating dataset and number of features (which may vary given the CLIP model and FAISS compression method used). After each dataset generates an index, its (`dataset_name`, `dataset_path`) are added to `collection.txt` if they weren't already there. This provides an easy reference to reconstruct an ordered compound index and dataloader.

#### Generating Image Indexes

To generate an image index, run `python build_image_index.py <dataset_name> <dataset_path> <n_components>`, where

- `dataset_name` is the name of the dataset/repository you would like to index.
- `dataset_path` is the relative/absolute filepath to the dataset. The Dataloader will recursively include all images under this directory.
- `n_components` is the number of components the feature vectors will contain. PCA compression is coming soon.
- These values have default values in `lib/hparams.py`.

#### Generating Text Indexes

1. Review the hyperparameters in the `vocabulary` section of `lib/hparams.py`. Give the vocabulary a name and define the URL from which it can be retrieved.
2. Run `python build_text_index.py` and review the indexes in `indexes_text/`. The default configuration should create the following file subtree.

   - Text indexes might take a while because they're not partitioned yet.

   ```
   indexes_text/
      | text_aidemos_512.index
      | text_aidemos_1024.index
   ```

## Deployment

- Run `sh scripts/api.sh` from the base of the repository to start the server (default on port `5020`).

## Runtime Usage

To interface with the API, deploy a frontend like [rusbridger/Multimodal-CLIP-Applications-Frontend](https://github.com/rusbridger/Multimodal-CLIP-Applications-Frontend).

## Cleaning Up

To clean up the workspace, run `sh scripts/clean.sh` with optional arguments (detailed in the script).
