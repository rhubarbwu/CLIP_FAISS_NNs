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
- Make sure that any datasets (AKA repositories) you want to use are accessible with a relative/absolute Unix path from the base of the repository).

### Indexing

The indexes will be stored in `indexes` and have a filename indicating dataset, class, and number of features (which may vary given the CLIP model and FAISS compression method used).

#### Generating Image Indexes

1. To work with a single dataset, in `lib/hparams.py` `dataset` section, define the `dataset_name` and `dataset_path` as the label and relative path to the dataset.
2. Then, define how the the dataset will be partitioned. The hyperparameters and their default configuration for partition are define in `lib/hparams.py` under the `partition` section.
   - The generation of each part's index will print a progress message. The time it takes could vary depending on partition sizes used.
3. Run `python build_image_index.py` and review the indexes in `indexes/`.

   ```
   indexes/
      | image_tiny-imagenet-200_n01443537_1024.index
      | image_tiny-imagenet-200_n01629819_1024.index
      | image_tiny-imagenet-200_n01641577_1024.index
      | image_tiny-imagenet-200_n01644900_1024.index
      | image_tiny-imagenet-200_n01698640_1024.index
      | image_tiny-imagenet-200_n01742172_1024.index
      | image_tiny-imagenet-200_n01768244_1024.index
      | image_tiny-imagenet-200_n01770393_1024.index
      | image_tiny-imagenet-200_n01774384_1024.index
      | image_tiny-imagenet-200_n01774750_1024.index
   ```

4. Review the generated prototype `image-search.html` in `frontend`.

#### Generating Text Indexes

1. Review the hyperparameters in the `vocabulary` section of `lib/hparams.py`. Give the vocabulary a name and define the URL from which it can be retrieved.
2. Run `python build_text_index.py` and review the indexes in `indexes_text/`. The default configuration should create the following file subtree.

   - Text indexes might take a while because they're not partitioned yet.

   ```
   indexes_text/
      | text_aidemos_512.index
      | text_aidemos_1024.index
   ```

3. Review the generated prototype `image-classification.html` in `frontend`.

## Deployment

- Run `sh scripts/api.sh` from the base of the repository to start the server (default on port `5020`).

## Runtime Usage

To interface with the API, deploy a frontend like [rusbridger/Multimodal-CLIP-Applications-Frontend](https://github.com/rusbridger/Multimodal-CLIP-Applications-Frontend).

## Cleaning Up

To clean up the workspace, run `sh scripts/clean.sh` with optional arguments (detailed in the script).
