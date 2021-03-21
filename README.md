# Multimodal CLIP Applications

## Dependencies

Make sure you have `wget`.

Install PyTorch and CLIP using the following commands.

```sh
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
# or however else you can install PyTorch like if you don't have a CUDA GPU

pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

Additionally install the following dependencies from the PyPI as needed, preferably using `pip`.

- `flask` (for deployment, see below.)
- `faiss-cpu`
- `torchvision`

## Data

- Run `scripts/init.sh` to prepare the workspace.
- Run `scripts/tiny-imagenet-200.sh` to use the default dataset (`tiny-imagenet-200`) to encode images.

### Deployment

- Run `scripts/api.sh` from the base of the repository to start the server (default on port `5020`).
- You can use the examples in `frontend` to make GET/POST requests.

## Scripts

This project contains Shell and Python scripts (in `scripts/` and `lib/`), and they should all be invoked from the root directory of this repository. To get started, run `scripts/init.sh`.
