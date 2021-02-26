# Multimodal CLIP Applications

## Dependencies

Make sure you have `wget`.

Install PyTorch and CLIP using the following commands.

```sh
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

Additionally install the following dependencies from the PyPI as needed, preferably using `pip`.

- `annoy`
- `faiss-cpu`
- `torchvision`

## Scripts

This project contains Shell and Python scripts (in `scripts/` and `lib/`), and they should all be invoked from the root directory of this repository. To get started, run `scripts/init.sh`.
