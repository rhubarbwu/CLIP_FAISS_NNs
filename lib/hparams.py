# dataset, default tiny-imagenet-200
from .data import tiny_imagenet_200_classes

dataset_name = "tiny-imagenet-200-val"
dataset_path = "../data/tiny-imagenet-200/val"

# CLIP/FAISS parameters
# model_selection, n_components = "RN50", 1024  # model that CLIP should use
model_selection, n_components = "ViT-B/32", 512  # alternate CLIP model
n_neighbours = 25  # how many neighbours FAISS should report per query

# vocabulary
"""
For image classification, define where to retrieve the text labels used.

By default, the JSON dictionary is parsed in Python and flattened into a single text-sorted list.
"""
vocab_name = "aidemos"
vocab_url = "http://aidemos.cs.toronto.edu/at-backend-test/resources/classification"
