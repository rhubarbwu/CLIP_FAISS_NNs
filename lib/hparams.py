# dataset, default tiny-imagenet-200
from .data import tiny_imagenet_200_classes

dataset_name = "tiny-imagenet-200"
dataset_path = "data/tiny-imagenet-200/train"

# CLIP/FAISS parameters
model_selection, n_components = "RN50", 1024  # model that CLIP should use
# model_selection, n_components = "ViT-B/32", 512  # alternate CLIP model
n_neighbours = 25  # how many neighbours FAISS should report per query

# partition
""" 
The partition of the dataset used, to provide flexibility in including or excluding parts of a dataset during the runtime of the application.

- partition_size: an optional int used as a standard part size.
- partition: list of (label, part size) tuples that define what label each index will have and how large it is.

The default configuration is to use the tiny_imagenet_200_classes labels and split them into 500 parts for the first 10 classes (depending on n_classes).
"""
n_classes = 10
partition_size = 500
partition = [(label, partition_size)
             for label in tiny_imagenet_200_classes[:n_classes]]

# vocabulary
"""
For image classification, define where to retrieve the text labels used.

By default, the JSON dictionary is parsed in Python and flattened into a single text-sorted list.
"""
text_list_file = "data/tiny-imagenet-200/words.txt"
vocab_name = "aidemos"
vocab_url = "http://aidemos.cs.toronto.edu/at-backend-test/resources/classification"
