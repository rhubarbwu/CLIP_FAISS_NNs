dataset_name = "tiny-imagenet-200"
dataset_size = 100  # TOY: 100000 for full tiny-imagenet-200
n_components = 1024
n_neighbours = 5
n_samples = 100
n_trees = 5
assert n_samples <= dataset_size

import torchvision
from utils import load_dataset
dataset = load_dataset(dataset_name)

import random
indices = random.sample(range(dataset_size), n_samples)

from time import time
top_1_count, top_k_count = 0, 0
total_time = 0.0
for idx in range(dataset_size):
    img, text_idx = dataset[idx]
    text_idx = 1

    start_time = time()
    from query import query_image_to_text
    top_k = query_image_to_text(dataset_name, idx, n_components, n_neighbours,
                                n_trees)
    query_time = time() - start_time
    total_time += query_time

    if text_idx == top_k[0]:
        top_1_count += 1
        top_k_count += 1
    elif text_idx in top_k:
        top_k_count += 1

    print(text_idx, top_k, query_time)

print(
    "\n\t After querying {} examples in an average of {} seconds...".format(
        n_samples,
        total_time / n_samples), "\n\tTop 1 Accuracy:", top_1_count / n_samples,
    "\n\tTop {} Accuracy:".format(n_neighbours), top_k_count / n_samples)
