dataset_name = "tiny-imagenet-200"
dataset_size = 100000
num_of_neighbours = 5
num_of_trees = 25
num_of_samples = 100
assert num_of_samples <= dataset_size

import os
from utils import get_class_index_filename
if not os.path.isfile(get_class_index_filename(dataset_name, num_of_trees)):
    from build_class_index import build_class_index
    build_class_index(dataset_name, num_of_trees, dataset_name)

import torchvision
from utils import load_dataset
dataset = load_dataset(dataset_name)

import random
indices = random.sample(range(dataset_size), num_of_samples)

import time
from query_image_text import query_image_to_text_one
top_1_count, top_k_count = 0, 0
total_time = 0.0
for idx in indices:
    img, class_idx = dataset[idx]  # Ground Truth (GT), original label.
    top_k, query_time = query_image_to_text_one(dataset_name, img,
                                                num_of_neighbours, num_of_trees)

    if class_idx == top_k[0]:
        top_1_count += 1
        top_k_count += 1
    elif class_idx in top_k:
        top_k_count += 1
    total_time += query_time

    print(class_idx, top_k, query_time)

print(
    "\n\t After querying {} examples in {} seconds...".format(
        num_of_samples, total_time / num_of_samples), "\n\tTop 1 Accuracy:",
    top_1_count / num_of_samples,
    "\n\tTop {} Accuracy:".format(num_of_neighbours),
    top_k_count / num_of_samples)
