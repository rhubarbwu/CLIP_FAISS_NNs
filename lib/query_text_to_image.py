dataset_name = "tiny-imagenet-200"
dataset_size = 100  # TOY: 100000 for full tiny-imagenet-200
n_components = 1024
n_neighbours = 5
n_samples = 100
n_trees = 5
assert n_samples <= dataset_size

from query import query_text_to_image
top_k = query_text_to_image(dataset_name, n_components, n_neighbours, n_trees,
                            "goldfish, Carassius auratus")
print(top_k)