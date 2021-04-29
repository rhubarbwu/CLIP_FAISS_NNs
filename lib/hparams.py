# collection of image datasets/repositories available
open("collection_images.txt", 'a').close()
with open("collection_images.txt") as f:
    lines = f.readlines()
    f.close()
collection_images = []
for line in lines:
    split = line.split()
    if len(split) == 2:
        collection_images.append((split[0], split[1]))

# collection of text datasets/repositories available
open("collection_text.txt", 'a').close()
with open("collection_text.txt") as f:
    lines = f.readlines()
    f.close()
collection_text = []
for line in lines:
    split = line.split()
    if len(split) == 2:
        collection_text.append((split[0], split[1]))

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
