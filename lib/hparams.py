# collection of datasets/repositories available
with open("collection.txt") as f:
    lines = f.readlines()
    f.close()
collection = []
for line in lines:
    split = line.split()
    if len(split) == 2:
        collection.append((split[0], split[1]))

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
