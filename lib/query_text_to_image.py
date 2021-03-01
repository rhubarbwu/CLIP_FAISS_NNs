from index import get_image_index_filename

import clip
import faiss
import torch
import torchvision

# Load the desired model. CAN BE AMORTIZED.
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(model_selection, device)


def query_text_to_image(dataset_name, n_neighbours, text):

    # Encode the text. Main bottleneck.
    text_input = clip.tokenize([text]).to(device)
    text_features = model.encode_text(text_input)

    # Construct filename and search for nearest neighbours. Negligible time.
    index_filename = get_image_index_filename(dataset_name, n_components)
    index = faiss.read_index(index_filename)
    _, top_k = index.search(text_features.detach().numpy(), n_neighbours)

    return top_k[0]


if __name__ == "__main__":

    dataset_name = "tiny-imagenet-200"
    dataset_size = 100  # TOY: 100000 for full tiny-imagenet-200
    n_components = 1024
    n_neighbours = 5
    n_samples = 100
    n_trees = 5
    assert n_samples <= dataset_size

    from time import time
    start_time = time()
    top_k = query_text_to_image(dataset_name, n_neighbours,
                                "goldfish, Carassius auratus")
    print(top_k, time() - start_time)

    start_time = time()
    top_k = query_text_to_image(dataset_name, n_neighbours, "lionfish")
    print(top_k, time() - start_time)

    start_time = time()
    top_k = query_text_to_image(dataset_name, n_neighbours, "catfish")
    print(top_k, time() - start_time)

    start_time = time()
    top_k = query_text_to_image(dataset_name, n_neighbours, "tench")
    print(top_k, time() - start_time)