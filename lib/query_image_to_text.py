from annoy import AnnoyIndex
from hparams import *
from index import get_image_index_filename, get_text_index_filename

import clip
import faiss
import torch


def query_image_to_text_annoy(dataset_name, n_neighbours, idx):

    # Load the image AnnoyIndex and get the encoding of the image for idx.
    image_index_filename = get_image_index_filename(dataset_name, n_components,
                                                    n_trees)
    image_index = AnnoyIndex(n_components, "angular")
    image_index.load(image_index_filename)
    img_features = image_index.get_item_vector(idx)

    # Load the text AnnoyIndex.
    text_index = AnnoyIndex(n_components, "angular")
    text_index.load(get_text_index_filename(dataset_name, n_components,
                                            n_trees))

    # For the image encoding, find the nearest neighbour text encodings.
    top_k = text_index.get_nns_by_vector(img_features, n_neighbours)
    return top_k


def query_image_to_text_faiss(dataset_name, n_neighbours, idx):

    # Load the image FAISS index and get the encoding of the image for idx.
    image_index_filename = get_image_index_filename(dataset_name, n_components)
    image_index = faiss.read_index(image_index_filename)
    img_features = torch.from_numpy(image_index.reconstruct(idx)).unsqueeze(0)

    # Load the text FAISS index.
    text_index_filename = get_text_index_filename(dataset_name, n_components)
    text_index = faiss.read_index(text_index_filename)

    _, top_k = text_index.search(img_features.detach().numpy(), n_neighbours)

    return top_k[0]


if __name__ == "__main__":
    dataset_size = 10000
    n_samples = 10000
    assert n_samples <= dataset_size

    import torchvision
    dataset = torchvision.datasets.ImageFolder(dataset_path)

    import random
    indices = random.sample(range(dataset_size), n_samples)

    from utils import build_text_id_to_value_map
    text, text_list = build_text_id_to_value_map()

    from time import time
    top_1_count, top_k_count = 0, 0
    total_time = 0.0
    for idx in range(n_samples):
        img, text_idx = dataset[idx]

        start_time = time()
        query_image_index_func = query_image_to_text_annoy
        top_k = query_image_index_func(dataset_name, n_neighbours, idx)
        query_time = time() - start_time
        total_time += query_time

        top_k = [text_list[i][0] for i in top_k]
        text_id = dataset.classes[text_idx]

        if text_id == top_k[0]:
            top_1_count += 1
            top_k_count += 1
        elif text_id in top_k:
            top_k_count += 1

        print(text_id, top_k, query_time)

    print(
        "\n\t After querying {} examples in an average of {} seconds...".format(
            n_samples, total_time / n_samples), "\n\tTop 1 Accuracy:",
        top_1_count / n_samples, "\n\tTop {} Accuracy:".format(n_neighbours),
        top_k_count / n_samples)
