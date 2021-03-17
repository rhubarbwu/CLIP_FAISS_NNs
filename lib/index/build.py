from ..data import *
from ..hparams import *
from ..preprocessing import *
from .utils import *

import faiss, numpy as np, pickle, torchvision


def build_image_index_faiss(
        dataset_name,
        dataset_path,
        partition=None,  # defined as (start, len)
        label=None,
        n_components=n_components,
        model_selection=model_selection,
        verbose=False):

    if verbose:
        print("Building new image FAISS index for {} components...".format(
            n_components))

    # Load the dataset and build map.
    dataset = load_dataset(dataset_path)

    # Compute image partition.
    if partition is None:
        partition = (0, len(dataset.imgs))
    elif type(partition) == int:
        partition = (0, partition)
    start, end = partition[0], partition[0] + partition[1]

    # Construct index.
    index = faiss.IndexFlatL2(n_components)

    # Add each image's encoding to the index.
    for i in range(start, end):
        img_features = encode_image(dataset, i).astype('float32')
        index.add(img_features)

    # Write to index.
    index_filename = get_image_index_filename(dataset_name, n_components,
                                              label)
    faiss.write_index(index, index_filename)
    if verbose:
        print("Index for {} written to {}.".format(label, index_filename))


def build_text_index_faiss(dataset_name,
                           n_components=n_components,
                           classes=None,
                           model_selection=model_selection,
                           verbose=False):

    if verbose:
        print("Building new text FAISS index for {} components...".format(
            n_components))

    # Build text list and map and encode.
    text_values, text_id_to_value_map = build_text_id_to_value_map(classes)
    text_features = encode_text(text_values)

    # Filename of the index to load/write to.
    index_filename = get_text_index_filename(dataset_name, n_components)

    # Build NNs Index.
    n, c = text_features.shape
    quantizer = faiss.IndexFlatIP(c)
    index = faiss.IndexIDMap(quantizer)
    idx = np.arange(n)
    index.add_with_ids(text_features, idx)
    faiss.write_index(index, index_filename)
