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
    index = faiss.IndexFlatIP(n_components)

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
                           verbose=False,
                           text_values=None):

    if verbose:
        print("Building new text FAISS index for {} components...".format(
            n_components))

    # Build text list and map and encode.
    if text_values is None:
        text_values, text_id_to_value_map = build_text_id_to_value_map(classes)

    # Set up index.
    index = faiss.IndexFlatIP(n_components)

    # Add each text_value's encoding.
    for text_value in text_values:
        text_features = encode_text([text_value])
        index.add(text_features)

    # Get filename and write to disk.
    index_filename = get_text_index_filename(dataset_name, n_components)
    faiss.write_index(index, index_filename)
