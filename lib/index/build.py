from ..data import *
from ..hparams import *
from ..preprocessing import *
from ..preprocessing.encode import encode_img, encode_txt
from .utils import *

import faiss, numpy as np, pickle, torchvision


def build_img_index_faiss(dataset_name,
                          dataset_path,
                          n_images=None,
                          n_components=n_components,
                          model_selection=model_selection,
                          verbose=False):

    if verbose:
        print("Building new image FAISS index for {} with {} components...".
              format(dataset_name, n_components))

    # Load the dataset.
    dataset = load_images(dataset_path)

    # Construct index.
    index = faiss.IndexFlatIP(n_components)

    # Add each image's encoding to the index.
    for i in range(len(dataset) if n_images is None else n_images):
        img_features = encode_img(dataset, i).astype('float32')
        index.add(img_features)
        if verbose:
            print("Image {} encoded and added to index.".format(i))

    # Write to index.
    index_filename = get_image_index_filename(dataset_name, n_components)
    faiss.write_index(index, index_filename)


def build_txt_index_faiss(dataset_name,
                          text_values,
                          n_components=n_components,
                          classes=None,
                          model_selection=model_selection,
                          verbose=False):

    if verbose:
        print("Building new text FAISS index for {} components...".format(
            n_components))

    # Set up index.
    index = faiss.IndexFlatIP(n_components)

    # Add each text_value's encoding.
    for text_value in text_values:
        text_features = encode_txt([text_value])
        index.add(text_features)

    # Get filename and write to disk.
    index_filename = get_txt_index_filename(dataset_name, n_components)
    faiss.write_index(index, index_filename)
