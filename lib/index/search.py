from faiss import IndexIDMap, IndexFlatIP, read_index
from numpy import arange

from lib.hparams import n_components, n_neighbours
from lib.preprocessing import encode_image, encode_text
from lib.preprocessing.model import *


def search_image(repository_indexes, text_query):
    index = IndexIDMap(IndexFlatIP(n_components))

    for filename in repository_indexes:
        curr_index = read_index("indexes_images/{}".format(filename))
        v = curr_index.reconstruct_n(0, curr_index.ntotal)
        if curr_index.d != n_components:
            print(
                "Index \"{}\" does not have the correct number of compoennts, excluding it from the index."
                .format(filename))
            continue

        idx = arange(index.ntotal, index.ntotal + curr_index.ntotal)
        index.add_with_ids(v, idx)

    encodings = encode_text([text_query])
    D, I = index.search(encodings, n_neighbours)

    return "Top {} similar image results: {}".format(n_neighbours, str(I[0]))
