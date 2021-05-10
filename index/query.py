from ..data.dataset import build_txt_data_subset
from ..hparams import n_components
from ..preprocessing import encode_txt
from ..preprocessing.model import *
from .utils import *

from faiss import IndexIDMap, IndexFlatIP, read_index
from numpy import array


def merge_index(repos, n_components, directory):
    index = IndexFlatIP(n_components)
    for repo in repos:
        curr_index = read_index(
            get_image_index_filename(repo, directory, n_components))
        v = curr_index.reconstruct_n(0, curr_index.ntotal)
        if curr_index.d != n_components:
            print(
                "Index \"{}\" does not have the correct number of compoennts, excluding it from the index."
                .format(repo))
            continue
        index.add(v)

    return index


def classify_img(img_repos, txt_repos, image_repo_dir, text_repo_dir, i, nnn):
    img_index = merge_index(img_repos, n_components, image_repo_dir)
    img_encoding = array([img_index.reconstruct(i)])

    txt_index = merge_index(txt_repos, n_components, text_repo_dir)
    D, I = txt_index.search(img_encoding, nnn)
    results = I[0]

    return results


def search_sim(img_repos, image_repo_dir, i, nnn):
    img_index = merge_index(img_repos, n_components, image_repo_dir)
    img_encoding = array([img_index.reconstruct(i)])
    D, I = img_index.search(img_encoding, nnn + 1)

    return I[0][1:]


def search_txt(img_repos, image_repo_dir, txt_query, nnn):
    img_index = merge_index(img_repos, n_components, image_repo_dir)
    txt_encoding = encode_txt([txt_query])
    D, I = img_index.search(txt_encoding, nnn)

    return I[0]
