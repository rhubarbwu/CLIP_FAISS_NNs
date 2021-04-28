from faiss import IndexIDMap, IndexFlatIP, read_index
from numpy import array

from lib.data.vocabulary import build_vocabulary
from lib.hparams import n_components, vocab_url
# from lib.preprocessing.model import *

vocab = build_vocabulary(vocab_url)


def merge_index(repos, n_components, directory):
    index = IndexFlatIP(n_components)
    for repo in repos:
        curr_index = read_index("{}/{}_{}.index".format(
            directory, repo, n_components))
        v = curr_index.reconstruct_n(0, curr_index.ntotal)
        if curr_index.d != n_components:
            print(
                "Index \"{}\" does not have the correct number of compoennts, excluding it from the index."
                .format(repo))
            continue
        index.add(v)

    return index


def classify_image(img_repos, text_repos, i, nnn):

    img_index = merge_index(img_repos, n_components, "indexes_images")
    img_encoding = array([img_index.reconstruct(i)])

    text_index = merge_index(text_repos, n_components, "indexes_text")
    D, I = text_index.search(img_encoding, nnn)
    results = [vocab[i] for i in I[0]]

    return results


def search_image(img_repos, i, nnn):
    return results


def search_text(img_repos, i, nnn):
    return results