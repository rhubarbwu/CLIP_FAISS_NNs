from lib.hparams import n_components, collection

from faiss import read_index
from os.path import exists


def build_image_index_list():

    repo_names = []
    for repo in collection:
        repo_name = repo[0]
        index_filename = "indexes_images/{}_{}.index".format(
            repo_name, n_components)

        repo_path = repo[1]

        if exists(index_filename) and exists(repo_path):
            repo_names.append(repo_name)

    return repo_names