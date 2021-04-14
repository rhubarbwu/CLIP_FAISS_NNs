from lib.data import load_dataset
from lib.hparams import n_components, collection

from faiss import read_index
from os.path import exists


def build_image_dataset_list():
    datasets = dict()

    for repo in collection:
        repo_name, repo_path = repo[0], repo[1]
        index_filename = "indexes_images/{}_{}.index".format(
            repo_name, n_components)

        if exists(index_filename) and exists(repo_path):
            datasets[repo_name] = load_dataset(repo_path)

    return datasets