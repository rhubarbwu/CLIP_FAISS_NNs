from lib.hparams import collection, n_components

from faiss import read_index
from os.path import exists
import torchvision


def load_dataset(dataset_path):
    return torchvision.datasets.ImageFolder(dataset_path)


def build_image_dataset_map():
    datasets = dict()
    count = 0

    for repo in sorted(collection):
        repo_name, repo_path = repo[0], repo[1]
        index_filename = "indexes_images/{}_{}.index".format(
            repo_name, n_components)

        if exists(index_filename) and exists(repo_path):
            dataset = load_dataset(repo_path)
            datasets[repo_name] = load_dataset(repo_path)
            count += len(dataset)

    return datasets


def build_image_data_subset(datasets, repos):
    subsets = []
    subset_size = 0
    for r in repos:
        subsets.append(datasets[r])
        subset_size += len(datasets[r])

    return subsets, subset_size


def index_into_subsets(subsets, index):
    curr_idx = 0
    for ss in subsets:
        if curr_idx + len(ss) >= index:
            return ss.imgs[index - curr_idx][0]
        curr_idx += len(ss)

    return None
