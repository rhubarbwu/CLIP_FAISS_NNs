from lib.hparams import collection, n_components
from lib.index import build_image_index_faiss

from sys import argv
from time import time


def update_collection(dataset_name, dataset_path):

    if (dataset_name, dataset_path) in collection:
        return

    collection.append((dataset_name, dataset_path))
    lines = []
    for (dataset_name, dataset_path) in collection:
        lines.append("{} {}\n".format(dataset_name, dataset_path))

    with open("collection.txt", "w") as f:
        f.writelines(lines)
    f.close()


if len(argv) <= 2:
    print("Please specify dataset/repository name.")
    exit(0)
dataset_name, dataset_path = argv[1], argv[2]
if len(argv) >= 4:
    n_components = int(argv[3])

start_time = time()
build_image_index_faiss(dataset_name,
                        dataset_path,
                        n_components=n_components,
                        verbose=True)
print("Image indexing completed in {}.".format(time() - start_time))
update_collection(dataset_name, dataset_path)
