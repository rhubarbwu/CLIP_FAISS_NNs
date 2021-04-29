from lib.data.dataset import load_text
from lib.hparams import collection_text, n_components
from lib.index import build_txt_index_faiss

from sys import argv
from time import time


def update_collection_text(dataset_name, dataset_path):

    if (dataset_name, dataset_path) in collection_text:
        return

    collection_text.append((dataset_name, dataset_path))
    lines = []
    for (dataset_name, dataset_path) in collection_text:
        lines.append("{} {}\n".format(dataset_name, dataset_path))

    with open("collection_text.txt", "w") as f:
        f.writelines(lines)
    f.close()


if len(argv) <= 2:
    print("Please specify dataset/repository name.")
    exit(0)
dataset_name, dataset_path = argv[1], argv[2]
if len(argv) >= 4:
    n_components = int(argv[3])

start_time = time()
text_values = load_text(dataset_path)

build_txt_index_faiss(dataset_name,
                      text_values,
                      n_components=n_components,
                      verbose=True)
print("Image indexing completed in {}.".format(time() - start_time))
update_collection_text(dataset_name, dataset_path)
