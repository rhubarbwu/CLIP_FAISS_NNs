from lib.hparams import dataset_name, dataset_path
from lib.index import build_image_index_faiss

from sys import argv
from time import time

if len(argv) >= 3:
    dataset_name, dataset_path, n_components = argv[1], argv[2]
if len(argv) >= 4:
    n_components = int(argv[3])

start_time = time()
build_image_index_faiss(dataset_name,
                        dataset_path,
                        n_components=n_components,
                        verbose=True)
print("Image indexing completed in {}.".format(time() - start_time))
