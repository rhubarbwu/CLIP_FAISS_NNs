from lib.hparams import collection_images, n_components
from lib.index import build_img_index_faiss
from lib.index.collection import update_collection_images

from sys import argv
from time import time

if len(argv) <= 2:
    print("Please specify dataset/repository name.")
    exit(0)
dataset_name, dataset_path = argv[1], argv[2]
if len(argv) >= 4:
    n_components = int(argv[3])

start_time = time()
build_img_index_faiss(dataset_name,
                      dataset_path,
                      n_components=n_components,
                      verbose=True)
print("Image indexing completed in {}.".format(time() - start_time))
update_collection_images(dataset_name, dataset_path)
