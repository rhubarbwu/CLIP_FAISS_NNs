from lib.data import tiny_imagenet_200_classes
from lib.hparams import n_components
from lib.index import *

from faiss import read_index
from os import listdir
from time import time


def build_image_indexes():
    start_time = time()
    partitions = tiny_imagenet_200_classes[:5]
    for i in range(len(partitions)):
        build_image_index_faiss(dataset_name,
                                dataset_path,
                                partition=(i * partition_size, partition_size),
                                label=partitions[i],
                                verbose=True)
    print("Image indexing completed in {}.".format(time() - start_time))


def build_image_index_list():
    index_filenames = listdir("indexes/")
    for i in range(len(index_filenames) - 1, -1, -1):
        index = read_index("indexes/" + index_filenames[i])
        if index.d != n_components:
            index_filenames.pop(i)

    with open("frontend/image-search-base.html") as f:
        lines = f.readlines()
    f.close()

    for i in range(len(lines)):
        if "<legend>" in lines[i]:
            for filename in index_filenames:
                new_line = "<input type=\"checkbox\" name=\"check\" value=\"{}\">{}<br>\n".format(
                    filename, filename)
                lines.insert(i + 1, new_line)
                i += 1
            break

    with open("frontend/image-search.html", "w") as f:
        f.writelines(lines)
    f.close()


build_image_indexes()
build_image_index_list()