from lib.data.vocabulary import build_vocabulary
from lib.hparams import n_components, vocab_name, vocab_url
from lib.index import *

from faiss import read_index
from os import listdir
from time import time


def build_text_indexes():
    start_time = time()

    keys = build_vocabulary(vocab_url)

    get_text_index_filename(vocab_name, n_components)
    build_text_index_faiss(vocab_name, text_values=keys)
    print("Text indexing of {} values completed in {}.".format(
        len(keys),
        time() - start_time))


def build_text_index_list():
    index_filenames = listdir("indexes_text/")
    for i in range(len(index_filenames) - 1, -1, -1):
        index = read_index("indexes_text/" + index_filenames[i])
        if index.d != n_components:
            index_filenames.pop(i)

    with open("frontend/image-classification-base.html") as f:
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

    with open("frontend/image-classification.html", "w") as f:
        f.writelines(lines)
    f.close()


build_text_indexes()
build_text_index_list()
