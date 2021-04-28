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


build_text_indexes()
