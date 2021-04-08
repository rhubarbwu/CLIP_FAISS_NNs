from faiss import IndexIDMap, IndexFlatIP, read_index

from lib.data.vocabulary import build_vocabulary
from lib.hparams import n_components, n_neighbours, vocab_url
from lib.preprocessing.model import *

vocab = build_vocabulary(vocab_url)


def classify_image(repository_indexes, image_path):
    index = IndexFlatIP(n_components)

    for filename in repository_indexes:
        curr_index = read_index("indexes_text/{}".format(filename))
        v = curr_index.reconstruct_n(0, curr_index.ntotal)
        if curr_index.d != n_components:
            print(
                "Index \"{}\" does not have the correct number of compoennts, excluding it from the index."
                .format(filename))
            continue

        index.add(v)

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    encodings = model.encode_image(image).detach().cpu().numpy().astype(
        'float32')

    D, I = index.search(encodings, n_neighbours)

    results = [vocab[i] for i in I[0]]

    return results