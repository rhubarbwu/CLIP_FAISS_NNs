from faiss import IndexIDMap, IndexFlatIP, read_index
from flask import Flask, request, send_file
from io import StringIO
from numpy import arange, save
from uuid import uuid4

from lib import data, hparams, preprocessing
from lib.hparams import n_components, n_neighbours
from lib.preprocessing import encode_image, encode_text

app = Flask("Multimodal CLIP Application Demo")
dataset = data.load_dataset(hparams.dataset_path)


def save_encoding_for_download(type, encoding):
    filename = "encoding_{}_1024_".format(type) + str(uuid4())
    save("encodings/" + filename, encoding)
    filename += ".npy"
    return send_file("encodings/" + filename,
                     as_attachment=True,
                     attachment_filename=filename)


@app.route("/encode-image", methods=["POST"])
def encode_image_request():
    if request.method == "POST":
        image_index = request.form["input"]
    else:
        image_index = request.args.get("input")

    encoding = preprocessing.encode_image(dataset, int(image_index))
    return save_encoding_for_download("image", encoding)


@app.route("/encode-text", methods=["POST"])
def encode_text_request():
    if request.method == "POST":
        text_input = request.form["input"]
    else:
        text_input = request.args.get("input")

    encoding = preprocessing.encode_text([text_input])
    return save_encoding_for_download("text", encoding)


@app.route("/image-classification", methods=["POST", "GET"])
def image_classification():
    return "Unimplemented!"


@app.route("/image-search", methods=["POST"])
def image_search():
    index = IndexIDMap(IndexFlatIP(n_components))

    checked_filenames = request.form.getlist("check")
    for filename in checked_filenames:
        curr_index = read_index("indexes/{}".format(filename))
        v = curr_index.reconstruct_n(0, curr_index.ntotal)
        if curr_index.d != n_components:
            print(
                "Index \"{}\" does not have the correct number of compoennts, excluding it from the index."
                .format(filename))
            continue

        idx = arange(index.ntotal, index.ntotal + curr_index.ntotal)
        index.add_with_ids(v, idx)

    text = request.form.get("input")
    encodings = encode_text([text])
    D, I = index.search(encodings, n_neighbours)

    return "Top {} similar class results: {}".format(n_neighbours, str(I[0]))
