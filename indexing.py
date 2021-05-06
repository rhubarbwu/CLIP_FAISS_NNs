from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from json import dump
from os import environ

from lib.data import *
from lib.index.build import build_img_index_faiss, build_txt_index_faiss
from lib.data.collection import update_collection_images, update_collection_text
from lib.index.query import classify_img, search_sim, search_txt

app = Flask("Multimodal CLIP Application Indexing")
CORS(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/api/add-image-repo", methods=["POST"])
def add_image_repo():
    if "BLOCKING" not in environ:
        return jsonify({}), 403

    data = request.json
    name, dataset_path = data["name"], data["path"]

    try:
        n_total = build_img_index_faiss(name, dataset_path, verbose=True)
        update_collection_images(name, dataset_path)
        return jsonify({"size": n_total}), 200
    except:
        return jsonify({}), 500


@app.route("/api/add-text-repo", methods=["POST"])
def add_text_repo():
    if "BLOCKING" not in environ:
        return jsonify({}), 403

    data = request.json
    name, vocab = data["name"], data["vocab"]

    text_values = load_text(vocab)

    try:
        build_txt_index_faiss(name,
                              text_values,
                              n_components=n_components,
                              verbose=True)

        filepath = "vocab/{}.json".format(name)
        update_collection_text(name, filepath)
        with open(filepath, "w") as outfile:
            dump(vocab, outfile)

        return jsonify({"size": len(text_values)})

    except:
        return jsonify({}), 500
