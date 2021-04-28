from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from random import sample

from lib.data import *
from lib.index.query import classify_image

app = Flask("Multimodal CLIP Application Demo")
CORS(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

datasets = build_image_dataset_map()
subset_preview_length = 6


@app.route("/pre_repos", methods=["GET"])
def get_repos():
    return jsonify({"repos": sorted(list(datasets.keys()))})


@app.route("/pre_imgs", methods=["POST"])
def get_images():
    data = request.json
    mode = data["mode"]["id"]
    repos = data["repos"]

    subsets, subset_size = build_image_data_subset(datasets, repos)

    subset_indices = sample(range(subset_size), subset_preview_length)
    filepaths = [(i, index_into_subsets(subsets, i)) for i in subset_indices]

    return jsonify({"filepaths": filepaths})


@app.route("/query", methods=["POST"])
def query():
    data = request.json
    mode = data["mode"]["id"]
    repos = data["repos"]
    index = data["index"]
    nnn = data["n_neighbours"]

    classified, filepaths = [], []
    if mode == "#classify":
        classified = classify_image(repos, ["aidemos"], index, nnn)
    else:
        print("Not implemented!")

    return jsonify({"classified": classified, "filepaths": filepaths})
