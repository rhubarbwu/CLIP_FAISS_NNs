from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from random import sample

from lib.data import *
from lib.index.query import classify_img, search_sim, search_txt

app = Flask("Multimodal CLIP Application Demo")
CORS(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

datasets = build_img_dataset_map()
subset_preview_length = 6


@app.route("/pre_repos", methods=["GET"])
def get_repos():
    return jsonify({"repos": sorted(list(datasets.keys()))})


@app.route("/pre_imgs", methods=["POST"])
def get_imgs():
    data = request.json
    mode = data["mode"]["id"]
    repos = data["repos"]

    subsets, subset_size = build_img_data_subset(datasets, repos)
    subset_indices = sample(range(subset_size), subset_preview_length)
    filepaths = [(int(i), index_into_subsets(subsets, i))
                 for i in subset_indices]

    return jsonify({"filepaths": filepaths})


@app.route("/classify", methods=["POST"])
def classify():
    data = request.json
    repos = data["repos"]
    index = data["index"]
    nnn = data["n_neighbours"]
    classified = classify_img(repos, ["aidemos"], index, nnn)

    return jsonify({"classified": classified})


@app.route("/search", methods=["POST"])
def search():
    data = request.json
    repos = data["repos"]
    query = "a picture of {}".format(data["query"])
    nnn = data["n_neighbours"]

    subsets, _ = build_img_data_subset(datasets, repos)
    result_indices = search_txt(repos, query, nnn)
    filepaths = [(int(i), index_into_subsets(subsets, i))
                 for i in result_indices]

    return jsonify({"filepaths": filepaths})


@app.route("/similar", methods=["POST"])
def similar():
    data = request.json
    repos = data["repos"]
    index = data["index"]
    nnn = data["n_neighbours"]

    subsets, _ = build_img_data_subset(datasets, repos)
    result_indices = search_sim(repos, index, nnn)
    filepaths = [(int(i), index_into_subsets(subsets, i))
                 for i in result_indices]

    return jsonify({"filepaths": filepaths})
