from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from random import sample

from lib.data import *
from lib.index.query import classify_img, search_sim, search_txt

app = Flask("Multimodal CLIP Application Demo")
CORS(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

img_repos = build_img_repo_map()
txt_repos = build_txt_repo_map()
subset_preview_length = 6


@app.route("/repos/images", methods=["GET"])
def get_img_repos():
    return jsonify({"repos": sorted(list(img_repos.keys()))})


@app.route("/repos/text", methods=["GET"])
def get_txt_repos():
    return jsonify({"repos": sorted(list(txt_repos.keys()))})


@app.route("/repos/images", methods=["POST"])
def get_imgs():
    data = request.json
    mode = data["mode"]["id"]
    repos = data["repos"]

    subsets, subset_size = build_img_data_subset(img_repos, repos)
    subset_indices = sample(range(subset_size), subset_preview_length)
    filepaths = [(int(i), index_into_subsets(subsets, i))
                 for i in subset_indices]

    return jsonify({"filepaths": filepaths})


@app.route("/classify", methods=["POST"])
def classify():
    data = request.json
    repos = data["repos"]
    txt_repos = data["txt_repos"]
    index = data["index"]
    nnn = data["n_neighbours"]
    classified = classify_img(repos, txt_repos, index, nnn)

    return jsonify({"classified": classified})


@app.route("/search", methods=["POST"])
def search():
    data = request.json
    repos = data["repos"]
    query = "a picture of {}".format(data["query"])
    nnn = data["n_neighbours"]

    subsets = build_img_data_subset(img_repos, repos)
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

    subsets = build_img_data_subset(img_repos, repos)
    result_indices = search_sim(repos, index, nnn)
    filepaths = [(int(i), index_into_subsets(subsets, i))
                 for i in result_indices]

    return jsonify({"filepaths": filepaths})
