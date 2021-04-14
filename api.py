from flask import Flask, jsonify, request
from flask_cors import CORS
from random import sample

from lib.hparams import n_components
from lib.index import *
from lib.index.merge import build_image_dataset_list

app = Flask("Multimodal CLIP Application Demo")
CORS(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

datasets = build_image_dataset_list()
repo_names = sorted(list(datasets.keys()))

subset = None
subset_preview_indices = None
subset_preview_length = 16


@app.route("/pre_repos", methods=["GET"])
def get_repos():
    return jsonify({"repos": repo_names})


@app.route("/pre_imgs", methods=["POST"])
def pick_images():
    data = request.json
    mode = data["mode"]
    repos = data["repos"]

    subset = None
    for repo in repos:
        if repo not in datasets: continue
        if subset is None: subset = datasets[repo]
        else: subset += datasets[repo]

    subset_indices = sample(range(len(subset)), subset_preview_length)
    print(subset_indices)
    return jsonify({})