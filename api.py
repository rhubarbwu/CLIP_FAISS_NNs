from flask import Flask, jsonify
from flask_cors import CORS

from lib.index import *

app = Flask("Multimodal CLIP Application Demo")
CORS(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/repos", methods=["GET"])
def get_repos():
    repository_indexes = build_image_index_list()
    return jsonify({"repos": repository_indexes})
