from flask import Flask, request, send_file
from io import StringIO
from numpy import save
from uuid import uuid4

from lib import data, hparams, preprocessing

app = Flask("Multimodal CLIP Application Demo")
dataset = data.load_dataset(hparams.dataset_path)


def save_encoding_for_download(type, encoding):
    filename = "encoding_{}_1024_".format(type) + str(uuid4())
    save("encodings/" + filename, encoding)
    filename += ".npy"
    return send_file("encodings/" + filename,
                     as_attachment=True,
                     attachment_filename=filename)


@app.route("/encode-image", methods=["POST", "GET"])
def encode_image():
    if request.method == "POST":
        image_index = request.form["input"]
    else:
        image_index = request.args.get("input")

    encoding = preprocessing.encode_image(dataset, int(image_index))
    return save_encoding_for_download("image", encoding)


@app.route("/encode-text", methods=["POST", "GET"])
def encode_text():
    if request.method == "POST":
        text_input = request.form["input"]
    else:
        text_input = request.args.get("input")

    encoding = preprocessing.encode_text([text_input])
    return save_encoding_for_download("text", encoding)
