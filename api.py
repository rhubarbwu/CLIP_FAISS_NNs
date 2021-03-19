from flask import Flask, render_template, url_for, request
from io import StringIO
from numpy import savetxt

from lib import data, hparams, preprocessing

app = Flask("Multimodal CLIP Application Demo")
dataset = data.load_dataset(hparams.dataset_path)


@app.route("/encode-image", methods=["POST", "GET"])
def encode_image():
    if request.method == "POST":
        image_index = request.form["input"]
    else:
        image_index = request.args.get("input")

    encoding = preprocessing.encode_image(dataset, int(image_index))

    output = StringIO()
    savetxt(output, encoding)

    return render_template("encode-image.html", output=output.getvalue())


@app.route("/encode-text", methods=["POST", "GET"])
def encode_text():
    if request.method == "POST":
        text_input = request.form["input"]
    else:
        text_input = request.args.get("input")

    encoding = preprocessing.encode_text([text_input])

    output = StringIO()
    savetxt(output, encoding)

    return render_template("encode-text.html", output=output.getvalue())
