#!/bin/sh

mkdir encodings data indexes indexes_text
cd data
wget https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
