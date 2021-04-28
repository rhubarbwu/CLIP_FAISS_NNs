#!/bin/sh

cd ../data
rm -rf tiny-imagenet-200.zip tiny-imagenet-200
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
