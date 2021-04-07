#!/bin/sh

case $1 in
"data") rm -rf data/* ;;
"indexes") rm -rf indexes*/*.index image-classification.html image-search.html ;;
*) rm -rf encodings/* ;;
esac

rm -rf **/__pycache__
