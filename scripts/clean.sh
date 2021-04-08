#!/bin/sh

case $1 in
"data") rm -rf data/* ;;
"indexes") rm -rf indexes*/*.index ;;
esac

rm -rf **/__pycache__
