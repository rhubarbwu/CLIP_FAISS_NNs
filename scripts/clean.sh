#!/bin/sh
case $1 in
"data") rm -rf data/* ;;
*) rm -rf indexes/*.index maps/*.pickle ;;
esac

rm -rf **/__pycache__