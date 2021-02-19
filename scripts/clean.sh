#!/bin/sh

case $1 in
"data")
    rm -rf data/*
    ;;
*)
    rm -rf indexes/*.ann maps/*.pickle
    ;;
esac
