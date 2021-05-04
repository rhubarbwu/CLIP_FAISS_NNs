#!/bin/sh

export FLASK_APP=api
export FLASK_ENV=development
export FLASK_RUN_PORT=5020
export FLASK_RUN_HOST=0.0.0.0
flask run
