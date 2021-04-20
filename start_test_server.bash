#!/bin/bash

export FLASK_ENV=production
export FLASK_APP=flask_server.py

flask run --port=5008
