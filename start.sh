#!/bin/bash

source ./venv/bin/activate

./venv/bin/python --version
./venv/bin/pip install -r requirements.txt
./venv/bin/python app.py -- similtext > ./log/similtext.out 2>&1 &
