#!/bin/bash

source ./venv/bin/activate

./venv/bin/python --version
pip install -r requirements.txt
./venv/bin/python app.py > ./log/similtext.out 2>&1 &
