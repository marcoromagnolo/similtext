#!/bin/bash

# Enable Python Virtual Env and Activate
python3 -m venv venv
source ./venv/bin/activate

# Install Python packages
pip install -r requirements.txt

# Create dirs
mkdir log
