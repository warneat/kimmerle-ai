#!/bin/bash

# Set up a new Python virtual environment
rm -rf venv/
python3.9 -m venv venv
python3.9 --version
source /bin/activate

# Upgrade pip and setuptools, install requirements
pip install --upgrade pip setuptools
pip install -r requirements.txt

# Clean labelfiles
python3.9 ./clean_csv.py

# Run the script
python ai-kimmerle.py

# Deactivate the virtual environment
deactivate

# Notify that the script has finished running
echo "Script execution completed."
