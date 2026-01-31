#!/bin/bash

# Check if src directory exists
if [ ! -d "src" ]; then
    echo "Error: 'src' directory not found. Please run this script from the project root."
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export USE_TORCH=1
export USE_TF=0

# Install requirements if not already
echo "Installing dependencies..."
python3 -m pip install -r requirements.txt

# Run Streamlit
echo "Starting Heritage OCR App..."
python3 -m streamlit run src/main.py
