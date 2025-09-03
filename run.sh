#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo ">>> Installing dependencies..."
pip install -r requirements.txt -Uqq

echo ">>> Downloading dataset from KaggleHub..."
# Run a Python script to get the download path and print it to stdout
FEATURE_DIR=$(python -c "import kagglehub; path = kagglehub.dataset_download('rabeyaakter23/how2sign-i3d-features-mediapipe-features'); print(path)")

# Check if the download was successful
if [ -z "$FEATURE_DIR" ]; then
    echo "Failed to download dataset or get feature directory path. Exiting."
    exit 1
fi
echo "Dataset downloaded to: $FEATURE_DIR"

# Define TSV directory based on the feature directory
TSV_DIR="$FEATURE_DIR/tsv_files_how2sign/tsv_files_how2sign"

echo ">>> Starting training..."
# Use 'python -m' and pass paths as arguments
python -m src.train --feature_dir "$FEATURE_DIR" --tsv_dir "$TSV_DIR"

echo ">>> Starting evaluation..."
# Use 'python -m' and pass paths as arguments
python -m src.evaluate --feature_dir "$FEATURE_DIR" --tsv_dir "$TSV_DIR"

echo ">>> Script finished. To view logs, run: tensorboard --logdir runs/"
