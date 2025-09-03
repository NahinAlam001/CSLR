#!/bin/bash

# Set up environment and dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download data (reproducible via kagglehub)
python -c "import kagglehub; import os; path = kagglehub.dataset_download('rabeyaakter23/how2sign-i3d-features-mediapipe-features'); os.environ['FEATURE_DIR'] = path"

# Set paths in config
export TSV_DIR="$FEATURE_DIR/tsv_files_how2sign/tsv_files_how2sign"
yq e -i '.feature_dir = env(FEATURE_DIR)' config.yaml
yq e -i '.tsv_dir = env(TSV_DIR)' config.yaml

# Train and evaluate
python src/train.py
python src/evaluate.py

# TensorBoard for tracking (experiment evolution)
tensorboard --logdir runs/ &
