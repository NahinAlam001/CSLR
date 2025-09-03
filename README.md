# Multimodal Transformer-Based System for Continuous Sign Language Translation

This repository contains a refactored implementation of a transformer-based model for continuous sign language translation using I3D and MediaPipe features from the How2Sign dataset.

![alt text](https://github.com/NahinAlam001/CSLR/blob/main/CSLR.drawio.png?raw=true)

## Setup
1. Create virtual environment: `python -m venv venv; source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `./run.sh`

## Reproducibility
- Seeds set to 42 for all random processes.
- Config in YAML for easy modification.
- Logs in `logs/` directory.
- TensorBoard: `tensorboard --logdir runs/`

## Data
Downloads How2Sign features via KaggleHub. Ensure Kaggle credentials are set if needed.

## Citation
If using, cite the original work or this refactored version.
