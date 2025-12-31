# Oxford 102 Flowers Classification System

## Overview
This repository contains a solution for the **AQREIGHT CV Engineer Technical Assessment**. It implements a robust end-to-end pipeline for fine-grained image classification on the Oxford 102 Flowers dataset using PyTorch. The solution utilizes transfer learning with a ResNet18 backbone, custom stratified data splitting, and includes explainability features.

## Features
- **Data Pipeline**: Custom `Flowers102Dataset` with stratified 70/15/15 splitting (Train/Val/Test) to handle class imbalance.
- **Model**: ResNet18 pretrained on ImageNet, fine-tuned for 102 flower classes.
- **Training**: Training loop with Early Stopping, Model Checkpointing, and TensorBoard logging.
- **Inference**: Standalone inference script for predicting flower species from images.
- **Explainability**: Grad-CAM implementation to visualize model focus.
- **Reproducibility**: Seeded operations for deterministic results.

## Project Structure
```
.
├── configs/             # Configuration files
├── notebooks/           # Jupyter notebooks for analysis and demos
│   ├── 01_EDA.ipynb     # Exploratory Data Analysis
│   ├── 02_Training.ipynb # Training Walkthrough
│   └── 03_Evaluation_and_Explainability.ipynb # Inference & Grad-CAM
├── src/                 # Source code
│   ├── data/
│   │   └── loader.py    # Dataset class and transforms
│   ├── model/           # Model checkpoints and binaries
│   ├── training/
│   │   └── train.py     # Training loop and main script
│   ├── utils/
│   │   └── common.py    # Utility functions (logging, seeding)
│   ├── explain.py       # Explainability tools (Grad-CAM)
├── logs/                # TensorBoard logs
├── checkpoints/         # Model checkpoints
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies
```

## Setup
1.  **Environment**: Ensure you have Python 3.8+ and PyTorch installed.
2.  **Dependencies**: Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Exploratory Data Analysis
Run `notebooks/01_EDA.ipynb` to analyze the dataset distribution and visualize samples.

### 2. Training
To train the model from scratch:
```bash
python src/training/train.py --epochs 25 --batch_size 32 --lr 0.001
```
Arguments:
- `--data_dir`: Path to download/store dataset (default: `../src/data`)
- `--dry_run`: Run a single epoch for testing.

**TensorBoard Monitoring**:
```bash
tensorboard --logdir logs
```

### 3. Evaluation & Explainability
Run `notebooks/03_Evaluation_and_Explainability.ipynb` to evaluate the model on the test set and visualize Grad-CAM heatmaps.

### 4. Additional Note
Due to my limited time and poor time management, I failed to complete the assessment successfully. If I had more time, I would have polished more on:

#### Model Architecture & Training
- Experiment with more advanced architectures (ResNet50, EfficientNet, Vision Transformers)
- Implement learning rate scheduling and optimization techniques

#### Production Readiness
- Complete inference.py implementation (current version is not complete)

#### Documentation & Code Quality
- Complete configurations system (currently incomplete)
- Error handling and validation
- Better code organization and modularity


Despite these limitations, the current implementation provides a base foundation with a working end-to-end pipeline, transfer learning with ResNet18, stratified data splitting, early stopping, model checkpointing, TensorBoard logging, and Grad-CAM explainability features.