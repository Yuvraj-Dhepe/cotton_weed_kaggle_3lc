#!/usr/bin/env python3
"""
Hyperparameter Tuning Script using native Ultralytics Genetic Evolution.
Usage:
    python src/tune_ultralytics.py
"""

import os

import tlc
from dotenv import load_dotenv
from tlc_ultralytics import YOLO

load_dotenv()

EPOCHS = 100  # Number of training epochs
BATCH_SIZE = 16  # Batch size (reduce if GPU memory issues)
IMAGE_SIZE = 640  # Input image size (FIXED by competition)
DEVICE = 0  # GPU device (0 for first GPU, 'cpu' for CPU)
WORKERS = 4  # Number of dataloader workers
ITERATIONS = 300  # Number of trials

# Run configuration
PROJECT_NAME = "kaggle_cotton_weed_detection"  # 3LC project name
RUN_NAME = "yolov8n_baseline"  # Change for each experiment (e.g., "v2_with_augmentation")
RUN_DESCRIPTION = "Baseline YOLOv8n training run"  # Describe this experiment


def main():
    TRAIN_TABLE_URL = os.getenv("TRAIN_TABLE_URL")
    VAL_TABLE_URL = os.getenv("VAL_TABLE_URL")

    print("\n" + "=" * 70)
    print("Loading Tables")
    print("=" * 70)

    print(f"\n Training table: {TRAIN_TABLE_URL}")
    train_table = tlc.Table.from_url(TRAIN_TABLE_URL)
    print(f"   OK - Loaded: {len(train_table)} samples")

    print(f"\n Validation table: {VAL_TABLE_URL}")
    val_table = tlc.Table.from_url(VAL_TABLE_URL)
    print(f"   OK - Loaded: {len(val_table)} samples")

    # Initialize the YOLO model
    model = YOLO("yolov8n.pt")

    # Define custom search space for Genetic Algorithm
    # Format: 'hyperparameter': (min, max)
    search_space = {
        "lr0": (1e-5, 1e-1),
        "lrf": (0.01, 1.0),
        "momentum": (0.6, 0.98),
        "weight_decay": (0.0, 0.0001),
        "warmup_epochs": (0.0, 5.0),
        "warmup_momentum": (0.0, 0.95),
        "box": (0.02, 10),
        "cls": (0.1, 4.0),
        "dfl": (0.2, 6.0),
        # "hsv_h": (0.0, 0.1),
        # "hsv_s": (0.0, 0.9),
        # "hsv_v": (0.0, 0.9),
        "degrees": (0.0, 45.0),
        "translate": (0.0, 0.9),
        "scale": (0.0, 0.9),
        # "shear": (0.0, 10.0),
        # "perspective": (0.0, 0.001),
        "flipud": (0.0, 1.0),
        "fliplr": (0.0, 1.0),
        "mosaic": (0.0, 1.0),
        "mixup": (0.0, 1.0),
        "copy_paste": (0.0, 1.0),
    }
    train_args = {
        "data": "path/to/cotton_weed_dataset.yaml",  # Update with actual path
        "name": RUN_NAME,
        "epochs": EPOCHS,
        "imgsz": IMAGE_SIZE,
        "batch": BATCH_SIZE,
        "device": DEVICE,
        "workers": WORKERS,
        "val": True,
        "project": PROJECT_NAME,
        "cos_lr": True,
        "iterations": ITERATIONS,
        "plots": True,
        "save": False,
        "copy_paste_mode": "flip",
        "space": search_space,
        "patience": 30,
    }
    # Run hyperparameter tuning
    # This uses a Genetic Algorithm (GA) for hyperparameter evolution
    # Arguments:
    #   data: Path to dataset YAML
    #   epochs: Epochs per trial
    #   iterations: Number of trials (generations)
    #   optimizer: Optimizer to use
    #   plots: Save plots
    #   save: Save checkpoints
    #   val: Validate during training
    # Metric:
    #   The tuning process MAXIMIZES a "fitness" score.
    #   Fitness = 0.1 * mAP@50 + 0.9 * mAP@50-95
    #   It prioritizes high-quality detections (mAP@50-95).

    # Passing Training Arguments:
    # You can pass any argument accepted by model.train() directly to model.tune().
    # Example: copy_paste_mode="flip", cos_lr=True, etc.
    model.tune(**train_args)


if __name__ == "__main__":
    main()
