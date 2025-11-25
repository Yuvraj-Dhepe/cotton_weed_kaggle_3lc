#!/usr/bin/env python3
"""
Training Script for Cotton-Weed Detection Challenge

Simple, editable script for quick iteration - just modify the configuration
section below and run!

Usage:
    python train.py

Competition Rules:
    - YOLOv8n only (REQUIRED)
    - 640 input size (FIXED)
    - Hyperparameter tuning allowed
    - No ensembles

Learn more: See cotton_weed_starter_notebook.ipynb for explanations
"""

import os

import click
import tlc
import torch
from dotenv import load_dotenv
from tlc_ultralytics import YOLO, Settings

# ============================================================================
# CONFIGURATION - Edit these values for your training run
# ============================================================================

# Load environment variables from .env (if present)
load_dotenv()

# 3LC Table URLs (get these from Dashboard)
# Click your table -> Copy URL from browser or table info panel
# These values are read from environment first (via .env). If not set, fall back
# to the previous default local paths.
TRAIN_TABLE_URL = os.getenv("TRAIN_TABLE_URL")
VAL_TABLE_URL = os.getenv("VAL_TABLE_URL")

# Run configuration
PROJECT_NAME = "kaggle_cotton_weed_detection"  # 3LC project name
RUN_NAME = "yolov8n_baseline"  # Change for each experiment (e.g., "v2_with_augmentation")
RUN_DESCRIPTION = "Baseline YOLOv8n training run"  # Describe this experiment

# Training hyperparameters
EPOCHS = 100  # Number of training epochs
BATCH_SIZE = 16  # Batch size (reduce if GPU memory issues)
IMAGE_SIZE = 640  # Input image size (FIXED by competition)
DEVICE = 0  # GPU device (0 for first GPU, 'cpu' for CPU)
WORKERS = 4  # Number of dataloader workers

# Advanced hyperparameters (optional)
Momentum = 0.98
WEIGHT_DECAY = 0.0001
PATIENCE = 30  # Early stopping patience (epochs without improvement)

# Data augmentation (set to True to enable advanced augmentation)
# ============================================================================
# TRAINING PIPELINE - No need to edit below this line
# ============================================================================


@click.command()
@click.option("--augment", is_flag=True, help="Enable data augmentation.")
@click.option(
    "--geo-aug",
    is_flag=True,
    help="Enable geometric augmentations (mosaic, flip, scale, degrees).",
)
@click.option(
    "--photo-aug",
    is_flag=True,
    help="Enable photometric augmentations (HSV).",
)
@click.option("--mixup-aug", is_flag=True, help="Enable Mixup augmentation.")
def main(augment, geo_aug, photo_aug, mixup_aug):
    """Main training pipeline."""
    print("=" * 70)
    print("COTTON WEED DETECTION - TRAINING")
    print("=" * 70)

    # Check environment
    print("\nEnvironment:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  3LC: {tlc.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Validate table URLs
    if (
        not TRAIN_TABLE_URL
        or not VAL_TABLE_URL
        or "paste_your" in TRAIN_TABLE_URL
        or "paste_your" in VAL_TABLE_URL
    ):
        print(
            "\n !!! ERROR: Please set your table URLs in the .env or environment!"
        )
        print("\n How to get URLs:")
        print("   1. Open Dashboard: https://dashboard.3lc.ai")
        print("   2. Click on the tables tab")
        print("   3. Copy URL from table info panel to clipboard")
        print(
            "   4. Paste URLs into TRAIN_TABLE_URL and VAL_TABLE_URL in .env"
        )
        return

    # Load tables
    print("\n" + "=" * 70)
    print("Loading Tables")
    print("=" * 70)

    print(f"\n Training table: {TRAIN_TABLE_URL}")
    train_table = tlc.Table.from_url(TRAIN_TABLE_URL)
    print(f"   OK - Loaded: {len(train_table)} samples")

    print(f"\n Validation table: {VAL_TABLE_URL}")
    val_table = tlc.Table.from_url(VAL_TABLE_URL)
    print(f"   OK - Loaded: {len(val_table)} samples")

    tables = {"train": train_table, "val": val_table}

    # Configure training
    print("\n" + "=" * 70)
    print("Training Configuration")
    print("=" * 70)
    print(f"\n  Run: {RUN_NAME}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Device: {'GPU ' + str(DEVICE) if DEVICE != 'cpu' else 'CPU'}")

    if augment:
        print("  Augmentation: ENABLED")
        print(f"    - Geometric: {'ON' if geo_aug else 'OFF'}")
        print(f"    - Photometric: {'ON' if photo_aug else 'OFF'}")
        print(f"    - Mixup: {'ON' if mixup_aug else 'OFF'}")
    else:
        print("  Augmentation: DISABLED (Using YOLO defaults)")

    # Create 3LC Settings
    settings = Settings(
        project_name=PROJECT_NAME,
        run_name=RUN_NAME,
        run_description=RUN_DESCRIPTION,
        image_embeddings_dim=3,
        collection_epoch_interval=5,
    )

    # Load model
    print("\n Loading YOLOv8n...")
    model = YOLO("yolov8n.pt")
    print("   OK - Model loaded (3M parameters)")

    # Train
    print("\n" + "=" * 70)
    print("Training Started")
    print("=" * 70 + "\n")

    train_args = {
        "tables": tables,
        "name": RUN_NAME,
        "epochs": EPOCHS,
        "imgsz": IMAGE_SIZE,
        "batch": BATCH_SIZE,
        "device": DEVICE,
        "workers": WORKERS,
        "patience": PATIENCE,
        "settings": settings,
        "val": True,
        "project": PROJECT_NAME,
        "cos_lr": True,
        "momentum": Momentum,
        "weight_decay": WEIGHT_DECAY,
    }

    # # Add augmentation if enabled
    # if augment:
    #     if geo_aug:
    #         train_args.update(
    #             {
    #                 # "mosaic": MOSAIC,
    #                 "fliplr": FlipLR,
    #                 "flipud": FlipUD,
    #                 "translate": TRANSLATE,
    #                 "degrees": DEGREES,
    #                 "scale": SCALE,
    #             }
    #         )
    #     if photo_aug:
    #         # We don't want to use this will change the features of the weeds
    #         train_args.update(
    #             {
    #                 "hsv_h": HSV_H,
    #                 "hsv_s": HSV_S,
    #                 "hsv_v": HSV_V,
    #             }
    #         )
    #     if mixup_aug:
    #         train_args.update(
    #             {
    #                 # "mixup": MIXUP,
    #                 "mosaic": MOSAIC,
    #                 "copy_paste": COPY_PASTE,
    #                 "copy_paste_mode": COPY_PASTE_MODE,
    #                 "cutmix": CUTMIX,
    #             }
    #         )
    model.train(**train_args)

    # Done!
    print("\n" + "=" * 70)
    print("OK - TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n Weights saved: runs/detect/{RUN_NAME}/weights/best.pt")
    print("\n Next Steps:")
    print("   1. Check Dashboard: http://localhost:8000")
    print("   2. Analyze errors and edit data")
    print("   3. Generate predictions: python predict.py")
    print("   4. Retrain with edited data!")


if __name__ == "__main__":
    main()
