#!/usr/bin/env python3
"""
Hyperparameter Optimization Script for Cotton-Weed Detection Challenge
Usage:
    python src/opt_tlc_ultralytics.py
"""

import gc
import os
import sys
import uuid

import click
import tlc
import torch
from dotenv import load_dotenv
from tlc_ultralytics import YOLO, Settings

try:
    import optuna
except ImportError:
    print("Please install optuna:")
    print("pip install optuna")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

# 3LC Table URLs
TRAIN_TABLE_URL = os.getenv("TRAIN_TABLE_URL")
VAL_TABLE_URL = os.getenv("VAL_TABLE_URL")

PROJECT_NAME = os.getenv("PROJECT_NAME")
EXPERIMENT_PREFIX = "optuna_tlc"
EPOCHS = 60  # Reduced epochs for optimization speed
IMAGE_SIZE = 640
DEVICE = 0
WORKERS = 4
NUM_TRIALS = 150  # Number of trials

# ============================================================================
# SETUP
# ============================================================================


def load_tables():
    if not TRAIN_TABLE_URL or not VAL_TABLE_URL:
        print("Error: TRAIN_TABLE_URL and VAL_TABLE_URL must be set in .env")
        sys.exit(1)

    print(f"Loading training table: {TRAIN_TABLE_URL}")
    train_table = tlc.Table.from_url(TRAIN_TABLE_URL)

    print(f"Loading validation table: {VAL_TABLE_URL}")
    val_table = tlc.Table.from_url(VAL_TABLE_URL)

    return {"train": train_table, "val": val_table}


# Load tables once globally to avoid reloading every trial
try:
    TABLES = load_tables()
except Exception as e:
    print(f"Failed to load tables: {e}")
    sys.exit(1)

# ============================================================================
# OPTIMIZATION FUNCTION
# ============================================================================


def create_objective(augment, geo_aug, photo_aug, complex_augs):
    def objective(trial):
        """
        Optuna objective function.
        """
        # Define search space
        lr0 = trial.suggest_float("lr0", 1e-4, 1e-2, log=True)
        lrf = trial.suggest_float("lrf", 0.01, 1.0)
        momentum = trial.suggest_float("momentum", 0.6, 0.98)
        weight_decay = trial.suggest_float(
            "weight_decay", 1e-5, 1e-3, log=True
        )
        warmup_epochs = trial.suggest_float("warmup_epochs", 0.0, 5.0)
        box = trial.suggest_float("box", 0.05, 10.0)
        cls = trial.suggest_float("cls", 0.5, 4.0)
        dfl = trial.suggest_float("dfl", 0.5, 3.0)

        # Optimizer & Scheduler
        copy_paste_mode = trial.suggest_categorical(
            "copy_paste_mode", ["flip", "mixup"]
        )

        # Augmentation Hyperparameters
        aug_params = {}
        if augment:
            if geo_aug:
                aug_params["degrees"] = trial.suggest_float(
                    "degrees", -45.0, 45.0
                )
                aug_params["scale"] = trial.suggest_float("scale", 0.0, 0.9)
                aug_params["translate"] = trial.suggest_float(
                    "translate", -1.0, 1.0
                )
                aug_params["mosaic"] = trial.suggest_float("mosaic", 0.0, 1.0)
                aug_params["fliplr"] = trial.suggest_float("fliplr", 0.0, 1.0)
                aug_params["flipud"] = trial.suggest_float("flipud", 0.0, 1.0)

            if photo_aug:
                aug_params["hsv_h"] = trial.suggest_float("hsv_h", 0.0, 0.1)
                aug_params["hsv_s"] = trial.suggest_float("hsv_s", 0.0, 0.9)
                aug_params["hsv_v"] = trial.suggest_float("hsv_v", 0.0, 0.9)

            if complex_augs:
                aug_params["mixup"] = trial.suggest_float("mixup", 0.0, 0.5)
                aug_params["copy_paste"] = trial.suggest_float(
                    "copy_paste", 0.0, 1.0
                )

        # Generate unique run name for 3LC tracking
        run_id = str(uuid.uuid4())[:8]
        run_name = f"{EXPERIMENT_PREFIX}_{run_id}"

        # Create 3LC Settings
        settings = Settings(
            project_name=PROJECT_NAME,
            run_name=run_name,
            run_description=f"Optuna trial {trial.number}",
            image_embeddings_dim=2,
        )

        # Initialize model
        model = YOLO("yolov8n.pt")

        # Define pruning callback
        def on_fit_epoch_end(trainer):
            # Get mAP50 from trainer metrics
            # Ultralytics stores metrics in trainer.metrics
            # Keys are usually 'metrics/mAP50(B)'
            metrics = trainer.metrics
            if metrics and "metrics/mAP50(B)" in metrics:
                current_map50 = metrics["metrics/mAP50(B)"]
                trial.report(current_map50, trainer.epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        print(f"Table URLS: {TABLES}")
        # Training arguments
        train_args = {
            "tables": TABLES,
            "name": run_name,
            "epochs": EPOCHS,
            "imgsz": IMAGE_SIZE,
            "device": DEVICE,
            "workers": WORKERS,
            "project": PROJECT_NAME,
            "exist_ok": True,
            "verbose": False,
            "settings": settings,
            # Hyperparameters
            "lr0": lr0,
            "lrf": lrf,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "box": box,
            "cls": cls,
            "dfl": dfl,
            "cos_lr": True,
            "copy_paste_mode": copy_paste_mode,
        }

        # Add augmentation params

        # Add augmentation params
        train_args.update(aug_params)

        map50 = 0.0

        # --------------------------------------------------------------------
        # MEMORY CLEANUP
        # --------------------------------------------------------------------
        try:
            results = model.train(**train_args)
            map50 = results.box.map50
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial failed with error: {e}")
            map50 = 0.0
        finally:
            # IMPORTANT: Clean up memory to prevent OOM on subsequent trials

            # 1. Delete the model reference
            if "model" in locals():
                del model

            # 2. Force Python Garbage Collection
            gc.collect()

            # 3. Empty PyTorch CUDA Cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return map50

    return objective


# ============================================================================
# MAIN EXECUTION
# ============================================================================


@click.command()
@click.option(
    "--augment", is_flag=True, help="Enable data augmentation tuning."
)
@click.option(
    "--geo-aug",
    is_flag=True,
    help="Tune geometric augmentations (mosaic, flip, scale, degrees).",
)
@click.option(
    "--photo-aug", is_flag=True, help="Tune photometric augmentations (HSV)."
)
@click.option(
    "--complex-augs",
    is_flag=True,
    help="Tune Mixup and Copy-Paste augmentations.",
)
def main(augment, geo_aug, photo_aug, complex_augs):
    print("=" * 70)
    print(
        "COTTON WEED DETECTION - HYPERPARAMETER OPTIMIZATION (OPTUNA + 3LC)"
    )
    print("=" * 70)

    if augment:
        print("Augmentation Tuning: ENABLED")
        print(f"  - Geometric: {'ON' if geo_aug else 'OFF'}")
        print(f"  - Photometric: {'ON' if photo_aug else 'OFF'}")
        print(
            f"  - Complex (Mixup/CopyPaste): {'ON' if complex_augs else 'OFF'}"
        )
    else:
        print("Augmentation Tuning: DISABLED (Using YOLO defaults)")

    # Create study
    # Optuna uses TPE (Tree-structured Parzen Estimator) by default.
    # We use SQLite storage to persist results.
    # Ensure temp directory exists
    os.makedirs("temp", exist_ok=True)
    storage_url = "sqlite:///temp/optuna.db"
    study_name = "cotton_weed_optimization"

    print(f"Study storage: {storage_url}")
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=5
        ),
    )

    print(f"Starting optimization with {NUM_TRIALS} trials...")

    # Create objective with captured arguments
    objective = create_objective(augment, geo_aug, photo_aug, complex_augs)

    study.optimize(objective, n_trials=NUM_TRIALS)

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)

    if len(study.trials) > 0:
        print(f"Best Result (mAP50): {study.best_value:.4f}")
        print("Best Config:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
        print("\nUpdate your train.py with these values!")
    else:
        print("No successful trials were completed.")


if __name__ == "__main__":
    main()
