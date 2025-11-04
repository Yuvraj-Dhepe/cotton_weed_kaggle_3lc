from pathlib import Path

import tlc

# Define constants for 3LC registration
PROJECT_NAME = "kaggle_cotton_weed_detection"
DATASET_NAME = "cotton_weed_det3"
WORK_DIR = Path("./data")
DATASET_YAML = WORK_DIR / "dataset.yaml"

print("=" * 70)
print("DATA REGISTRATION")
print("=" * 70)

# ============================================================================
# IDEMPOTENCY CHECK - Safe to run multiple times
# ============================================================================
try:
    # Check if tables already exist
    existing_train = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name=f"{DATASET_NAME}-train1",
    )
    existing_val = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name=f"{DATASET_NAME}-val1",
    )

    print("\n⚠️  Tables already exist!")
    print(f" Training: {len(existing_train)} samples")
    print(f" Validation: {len(existing_val)} samples")
    print("\n✅ Using existing tables (no duplicates created)")
    print(" This cell is safe to run multiple times!")

    # Set variables for compatibility
    train_table = existing_train
    val_table = existing_val

except Exception:
    # Tables don't exist, create them
    print("\n✅ No existing tables - creating new ones...")

    # Create training table
    print("\n Creating training table...")
    train_table = tlc.Table.from_yolo(
        dataset_yaml_file=str(DATASET_YAML),
        split="train",
        task="detect",
        dataset_name=DATASET_NAME,
        project_name=PROJECT_NAME,
        table_name=f"{DATASET_NAME}-train1",
    )

    # Create validation table
    print(" Creating validation table...")
    val_table = tlc.Table.from_yolo(
        dataset_yaml_file=str(DATASET_YAML),
        split="val",
        task="detect",
        dataset_name=DATASET_NAME,
        project_name=PROJECT_NAME,
        table_name=f"{DATASET_NAME}-val1",
    )

# Display registration results
print("\n✅ Tables created successfully!")
print("=" * 70)
print("\n Training Table:")
print(f"   Samples: {len(train_table)}")
print(f"   URL: {train_table.url}")

print("\n Validation Table:")
print(f"   Samples: {len(val_table)}")
print(f"   URL: {val_table.url}")

print("\n" + "=" * 70)
print("✅ Phase 1 Complete: Dataset Registered with 3LC!")
print("=" * 70)

print("\n Next Steps:")
print("  (Optional) Explore tables in Dashboard: https://dashboard.3lc.ai")
