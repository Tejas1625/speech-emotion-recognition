import os
import numpy as np

from src.data_loader import load_ravdess_data

# Keep downloaded audio separate from generated arrays.
RAW_DATA_DIR = os.path.join("data", "RAVDESS")
PROCESSED_DIR = os.path.join("data", "processed")

if __name__ == "__main__":
    # Makes random augmentation reproducible across runs.
    np.random.seed(42)

    if not os.path.exists(RAW_DATA_DIR):
        print(f"RAVDESS folder not found: {RAW_DATA_DIR}")
        raise SystemExit(1)

    X_clean, X_augmented, y, actor_ids = load_ravdess_data(RAW_DATA_DIR)

    # Creates the folder if it does not already exist.
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    np.save(os.path.join(PROCESSED_DIR, "X_clean.npy"), X_clean)
    np.save(os.path.join(PROCESSED_DIR, "X_augmented.npy"), X_augmented)
    np.save(os.path.join(PROCESSED_DIR, "y_labels.npy"), y)
    np.save(os.path.join(PROCESSED_DIR, "actor_ids.npy"), actor_ids)

    print(f"Saved {len(X_clean)} original clips and aligned augmentations.")