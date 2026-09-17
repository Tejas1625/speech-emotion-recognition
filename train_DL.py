import json
import os

import joblib
import numpy as np
import tensorflow as tf

from src.deep_models import run_cnn_model

# Reproducible initialization.
np.random.seed(42)
tf.random.set_seed(42)

PROCESSED_DIR = os.path.join("data", "processed")

X_CLEAN_PATH = os.path.join(PROCESSED_DIR, "X_clean.npy")
X_AUGMENTED_PATH = os.path.join(PROCESSED_DIR, "X_augmented.npy")
Y_PATH = os.path.join(PROCESSED_DIR, "y_labels.npy")
ACTORS_PATH = os.path.join(PROCESSED_DIR, "actor_ids.npy")

CNN_MODEL_PATH = os.path.join("models", "cnn_clean_only_ser.h5")
SCALER_PATH = os.path.join("models", "cnn_clean_only_scaler.pkl")
METRICS_PATH = os.path.join("models", "cnn_clean_only_metrics.json")

# First experiment: test whether augmentation is reducing generalization.
USE_AUGMENTATION = False

if __name__ == "__main__":
    required_paths = (
        X_CLEAN_PATH,
        X_AUGMENTED_PATH,
        Y_PATH,
        ACTORS_PATH,
    )

    if not all(os.path.exists(path) for path in required_paths):
        print("Processed data is missing. Run extract_data.py first.")
        raise SystemExit(1)

    model, scaler, metrics = run_cnn_model(
        np.load(X_CLEAN_PATH),
        np.load(X_AUGMENTED_PATH),
        np.load(Y_PATH),
        np.load(ACTORS_PATH),
        use_augmentation=USE_AUGMENTATION,
    )

    os.makedirs("models", exist_ok=True)

    model.save(CNN_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"\nSaved model: {CNN_MODEL_PATH}")
    print(f"Saved scaler: {SCALER_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")