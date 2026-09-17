import json
import os

import numpy as np
import tensorflow as tf

from src.sequence_models import run_sequence_cnn

# Keeps model-training randomness comparable across actor-split experiments.
np.random.seed(42)
tf.random.set_seed(42)

DATA_DIR = os.path.join("data", "processed_sequence")

X_CLEAN_PATH = os.path.join(DATA_DIR, "X_clean_sequence.npy")
X_AUGMENTED_PATH = os.path.join(DATA_DIR, "X_augmented_sequence.npy")
Y_PATH = os.path.join(DATA_DIR, "y_labels.npy")
ACTORS_PATH = os.path.join(DATA_DIR, "actor_ids.npy")

MODEL_PATH = os.path.join("models", "cnn_sequence_ser.keras")
METRICS_PATH = os.path.join("models", "cnn_sequence_metrics.json")

if __name__ == "__main__":
    required_paths = (
        X_CLEAN_PATH,
        X_AUGMENTED_PATH,
        Y_PATH,
        ACTORS_PATH,
    )

    if not all(os.path.exists(path) for path in required_paths):
        print("Sequence data is missing. Run extract_sequence_data.py first.")
        raise SystemExit(1)

    # Change only random_state for split robustness checks: 42, 7, then 19.
    model, metrics = run_sequence_cnn(
        np.load(X_CLEAN_PATH),
        np.load(X_AUGMENTED_PATH),
        np.load(Y_PATH),
        np.load(ACTORS_PATH),
    )

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")