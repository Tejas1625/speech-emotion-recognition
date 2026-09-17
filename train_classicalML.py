import json
import os

import joblib
import numpy as np

from src.classical_models import run_classical_models

# Generated feature arrays from extract_data.py
PROCESSED_DIR = os.path.join("data", "processed")

X_CLEAN_PATH = os.path.join(PROCESSED_DIR, "X_clean.npy")
X_AUGMENTED_PATH = os.path.join(PROCESSED_DIR, "X_augmented.npy")
Y_PATH = os.path.join(PROCESSED_DIR, "y_labels.npy")
ACTORS_PATH = os.path.join(PROCESSED_DIR, "actor_ids.npy")

# Local trained-model artifacts
MODEL_PATH = os.path.join("models", "classical_ser.pkl")
METRICS_PATH = os.path.join("models", "classical_metrics.json")

if __name__ == "__main__":
    np.random.seed(42)

    # Stop early if extraction has not been run.
    required_paths = (
        X_CLEAN_PATH,
        X_AUGMENTED_PATH,
        Y_PATH,
        ACTORS_PATH,
    )

    if not all(os.path.exists(path) for path in required_paths):
        print("Processed data is missing. Run extract_data.py first.")
        raise SystemExit(1)

    # Load clean features, training-only augmentations, labels, and speaker IDs.
    model, metrics = run_classical_models(
        np.load(X_CLEAN_PATH),
        np.load(X_AUGMENTED_PATH),
        np.load(Y_PATH),
        np.load(ACTORS_PATH),
    )

    # Create models/ automatically if it does not exist.
    os.makedirs("models", exist_ok=True)

    # Save the selected classical model and its exact evaluation metrics.
    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"\nSelected model: {metrics['selected_model']}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")