import os
import numpy as np

from src.sequence_features import load_ravdess_sequences

RAW_DATA_DIR = os.path.join("data", "RAVDESS")
OUTPUT_DIR = os.path.join("data", "processed_sequence")

if __name__ == "__main__":
    # Reproducible augmentation.
    np.random.seed(42)

    if not os.path.exists(RAW_DATA_DIR):
        print(f"RAVDESS folder not found: {RAW_DATA_DIR}")
        raise SystemExit(1)

    X_clean, X_augmented, y, actor_ids = load_ravdess_sequences(RAW_DATA_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(os.path.join(OUTPUT_DIR, "X_clean_sequence.npy"), X_clean)
    np.save(os.path.join(OUTPUT_DIR, "X_augmented_sequence.npy"), X_augmented)
    np.save(os.path.join(OUTPUT_DIR, "y_labels.npy"), y)
    np.save(os.path.join(OUTPUT_DIR, "actor_ids.npy"), actor_ids)

    print(f"Saved {len(X_clean)} clean and augmented log-mel spectrograms.")
    print(f"Sequence shape: {X_clean.shape}")