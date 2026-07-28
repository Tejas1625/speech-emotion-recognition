import os
import numpy as np

# Prevent OpenMP library collisions on Mac
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from src.data_loader import load_ravdess_data

# Point to your raw audio folder
DATA_DIR = os.path.join("data", "RAVDESS")

if __name__ == "__main__":
    print("--- EXTRACTING AUDIO FEATURES ---")
    print("This takes a few minutes, but you only do it ONCE!")

    if os.path.exists(DATA_DIR):
        X, y = load_ravdess_data(DATA_DIR)

        # Save the raw arrays directly into the data folder
        np.save(os.path.join('data', 'X_features.npy'), X)
        np.save(os.path.join('data', 'y_labels.npy'), y)

        print(f"Extraction Complete! Saved {len(X)} samples into the 'data' folder.")
    else:
        print(f"Error: Data folder not found at {DATA_DIR}")