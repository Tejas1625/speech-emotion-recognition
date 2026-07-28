import os
import numpy as np
from src.feature_extraction import extract_features


def load_ravdess_data(data_path):
    features, labels = [], []

    print(f"\n--- SCANNING DIRECTORY: {data_path} ---")
    actor_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

    for actor_dir in actor_folders:
        if actor_dir.startswith('.'): continue

        actor_path = os.path.join(data_path, actor_dir)
        files = os.listdir(actor_path)

        for file in files:
            if not file.startswith('.') and file.lower().endswith(".wav"):
                file_path = os.path.join(actor_path, file)

                try:
                    parts = file.split('-')
                    if len(parts) < 3: continue

                    emotion_label = int(parts[2]) - 1

                    # 1. Extract and append the CLEAN audio
                    data_clean = extract_features(file_path, augment=False)
                    if data_clean is not None:
                        features.append(data_clean)
                        labels.append(emotion_label)

                    # 2. Extract and append the AUGMENTED audio
                    data_augmented = extract_features(file_path, augment=True)
                    if data_augmented is not None:
                        features.append(data_augmented)
                        labels.append(emotion_label)

                except ValueError:
                    pass  # Skip if filename is broken

    print(f"Extraction complete. Total samples loaded: {len(features)}")
    return np.array(features), np.array(labels)