import os
import numpy as np

from src.feature_extraction import extract_features


def load_ravdess_data(data_path):
    """
    Extract clean and augmented features separately.

    Augmented samples are not mixed into the dataset immediately.
    Training scripts add them only to the training split, avoiding
    original-versus-augmented audio leakage.
    """
    clean_features = []
    augmented_features = []
    labels = []
    actor_ids = []

    # Each Actor_XX folder represents one speaker.
    for actor_dir in sorted(os.listdir(data_path)):
        actor_path = os.path.join(data_path, actor_dir)

        if actor_dir.startswith(".") or not os.path.isdir(actor_path):
            continue

        for filename in sorted(os.listdir(actor_path)):
            if filename.startswith(".") or not filename.lower().endswith(".wav"):
                continue

            try:
                # RAVDESS filename format: modality-vocal-channel-emotion-...
                # Emotion is the third part and is 1-indexed in the filename.
                emotion_label = int(filename.split("-")[2]) - 1
            except (IndexError, ValueError):
                print(f"Skipping malformed filename: {filename}")
                continue

            file_path = os.path.join(actor_path, filename)

            # Generate aligned clean and augmented representations.
            clean = extract_features(file_path, augment=False)
            augmented = extract_features(file_path, augment=True)

            # Keep only complete pairs.
            if clean is None or augmented is None:
                continue

            clean_features.append(clean)
            augmented_features.append(augmented)
            labels.append(emotion_label)
            actor_ids.append(actor_dir)

    return (
        np.asarray(clean_features),
        np.asarray(augmented_features),
        np.asarray(labels),
        np.asarray(actor_ids),
    )