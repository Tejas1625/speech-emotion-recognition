import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

EMOTIONS = [
    "Neutral",
    "Calm",
    "Happy",
    "Sad",
    "Angry",
    "Fearful",
    "Disgust",
    "Surprised",
]


def plot_confusion_matrix(y_true, y_pred, model_name, filename):
    """
    Create a confusion matrix for either:
    - Classical ML: predicted integer class labels
    - CNN: predicted probability arrays for all eight classes
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Convert one-hot encoded true labels to integer labels if needed.
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)

    # Convert CNN probability distributions to integer predicted labels.
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Include every emotion label even if an actor split lacks one emotion.
    label_ids = np.arange(len(EMOTIONS))
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    # Avoid division-by-zero when an emotion is absent from the test split.
    row_totals = cm.sum(axis=1, keepdims=True)
    percentages = np.divide(
        cm,
        row_totals,
        out=np.zeros_like(cm, dtype=float),
        where=row_totals != 0,
    )

    annotations = np.array([
        f"{count}\n({percentage:.1%})"
        for count, percentage in zip(cm.flatten(), percentages.flatten())
    ]).reshape(cm.shape)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        xticklabels=EMOTIONS,
        yticklabels=EMOTIONS,
    )

    plt.title(f"Confusion Matrix - {model_name}", fontsize=16, pad=20)
    plt.xlabel("Predicted Emotion")
    plt.ylabel("True Emotion")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Supports both root-level files and future paths such as assets/file.png.
    output_path = f"{filename}.png"
    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved confusion matrix: {output_path}")