import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# RAVDESS official emotion labels in order
EMOTIONS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']


def plot_confusion_matrix(y_true, y_pred, model_name, filename):
    """
    Generates and saves a high-quality confusion matrix heatmap.
    """
    # If y is one-hot encoded (like from your CNN), convert it back to single digits
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    # Calculate percentages for better readability
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    # Create a heatmap with both counts and percentages
    labels = [f"{v1}\n({v2:.1%})" for v1, v2 in zip(cm.flatten(), cm_percentage.flatten())]
    labels = np.asarray(labels).reshape(cm.shape)

    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)

    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, pad=20)
    plt.ylabel('True Emotion', fontsize=12)
    plt.xlabel('Predicted Emotion', fontsize=12)

    # Rotate x-axis labels so they don't overlap
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the image to your project folder
    plt.savefig(f"{filename}.png", dpi=300)
    print(f"Saved confusion matrix as {filename}.png")
    plt.close()