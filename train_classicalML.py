import os
import joblib
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from src.classical_models import run_classical_models
from src.visualizations import plot_confusion_matrix

# Paths
X_PATH = os.path.join('data', 'X_features.npy')
Y_PATH = os.path.join('data', 'y_labels.npy')
MODEL_SAVE_PATH = os.path.join('models', 'classical_ser.pkl')

if __name__ == "__main__":
    np.random.seed(42)
    print("--- TRAINING CLASSICAL ML ---")

    if os.path.exists(X_PATH) and os.path.exists(Y_PATH):
        X = np.load(X_PATH)
        y = np.load(Y_PATH)
        print(f"Data loaded instantly! Shape: {X.shape}")

        model = run_classical_models(X, y)

        # Save into your models folder
        joblib.dump(model, MODEL_SAVE_PATH)
        print(f"Successfully saved to {MODEL_SAVE_PATH}")
    else:
        print("Error: .npy files not found. Run extract_data.py first.")