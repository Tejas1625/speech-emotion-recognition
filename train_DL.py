import os
import joblib # Added joblib to save the scaler
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
from src.deep_models import run_cnn_model

X_PATH = os.path.join('data', 'X_features.npy')
Y_PATH = os.path.join('data', 'y_labels.npy')
MODEL_SAVE_PATH = os.path.join('models', 'cnn_ser.h5')
SCALER_SAVE_PATH = os.path.join('models', 'cnn_scaler.pkl') # New path

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    print("--- TRAINING DEEP LEARNING ---")

    if os.path.exists(X_PATH) and os.path.exists(Y_PATH):
        X = np.load(X_PATH)
        y = np.load(Y_PATH)
        print(f"Data loaded instantly! Shape: {X.shape}")

        # Catch both the model and the scaler
        model, scaler = run_cnn_model(X, y)

        # Save model AND scaler
        model.save(MODEL_SAVE_PATH)
        joblib.dump(scaler, SCALER_SAVE_PATH)
        print(f"Successfully saved model to {MODEL_SAVE_PATH}")
        print(f"Successfully saved scaler to {SCALER_SAVE_PATH}")
    else:
        print("Error: .npy files not found. Run extract_data.py first.")