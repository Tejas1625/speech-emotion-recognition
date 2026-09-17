import numpy as np

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from src.splitting import actor_disjoint_split
from src.visualizations import plot_confusion_matrix


def run_cnn_model(
    X_clean,
    X_augmented,
    y,
    actor_ids,
    use_augmentation=True,
):
    """
    Train a 1D-CNN with actor-disjoint evaluation.

    use_augmentation=False runs the clean-only experiment.
    """

    # No actor appears in more than one split.
    train_idx, val_idx, test_idx = actor_disjoint_split(actor_ids)

    # Use augmentation only in training, never validation/test.
    if use_augmentation:
        X_train = np.vstack((X_clean[train_idx], X_augmented[train_idx]))
        y_train = np.concatenate((y[train_idx], y[train_idx]))
        experiment_name = "CNN with augmentation"
    else:
        X_train = X_clean[train_idx]
        y_train = y[train_idx]
        experiment_name = "CNN clean-only"

    X_val = X_clean[val_idx]
    y_val = y[val_idx]

    X_test = X_clean[test_idx]
    y_test = y[test_idx]

    # Fit the scaler only using training samples.
    scaler = StandardScaler()

    X_train = np.expand_dims(scaler.fit_transform(X_train), axis=2)
    X_val = np.expand_dims(scaler.transform(X_val), axis=2)
    X_test = np.expand_dims(scaler.transform(X_test), axis=2)

    model = Sequential([
        # Explicit Input layer avoids the Keras input_shape warning.
        Input(shape=(X_train.shape[1], 1)),

        Conv1D(64, 3, activation="relu"),
        MaxPooling1D(2),
        Dropout(0.2),

        Conv1D(128, 3, activation="relu"),
        MaxPooling1D(2),
        Dropout(0.2),

        Conv1D(256, 3, activation="relu"),
        MaxPooling1D(2),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(8, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\n--- Training: {experiment_name} ---")

    model.fit(
        X_train,
        to_categorical(y_train, num_classes=8),
        # Validation set controls early stopping.
        validation_data=(X_val, to_categorical(y_val, num_classes=8)),
        epochs=200,
        batch_size=16,
        callbacks=[
            EarlyStopping(
                monitor="val_accuracy",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1,
            ),
        ],
        verbose=1,
    )

    # Final evaluation occurs once on unseen actors.
    probabilities = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    accuracy = float(np.mean(y_pred == y_test))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

    plot_confusion_matrix(
        y_test,
        y_pred,
        experiment_name,
        "cnn_confusion_matrix",
    )

    print(f"\nActor-held-out test accuracy: {accuracy:.3f}")
    print(f"Actor-held-out macro F1: {macro_f1:.3f}")

    return model, scaler, {
        "experiment": experiment_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }