import numpy as np

from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from src.splitting import actor_disjoint_split
from src.visualizations import plot_confusion_matrix


def run_sequence_cnn(X_clean, X_augmented, y, actor_ids,random_state=7  ):
    """
    Train a 2D-CNN on log-mel spectrogram sequences.

    Test actors remain unseen during all training decisions.
    """
    train_idx, val_idx, test_idx = actor_disjoint_split(actor_ids,random_state=random_state)

    # Augmentation is used only in training.
    X_train = np.concatenate(
        (X_clean[train_idx], X_augmented[train_idx]),
        axis=0,
    )
    y_train = np.concatenate((y[train_idx], y[train_idx]))

    X_val, y_val = X_clean[val_idx], y[val_idx]
    X_test, y_test = X_clean[test_idx], y[test_idx]

    # Conv2D expects: samples, mel bins, time frames, channels.
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Helps compensate for unequal emotion-class counts.
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weights = dict(zip(classes, weights))

    model = Sequential([
        Input(shape=X_train.shape[1:]),

        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(8, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        to_categorical(y_train, num_classes=8),
        validation_data=(X_val, to_categorical(y_val, num_classes=8)),
        epochs=100,
        batch_size=16,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(
                monitor="val_accuracy",
                patience=12,
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

    probabilities = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    accuracy = float(np.mean(y_pred == y_test))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

    plot_confusion_matrix(
        y_test,
        y_pred,
        "2D-CNN log-mel spectrogram",
        "cnn_sequence_confusion_matrix",
    )

    print(f"\nActor-held-out test accuracy: {accuracy:.3f}")
    print(f"Actor-held-out macro F1: {macro_f1:.3f}")

    return model, {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }