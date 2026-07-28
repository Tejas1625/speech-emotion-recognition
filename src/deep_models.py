from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from src.visualizations import plot_confusion_matrix


def run_cnn_model(X, y):
    # Fit the scaler and transform data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_reshaped = np.expand_dims(X_scaled, axis=2)
    y_cat = to_categorical(y, num_classes=8)

    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_cat, test_size=0.2, random_state=42)

    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(2),
        Dropout(0.2),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.2),
        Conv1D(256, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(8, activation='softmax')
    ])

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

    print("\n--- Training 1D-CNN with Anti-Overfitting Callbacks ---")
    history = model.fit(X_train, y_train, epochs=200, batch_size=16,
                        validation_data=(X_test, y_test), callbacks=[early_stop, reduce_lr], verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nCNN Final Test Accuracy: {accuracy:.3f}")

    cnn_preds = model.predict(X_test)
    plot_confusion_matrix(y_test, cnn_preds, "1D-CNN", "cnn_confusion_matrix")

    # CRITICAL FIX: Return the scaler so it can be saved!
    return model, scaler