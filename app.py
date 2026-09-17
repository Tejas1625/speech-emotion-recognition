import io
import os

import librosa
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


# These values must match the preprocessing used in sequence_features.py.
SAMPLE_RATE = 22_050
DURATION_SECONDS = 3
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

MODEL_PATH = os.path.join("models", "cnn_sequence_ser.keras")

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


st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
)


@st.cache_resource
def load_saved_model():
    """
    Loads the final 2D CNN once and reuses it across Streamlit reruns.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            "Run train_sequence_cnn.py first."
        )

    return load_model(MODEL_PATH, compile=False)


def audio_to_log_mel(file_bytes):
    """
    Converts an uploaded WAV file into the same normalized log-Mel
    spectrogram representation used during model training.

    Output shape before the channel dimension: (64, time_frames)
    """
    audio, _ = librosa.load(
        io.BytesIO(file_bytes),
        sr=SAMPLE_RATE,
        mono=True,
    )

    # Make every uploaded clip the same duration as training samples.
    target_length = SAMPLE_RATE * DURATION_SECONDS
    audio = librosa.util.fix_length(audio, size=target_length)

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Per-sample normalization, matching the sequence-data pipeline.
    mean = np.mean(log_mel)
    standard_deviation = np.std(log_mel)

    log_mel = (log_mel - mean) / (standard_deviation + 1e-8)

    return log_mel.astype(np.float32)


def predict_emotion(model, file_bytes):
    """
    Preprocesses uploaded audio and returns the predicted emotion
    plus probability for every class.
    """
    log_mel = audio_to_log_mel(file_bytes)

    # 2D CNN input shape:
    # (batch_size, mel_bands, time_frames, channels)
    model_input = log_mel[np.newaxis, ..., np.newaxis]

    probabilities = model.predict(model_input, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    return EMOTIONS[predicted_index], probabilities


def main():
    st.title("🎙️ Speech Emotion Recognition")
    st.caption(
        "8-class emotion classification using a 2D CNN trained on "
        "log-Mel spectrograms."
    )

    st.info(
        "Educational demonstration only. This model predicts acoustic "
        "emotion patterns and is not a clinical, psychological, or "
        "mental-health assessment tool."
    )

    try:
        model = load_saved_model()
    except Exception as error:
        st.error(f"Unable to load the trained model: {error}")
        st.stop()

    prediction_tab, evaluation_tab = st.tabs(
        ["🎧 Predict Emotion", "📊 Model Evaluation"]
    )

    with prediction_tab:
        st.subheader("Upload WAV Audio")

        uploaded_file = st.file_uploader(
            "Choose a WAV file",
            type=["wav"],
            help="For best results, upload spoken audio.",
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()

            st.audio(file_bytes, format="audio/wav")

            if st.button("Predict Emotion", type="primary"):
                try:
                    with st.spinner("Extracting log-Mel features and predicting..."):
                        predicted_emotion, probabilities = predict_emotion(
                            model,
                            file_bytes,
                        )

                    st.success(
                        f"Detected emotion: **{predicted_emotion}**"
                    )

                    probability_frame = pd.DataFrame(
                        {
                            "Emotion": EMOTIONS,
                            "Probability": probabilities,
                        }
                    ).set_index("Emotion")

                    st.subheader("Prediction Confidence")
                    st.bar_chart(probability_frame)

                    top_probability = float(np.max(probabilities))
                    st.caption(
                        f"Top predicted-class probability: "
                        f"{top_probability:.1%}"
                    )

                except Exception as error:
                    st.error(
                        "Audio processing failed. Ensure the uploaded file "
                        f"is a valid WAV file. Details: {error}"
                    )

    with evaluation_tab:
        st.subheader("Leakage-Safe Evaluation")

        st.markdown(
            """
            The model was evaluated using **actor-held-out splits**:

            - Audio from a test actor was never used during training.
            - This avoids speaker leakage from random clip-level splits.
            - Results are averaged across three different actor splits.
            """
        )

        metric_column_1, metric_column_2, metric_column_3 = st.columns(3)

        metric_column_1.metric(
            "Average Accuracy",
            "55.0%",
        )

        metric_column_2.metric(
            "Average Macro-F1",
            "53.2%",
        )

        metric_column_3.metric(
            "Evaluation",
            "Actor-held-out",
        )

        st.markdown("#### Split-wise Results")

        results = pd.DataFrame(
            {
                "Actor Split Seed": [42, 7, 19],
                "Accuracy": ["53.3%", "63.0%", "48.7%"],
                "Macro-F1": ["51.5%", "61.5%", "46.6%"],
            }
        )

        st.dataframe(
            results,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Variation across splits is expected because RAVDESS has a "
            "limited number of speakers. The average is reported rather "
            "than the best individual split."
        )


if __name__ == "__main__":
    main()