import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import streamlit as st
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model

EMOTIONS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

st.set_page_config(page_title="SER Dashboard", page_icon="🎙️", layout="wide")
st.title("🎙️ Speech Emotion Recognition (SER)")
st.markdown("An Acoustic Pattern Classification System evaluating Classical ML vs. Deep Learning.")


# --- Load Models & Scaler ---
@st.cache_resource
def load_saved_models():
    try:
        ml = joblib.load(os.path.join('models', 'classical_ser.pkl'))
        cnn = load_model(os.path.join('models', 'cnn_ser.h5'))
        cnn_scaler = joblib.load(os.path.join('models', 'cnn_scaler.pkl'))  # LOAD SCALER
        return ml, cnn, cnn_scaler
    except Exception as e:
        st.error(f"Models/Scaler not found. Please run the training scripts first. Details: {e}")
        return None, None, None


ml_model, cnn_model, cnn_scaler = load_saved_models()

tab1, tab2 = st.tabs(["🔴 Live Prediction Engine", "📊 Model Analytics"])

with tab1:
    st.subheader("Upload Audio for Inference")

    col1, col2 = st.columns([1, 2])
    with col1:
        model_choice = st.radio("Select Engine:", ("1D-CNN (Deep Learning)", "Tuned Classical ML"))
        uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

    with col2:
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')

            with st.spinner("Extracting & Scaling Features..."):
                try:
                    audio, sample_rate = librosa.load(uploaded_file, res_type='kaiser_fast')
                    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
                    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
                    rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)

                    features = np.hstack((mfccs, zcr, rms))
                    features = features.reshape(1, -1)
                    extraction_success = True
                except Exception as e:
                    st.error(f"Processing failed: {e}")
                    extraction_success = False

            if extraction_success and ml_model and cnn_model:
                st.markdown("### Prediction Results")

                if model_choice == "Tuned Classical ML":
                    # ML Pipeline scales the data automatically!
                    pred_idx = ml_model.predict(features)[0]
                    predicted_emotion = EMOTIONS[pred_idx]
                    st.success(f"**Detected Emotion:** {predicted_emotion}")

                else:
                    # CRITICAL FIX: Scale the CNN features manually
                    features_scaled = cnn_scaler.transform(features)
                    features_reshaped = np.expand_dims(features_scaled, axis=2)

                    predictions = cnn_model.predict(features_reshaped)[0]
                    pred_idx = np.argmax(predictions)
                    predicted_emotion = EMOTIONS[pred_idx]

                    st.success(f"**Detected Emotion:** {predicted_emotion}")
                    st.write("**Network Confidence Distribution:**")
                    prob_dict = {EMOTIONS[i]: float(predictions[i]) for i in range(len(EMOTIONS))}
                    st.bar_chart(prob_dict)

with tab2:
    st.subheader("System Performance & Architecture")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### 1D-Convolutional Neural Network")
        st.metric(label="Validation Accuracy", value="~81.0%")
    with col4:
        st.markdown("#### Tuned Classical ML Engine")
        st.metric(label="Validation Accuracy", value="~79.0%")