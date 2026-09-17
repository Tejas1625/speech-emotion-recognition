# Speech Emotion Recognition

An end-to-end machine-learning application that classifies spoken WAV audio into eight emotion classes using a 2D Convolutional Neural Network (CNN) trained on log-Mel spectrograms.

## Project Overview

Speech Emotion Recognition (SER) identifies emotional patterns in spoken audio. This project converts uploaded speech into log-Mel spectrograms and uses a 2D CNN to predict one of eight emotion categories.

The project includes:

- A reproducible feature-extraction and training pipeline.
- Actor-held-out evaluation for reliable unseen-speaker testing.
- A Streamlit interface for interactive WAV-file predictions.
- Probability visualization across all emotion classes.

## Emotion Classes

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

## Key Features

- **Log-Mel Feature Extraction:** Converts raw speech audio into normalized 64-band log-Mel spectrograms.
- **2D CNN Classification:** Learns time-frequency emotion patterns directly from spectrograms.
- **Actor-Held-Out Evaluation:** Ensures speakers in the test set are not seen during training, preventing speaker leakage.
- **Multi-Split Validation:** Reports average performance across three actor-disjoint splits.
- **Interactive Streamlit Interface:** Allows users to upload WAV files and view predictions with confidence scores.
- **Reproducible Pipeline:** Separates raw data, processed NumPy arrays, saved models, training code, and inference code.

## Evaluation Methodology

A random clip-level train/test split can inflate results because recordings from the same speaker may appear in both training and testing sets.

To avoid this, the final model uses actor-held-out evaluation:

- Training, validation, and test sets contain different actors.
- Test actors are completely unseen during model training.
- Performance is evaluated using accuracy and macro-F1.
- Results are averaged across three actor-split seeds.

## Final Results

| Actor Split Seed | Accuracy | Macro-F1 |
|---:|---:|---:|
| 42 | 53.3% | 51.5% |
| 7 | 63.0% | 61.5% |
| 19 | 48.7% | 46.6% |
| **Average** | **55.0%** | **53.2%** |

The average is reported instead of the best split because RAVDESS has a limited number of speakers, and results naturally vary depending on the held-out actors.

## Model Pipeline

```text
WAV Audio
   ↓
Resample to 22,050 Hz and fix duration to 3 seconds
   ↓
64-band Log-Mel Spectrogram
   ↓
2D CNN Blocks
Conv2D → Batch Normalization → Max Pooling → Dropout
   ↓
Global Average Pooling
   ↓
Dense Softmax Classifier
   ↓
8 Emotion Classes
```

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Audio Processing | Librosa |
| Data Processing | NumPy, Pandas |
| Evaluation | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Interactive Interface | Streamlit |

## Project Structure

```text
speech-emotion-recognition/
│
├── .gitignore
├── README.md
├── requirements.txt
├── app.py                         # Streamlit inference interface
├── extract_sequence_data.py       # Creates log-Mel spectrogram arrays
├── train_sequence_cnn.py          # Trains the final 2D CNN
│
├── src/
│   ├── sequence_features.py       # Audio loading and log-Mel extraction
│   ├── sequence_models.py         # 2D CNN architecture and training logic
│   ├── splitting.py               # Actor-held-out splitting logic
│   └── visualizations.py          # Confusion-matrix visualization
│
├── data/                          # Local only; ignored by Git
│   ├──|
│   │  └── RAVDESS/
│   └── processed_sequence/
│
└── models/                        # Local only; ignored by Git
    └── cnn_sequence_ser.keras
```

## Dataset

This project uses the speech portion of the RAVDESS dataset:

> Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).

Download the dataset here:

[RAVDESS Emotional Speech Audio Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

After downloading, place the `Actor_XX` folders under:

```text
data/raw/RAVDESS/
```

The raw dataset is excluded from this repository because of file-size constraints.

## Installation

```bash
git clone https://github.com/Tejas1625/speech-emotion-recognition.git
cd speech-emotion-recognition
pip install -r requirements.txt
```

## Run Locally

### 1. Extract log-Mel spectrogram features

```bash
python extract_sequence_data.py
```

### 2. Train the 2D CNN

```bash
python train_sequence_cnn.py
```

### 3. Launch the Streamlit interface

```bash
streamlit run app.py
```

Upload a `.wav` speech file to view the predicted emotion and class-probability distribution.

## Notes

- This is an educational machine-learning project.
- It predicts acoustic emotion patterns from speech and is not a clinical or mental-health assessment tool.
- The dataset, generated features, and trained model are excluded from GitHub through `.gitignore`.