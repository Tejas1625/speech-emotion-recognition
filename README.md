# Speech Emotion Recognition (SER)

## Project Overview
**SER** is a deep learning-based audio processing system designed to analyze raw human speech and classify it into one of eight distinct emotional states (e.g., Happy, Sad, Angry, Fearful). 

Unlike traditional image-based spectrogram models, this project focuses on **hardware efficiency** and **raw signal processing**:
1.  **Decoupled Architecture:** Utilizes a highly optimized 3-block 1D-Convolutional Neural Network (1D-CNN) that processes 1D audio arrays directly, significantly reducing computational overhead.
2.  **Acoustic Feature Extraction:** Mathematically transforms raw audio waves into robust numerical representations using MFCCs, Zero-Crossing Rate (ZCR), and RMS energy.

## Key Features
- **Dynamic Data Augmentation:** Automatically injects white noise and applies pitch shifting to the audio data during preprocessing to prevent model overfitting.
- **High-Accuracy Classification:** Outperforms classical machine learning baselines (SVM, Random Forests) by achieving ~81% validation accuracy through gradient descent optimization.
- **Multiclass Emotion Detection:** Successfully maps complex audio signals to 8 distinct emotional categories.
- **End-to-End Pipeline:** Includes standalone scripts for dataset loading, feature extraction, and model training.
- **Interactive Web Interface:** Deployed a user-friendly frontend using Streamlit (via app.py) allowing users to upload .wav files and see real-time emotion predictions and probability distributions.

## Tech Stack
- **Backend:** Python
- **Frontend/UI:** Streamlit
- **Deep Learning:** TensorFlow, Keras (1D-CNN)
- **Machine Learning:** Scikit-learn (Baselines, Validation)
- **Audio Processing:** Librosa
- **Data Manipulation:** NumPy, Pandas

## Dataset
The model was trained on the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset, a validated collection of emotional speech audio files.

- **Format:** `.wav` files
- **Labels:** *Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised*.

### How to Get the Data
Due to GitHub's file size limits for media, the raw audio dataset must be downloaded externally.
- **Download:** [RAVDESS Emotional Speech Audio (Kaggle)](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- **Setup:** Extract the downloaded archive and move all the `Actor_XX` folders into the `data/raw/` directory of this project.

## Project Structure
```text
speech-emotion-recognition/
│
├── .gitignore                # Tells Git to skip .venv, data, and models
├── app.py                    # Main Flask application / API
├── extract_data.py           # Script for dataset extraction
├── train_DL.py               # Deep Learning (1D-CNN) training script
├── train_classicalML.py      # Classical ML (SVM, Random Forest) training script
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── src/                      # Core modules
│   ├── classical_models.py   # Baseline ML model definitions
│   ├── deep_models.py        # 1D-CNN architecture definition
│   ├── data_loader.py        # Data loading & preprocessing utilities
│   ├── feature_extraction.py # MFCCs, ZCR, and RMS energy extraction
│   └── visualizations.py     # Plotting functions for confusion matrices
│
├── assets/                   # Evaluation plots and images
│   ├── cml_confusion_matrix.png
│   └── cnn_confusion_matrix.png
│
├── data/                     # Local data storage (ignored by git)
└── models/                   # Saved model weights (ignored by git)
```
## How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Tejas1625/speech-emotion-recognition.git
cd speech-emotion-recognition
```
### 2. Install Dependencies 
- Ensure you have Python 3.8+ installed
```bash
pip install -r requirements.txt
```
### 3. Set up Data
- Download the RAVDESS dataset from Kaggle.
- Extract and place the Actor folders directly into the `data/` directory.

### 4. Extract data and features
- Run the extraction script to process the raw .wav files into numerical arrays (MFCCs).
```bash
python extract_data.py
```
- **Result:** This will populate the `data/` folder with .npy feature files.

### 5. Train the Models
- Train the classical ML models and 1D-CNN architecture on the processed features.
```bash
python train_classicalML.py
python train_DL.py
```
- **Result:** This trains the network, outputs the validation accuracy metrics, and saves the final architecture to `models/`

### 6. Start the Application
Launch the Streamlit frontend to interact with the trained model:
```Bash
streamlit run app.py
```
- **Open Browser:** This will automatically open a local web server (usually at http://localhost:8501).
- **Test It:** Upload a .wav file through the UI to see the model predict the emotion in real-time.
