import os

import librosa
import numpy as np

# Fixed audio format makes all spectrograms the same shape.
SAMPLE_RATE = 22050
DURATION_SECONDS = 3
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512


def augment_audio(audio):
    """Apply one light augmentation only to training copies."""
    if np.random.rand() < 0.5:
        noise = np.random.randn(len(audio))
        return audio + np.random.uniform(0.001, 0.004) * noise

    return librosa.effects.pitch_shift(
        y=audio,
        sr=SAMPLE_RATE,
        n_steps=np.random.uniform(-0.5, 0.5),
    )


def audio_to_log_mel(file_path, augment=False):
    """
    Convert one audio file into a 64 x time-frame log-mel spectrogram.

    Unlike averaged MFCC features, this preserves temporal emotion cues.
    """
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)

        # Pad short clips and trim long clips to a consistent duration.
        target_length = SAMPLE_RATE * DURATION_SECONDS
        audio = librosa.util.fix_length(audio, size=target_length)

        if augment:
            audio = augment_audio(audio)

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )

        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Per-sample normalization reduces loudness and recording variation.
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

        return log_mel.astype(np.float32)

    except Exception as error:
        print(f"Could not process {file_path}: {error}")
        return None


def load_ravdess_sequences(data_path):
    """
    Load clean and augmented spectrograms separately.

    Actor IDs are retained for actor-disjoint training, validation, and testing.
    """
    clean_sequences = []
    augmented_sequences = []
    labels = []
    actor_ids = []

    for actor_dir in sorted(os.listdir(data_path)):
        actor_path = os.path.join(data_path, actor_dir)

        if actor_dir.startswith(".") or not os.path.isdir(actor_path):
            continue

        for filename in sorted(os.listdir(actor_path)):
            if filename.startswith(".") or not filename.endswith(".wav"):
                continue

            try:
                # Third RAVDESS filename field is emotion, numbered 1 to 8.
                emotion_label = int(filename.split("-")[2]) - 1
            except (IndexError, ValueError):
                continue

            file_path = os.path.join(actor_path, filename)

            clean = audio_to_log_mel(file_path, augment=False)
            augmented = audio_to_log_mel(file_path, augment=True)

            if clean is None or augmented is None:
                continue

            clean_sequences.append(clean)
            augmented_sequences.append(augmented)
            labels.append(emotion_label)
            actor_ids.append(actor_dir)

    return (
        np.asarray(clean_sequences),
        np.asarray(augmented_sequences),
        np.asarray(labels),
        np.asarray(actor_ids),
    )