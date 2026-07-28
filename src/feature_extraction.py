import librosa
import numpy as np


def add_noise(data, noise_factor=0.005):
    """Injects random white noise into the audio signal."""
    noise = np.random.randn(len(data))
    return data + noise_factor * noise


def pitch_shift(data, sampling_rate, n_steps=2.0):
    """Shifts the pitch of the audio up or down slightly."""
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=n_steps)


def extract_features(file_path, augment=False):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

        # Apply augmentation if the flag is True
        if augment:
            # Randomly choose between adding noise or shifting pitch
            choice = np.random.choice(['noise', 'pitch'])

            if choice == 'noise':
                # Randomize the amount of noise so it's different every time
                audio = add_noise(audio, noise_factor=np.random.uniform(0.001, 0.004))
            elif choice == 'pitch':
                # Randomly shift pitch between -2 and +2 semitones
                audio = pitch_shift(audio, sample_rate, n_steps=np.random.uniform(-0.5, 0.5))

        # Extract standard acoustic features
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)

        return np.hstack((mfccs, zcr, rms))

    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None