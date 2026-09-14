import librosa
import numpy as np


def extract_mel_spectrogram(
    file_path,
    sample_rate=16000,
    n_fft=2048,
    hop_length=512,
    n_mels=128
):
    """
    Load an audio file and extract its Mel-spectrogram.

    Returns:
        mel_db: Mel-spectrogram in decibels
        sample_rate: Sampling rate used for processing
    """

    audio, sr = librosa.load(
        file_path,
        sr=sample_rate,
        mono=True
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db, sr
