"""
src/preprocessing/mfcc.py

MFCC feature extraction module for Branch B.
Extracts MFCC features from 1D audio waveform arrays and formats them
as temporal sequences (time_frames, n_mfcc) for recurrent models (LSTM/GRU).
"""

import librosa
import numpy as np


def extract_mfcc(
    audio: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Extracts MFCC features from a 1D audio waveform.

    Parameters:
    -----------
    audio : np.ndarray
        1D floating point audio signal (e.g. 64000 samples for 4s at 16kHz).
    sr : int
        Sample rate in Hz (default: 16000).
    n_mfcc : int
        Number of MFCC coefficients to retain (default: 40).
    n_fft : int
        FFT window size (default: 2048).
    hop_length : int
        Number of audio samples between adjacent frames (default: 512).

    Returns:
    --------
    mfcc_seq : np.ndarray
        2D float32 array of shape (time_frames, n_mfcc), e.g. (126, 40).
    """
    if audio.ndim != 1:
        raise ValueError(f"Expected 1D audio signal, got shape {audio.shape}")

    # librosa.feature.mfcc returns shape (n_mfcc, time_frames)
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    # Transpose to (time_frames, n_mfcc) so each row is a time step
    mfcc_seq = mfcc.T.astype(np.float32)

    return mfcc_seq