import librosa
import numpy as np


def load_audio(
    file_path,
    sample_rate=16000
):
    """
    Load an audio file as mono audio at the target sample rate.
    """
    audio, sr = librosa.load(
        file_path,
        sr=sample_rate,
        mono=True
    )
    audio, _ = librosa.effects.trim(audio)
    audio = librosa.util.normalize(audio)   #different audios can have different volume levels
    return audio, sr


def segment_audio(
    audio,
    sample_rate=16000,
    segment_duration=4
):
    """
    Split audio into fixed-length segments.

    Short final segments are zero-padded.
    """

    segment_length = sample_rate * segment_duration

    segments = []

    for start in range(0, len(audio), segment_length):
        segment = audio[start:start + segment_length]

        if len(segment) < segment_length:
            segment = np.pad(
                segment,
                (0, segment_length - len(segment))
            )

        segments.append(segment)

    return np.array(segments)


def extract_mel_spectrogram(
    audio,
    sample_rate=16000,
    n_fft=2048,
    hop_length=512,
    n_mels=128
):
    """
    Convert an audio segment into a Mel-spectrogram in dB.
    """

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db