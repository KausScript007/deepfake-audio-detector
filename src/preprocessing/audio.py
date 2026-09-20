import subprocess

import librosa
import numpy as np


def load_audio(
    file_path,
    sample_rate=16000
):
    """
    Load an audio file as mono audio at the target sample rate.

    FFmpeg is used for decoding because some ASVspoof
    FLAC files are not reliably decoded by libsndfile.
    """

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(file_path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-"
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )

    audio = np.frombuffer(
        result.stdout,
        dtype=np.float32
    )

    audio, _ = librosa.effects.trim(audio)

    audio = librosa.util.normalize(audio)

    return audio, sample_rate


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