"""
channel_splitter.py — Stereo WAV Channel Splitter

Splits a stereo WAV into left (beat) and right (rhythm) mono files,
plots both waveforms for visual verification, and returns the arrays.
"""

from __future__ import annotations

import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


def split_channels(
    stereo_path: str,
    plot: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Split a stereo WAV into left (beat) and right (rhythm) mono WAV files.

    Parameters
    ----------
    stereo_path : str
        Path to the input stereo .wav file.
    plot : bool
        If True, display side-by-side waveform plots for visual verification.

    Returns
    -------
    beat_array : np.ndarray
        Left-channel audio samples.
    rhythm_array : np.ndarray
        Right-channel audio samples.
    sample_rate : int
        Sample rate of the recording in Hz.
    """
    data, sr = sf.read(stereo_path, always_2d=True)

    if data.shape[1] < 2:
        raise ValueError(f"'{stereo_path}' is not a stereo file (only 1 channel found).")

    beat_array = data[:, 0]
    rhythm_array = data[:, 1]

    base = os.path.splitext(stereo_path)[0]
    beat_path = f"{base}_beat.wav"
    rhythm_path = f"{base}_rhythm.wav"

    sf.write(beat_path, beat_array, sr)
    sf.write(rhythm_path, rhythm_array, sr)

    print(f"Saved beat channel  → {beat_path}")
    print(f"Saved rhythm channel → {rhythm_path}")

    if plot:
        _plot_channels(beat_array, rhythm_array, sr)

    return beat_array, rhythm_array, sr


def _plot_channels(
    beat_array: np.ndarray,
    rhythm_array: np.ndarray,
    sr: int,
) -> None:
    times_beat = np.arange(len(beat_array)) / sr
    times_rhythm = np.arange(len(rhythm_array)) / sr

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    axes[0].plot(times_beat, beat_array, color="steelblue", linewidth=0.5)
    axes[0].set_title("Left Channel — Beat")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(times_rhythm, rhythm_array, color="darkorange", linewidth=0.5)
    axes[1].set_title("Right Channel — Rhythm")
    axes[1].set_xlabel("Time (s)")

    fig.suptitle("Stereo Channel Split", fontsize=13)
    plt.tight_layout()
    plt.show()
