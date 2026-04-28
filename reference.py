"""
reference.py — Reference Sequence Builder

Builds thebeat objects from a reference .wav file and known IOIs.
"""

import numpy as np
from thebeat import SoundStimulus, Sequence


def build_reference(
    wav_path: str,
    iois_ms: list[float],
    noise_sd: float = 20.0,
) -> tuple[SoundStimulus, Sequence, np.ndarray, float]:
    """
    Build a reference Sequence from a .wav file and known inter-onset intervals.

    Parameters
    ----------
    wav_path : str
        Path to the reference .wav file.
    iois_ms : list[float]
        Known inter-onset intervals in milliseconds defining the ideal rhythm.
    noise_sd : float
        Standard deviation (ms) used as the tolerance window for IOI comparisons.
        A test beat is considered acceptable if its deviation from the reference
        IOI is within ±2 × noise_sd.

    Returns
    -------
    stimulus : SoundStimulus
        Loaded audio (used for waveform plotting).
    sequence : Sequence
        thebeat Sequence encoding the ideal timing (used for CCF and sequence plots).
    iois : np.ndarray
        The reference IOI array in milliseconds.
    noise_sd : float
        The tolerance value passed through for downstream use.
    """
    stimulus = SoundStimulus.from_wav(filepath=wav_path, name="reference")
    iois = np.array(iois_ms, dtype=float)
    sequence = Sequence(iois=iois)
    return stimulus, sequence, iois, noise_sd
