"""
extractor.py — Onset Extractor for Human Audio

Uses librosa to detect note onsets in a human-performed WAV recording,
converts them to IOIs, and wraps the result in thebeat objects.
"""

import numpy as np
import librosa
from thebeat import SoundStimulus, Sequence


def extract_onsets(
    wav_path: str,
    hop_length: int = 512,
    backtrack: bool = True,
) -> tuple[SoundStimulus, Sequence, np.ndarray]:
    """
    Extract onset times from a human performance WAV and return thebeat objects.

    Parameters
    ----------
    wav_path : str
        Path to the human-performed .wav file.
    hop_length : int
        Hop length (samples) for the onset detection STFT. Smaller values give
        finer time resolution but are slower. Default 512 works well for most
        percussion/rhythm recordings.
    backtrack : bool
        If True, onset frames are moved back to the nearest preceding energy
        trough, which tends to give more accurate attack times.

    Returns
    -------
    stimulus : SoundStimulus
        Loaded audio (used for waveform plotting).
    sequence : Sequence
        thebeat Sequence built from the detected IOIs (used for CCF and sequence plots).
    iois_ms : np.ndarray
        Extracted inter-onset intervals in milliseconds.

    Raises
    ------
    ValueError
        If fewer than 2 onsets are detected (cannot compute any IOIs).
    """
    # Load with librosa for onset detection
    y, sr = librosa.load(wav_path, sr=None, mono=True)

    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        hop_length=hop_length,
        backtrack=backtrack,
        units="frames",
    )
    onset_times_sec = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    if len(onset_times_sec) < 2:
        raise ValueError(
            f"Only {len(onset_times_sec)} onset(s) detected in '{wav_path}'. "
            "Need at least 2 to compute IOIs. Check the recording or lower the "
            "onset detection threshold."
        )

    # Convert onset times (seconds) → IOIs (milliseconds)
    iois_ms = np.diff(onset_times_sec) * 1000.0
    #iois_ms = [398, 209, 197, 198, 205, 191, 411, 398, 209, 197, 198, 205, 191, 411, 398, 209, 197, 198, 205, 191, 411, 398, 209, 197, 198, 205, 191, 411]

    stimulus = SoundStimulus.from_wav(filepath=wav_path, name="test")
    sequence = Sequence(iois=iois_ms)

    return stimulus, sequence, iois_ms
