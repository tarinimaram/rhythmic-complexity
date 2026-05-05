"""
chunk_dual_analyzer.py — Chunked Dual-Channel Accuracy

Applies both direct IOI and cross-channel relational checks across
two chunking strategies: fixed 4×7 chunks and a sliding window (size 7, step 1).
"""

from __future__ import annotations

import numpy as np
from dual_channel_analyzer import check_direct_ioi, check_relational

_CHUNK_SIZE = 7
_N_CHUNKS = 4


def _run_both_checks(
    human_chunk: np.ndarray,
    beat_iois: np.ndarray,
    rhythm_iois: np.ndarray,
    ref_chunk: np.ndarray,
    human_part: str,
    noise_sd: float,
    onset_offset_ms: float = 0.0,
) -> tuple[float, float]:
    """Return (direct_accuracy_pct, relational_accuracy_pct) for one chunk."""
    direct = check_direct_ioi(human_chunk, ref_chunk, noise_sd)
    relational = check_relational(
        human_chunk, beat_iois, rhythm_iois, human_part, noise_sd,
        human_onset_offset_ms=onset_offset_ms,
    )
    return direct["accuracy_pct"], relational["accuracy_pct"]


# ---------------------------------------------------------------------------
# Algorithm 1 — Fixed 4×7 Chunks
# ---------------------------------------------------------------------------

def fixed_dual_chunks(
    beat_iois: list | np.ndarray,
    rhythm_iois: list | np.ndarray,
    human_iois: list | np.ndarray,
    human_part: str,
    noise_sd: float,
    pass_threshold: float = 70.0,
) -> dict:
    """
    Divide the 28 IOIs into 4 non-overlapping 7-IOI chunks and run both checks
    on each chunk. The best chunk (highest average of the two accuracy scores)
    determines the overall verdict.

    Returns
    -------
    dict:
        chunk_direct_accuracies     : list[float]
        chunk_relational_accuracies : list[float]
        best_chunk_index            : int  (0-based)
        best_direct_accuracy        : float
        best_relational_accuracy    : float
        verdict                     : "PASS" | "FAIL"
    """
    beat_iois = np.asarray(beat_iois, dtype=float)
    rhythm_iois = np.asarray(rhythm_iois, dtype=float)
    human_iois = np.asarray(human_iois, dtype=float)
    ref_iois = beat_iois if human_part == "beat" else rhythm_iois

    n = min(len(human_iois), len(ref_iois))
    human_iois = human_iois[:n]
    ref_iois = ref_iois[:n]

    # Precompute absolute onset times so each chunk's human onsets are placed
    # at the correct position in the full reference timeline for relational scoring.
    ref_onsets = np.concatenate([[0.0], np.cumsum(ref_iois)])

    direct_accs: list[float] = []
    relational_accs: list[float] = []

    for i in range(_N_CHUNKS):
        start = i * _CHUNK_SIZE
        end = start + _CHUNK_SIZE
        h_chunk = human_iois[start:end]
        r_chunk = ref_iois[start:end]

        if len(h_chunk) == 0:
            direct_accs.append(0.0)
            relational_accs.append(0.0)
            continue

        onset_offset = float(ref_onsets[start])
        d_acc, rel_acc = _run_both_checks(
            h_chunk, beat_iois, rhythm_iois, r_chunk, human_part, noise_sd,
            onset_offset_ms=onset_offset,
        )
        direct_accs.append(d_acc)
        relational_accs.append(rel_acc)

    avg_scores = [(d + r) / 2.0 for d, r in zip(direct_accs, relational_accs)]
    best_idx = int(np.argmax(avg_scores))

    best_d = direct_accs[best_idx]
    best_r = relational_accs[best_idx]
    verdict = "PASS" if (best_d >= pass_threshold and best_r >= pass_threshold) else "FAIL"

    return {
        "chunk_direct_accuracies": direct_accs,
        "chunk_relational_accuracies": relational_accs,
        "best_chunk_index": best_idx,
        "best_direct_accuracy": best_d,
        "best_relational_accuracy": best_r,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Algorithm 2 — Sliding Window (size 7, step 1)
# ---------------------------------------------------------------------------

def sliding_dual_window(
    beat_iois: list | np.ndarray,
    rhythm_iois: list | np.ndarray,
    human_iois: list | np.ndarray,
    human_part: str,
    noise_sd: float,
    pass_threshold: float = 70.0,
) -> dict:
    """
    Slide a 7-IOI window across the human array. For each position, compare
    against the aligned reference window (same start index). The window with
    the highest average accuracy determines the overall verdict.

    Returns
    -------
    dict:
        chunk_direct_accuracies     : list[float]  (one per window)
        chunk_relational_accuracies : list[float]
        best_chunk_index            : int  (start index of best window)
        best_direct_accuracy        : float
        best_relational_accuracy    : float
        verdict                     : "PASS" | "FAIL"
    """
    beat_iois = np.asarray(beat_iois, dtype=float)
    rhythm_iois = np.asarray(rhythm_iois, dtype=float)
    human_iois = np.asarray(human_iois, dtype=float)
    ref_iois = beat_iois if human_part == "beat" else rhythm_iois

    n = min(len(human_iois), len(ref_iois))
    human_iois = human_iois[:n]
    ref_iois = ref_iois[:n]

    ref_onsets = np.concatenate([[0.0], np.cumsum(ref_iois)])
    n_windows = max(0, n - _CHUNK_SIZE + 1)

    direct_accs: list[float] = []
    relational_accs: list[float] = []

    for i in range(n_windows):
        h_chunk = human_iois[i : i + _CHUNK_SIZE]
        r_chunk = ref_iois[i : i + _CHUNK_SIZE]
        onset_offset = float(ref_onsets[i])

        d_acc, rel_acc = _run_both_checks(
            h_chunk, beat_iois, rhythm_iois, r_chunk, human_part, noise_sd,
            onset_offset_ms=onset_offset,
        )
        direct_accs.append(d_acc)
        relational_accs.append(rel_acc)

    if not direct_accs:
        return {
            "chunk_direct_accuracies": [],
            "chunk_relational_accuracies": [],
            "best_chunk_index": 0,
            "best_direct_accuracy": 0.0,
            "best_relational_accuracy": 0.0,
            "verdict": "FAIL",
        }

    avg_scores = [(d + r) / 2.0 for d, r in zip(direct_accs, relational_accs)]
    best_idx = int(np.argmax(avg_scores))

    best_d = direct_accs[best_idx]
    best_r = relational_accs[best_idx]
    verdict = "PASS" if (best_d >= pass_threshold and best_r >= pass_threshold) else "FAIL"

    return {
        "chunk_direct_accuracies": direct_accs,
        "chunk_relational_accuracies": relational_accs,
        "best_chunk_index": best_idx,
        "best_direct_accuracy": best_d,
        "best_relational_accuracy": best_r,
        "verdict": verdict,
    }


def print_chunk_dual_report(result: dict, algorithm: str, pass_threshold: float) -> None:
    best_idx = result["best_chunk_index"]
    direct_accs = result["chunk_direct_accuracies"]
    rel_accs = result["chunk_relational_accuracies"]
    label = "Chunk" if algorithm == "fixed" else "Window"

    print(f"\n=== {'Fixed Chunks' if algorithm == 'fixed' else 'Sliding Window'} — Dual-Channel ===")
    print(f"{'#':>4}  {'Direct%':>9}  {'Relational%':>12}  {'Avg%':>7}  {'Best':>6}")
    print("-" * 48)
    for i, (d, r) in enumerate(zip(direct_accs, rel_accs)):
        avg = (d + r) / 2.0
        marker = "← best" if i == best_idx else ""
        print(f"{i+1:>4}  {d:>8.1f}%  {r:>11.1f}%  {avg:>6.1f}%  {marker}")
    print("-" * 48)
    print(f"Best {label} {best_idx+1}: Direct={result['best_direct_accuracy']:.1f}%  "
          f"Relational={result['best_relational_accuracy']:.1f}%")
    print(f"Verdict: {result['verdict']}  (threshold: {pass_threshold:.0f}%)")
