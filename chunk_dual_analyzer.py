"""
chunk_dual_analyzer.py — Chunked Dual-Channel Accuracy

Applies both direct IOI and cross-channel relational checks across
two chunking strategies: fixed 4×N chunks and a sliding window (size N, step 1),
where N=4 when the human performed the beat and N=7 for the rhythm.
"""

from __future__ import annotations

import numpy as np
from dual_channel_analyzer import check_direct_ioi, check_relational

_BEAT_CHUNK_SIZE = 4
_RHYTHM_CHUNK_SIZE = 7
_N_CHUNKS = 4


def _chunk_size_for(human_part: str) -> int:
    return _BEAT_CHUNK_SIZE if human_part == "beat" else _RHYTHM_CHUNK_SIZE


def _run_both_checks(
    human_chunk: np.ndarray,
    beat_iois: np.ndarray,
    rhythm_iois: np.ndarray,
    ref_chunk: np.ndarray,
    human_part: str,
    noise_sd: float,
    onset_offset_ms: float = 0.0,
) -> tuple[float, float, list[dict]]:
    """Return (direct_accuracy_pct, relational_accuracy_pct, per_onset) for one chunk."""
    direct = check_direct_ioi(human_chunk, ref_chunk, noise_sd)
    relational = check_relational(
        human_chunk, beat_iois, rhythm_iois, human_part, noise_sd,
        human_onset_offset_ms=onset_offset_ms,
    )
    return direct["accuracy_pct"], relational["accuracy_pct"], relational["per_onset"]


# ---------------------------------------------------------------------------
# Algorithm 1 — Fixed 4×7 Chunks
# ---------------------------------------------------------------------------

def fixed_dual_chunks(
    beat_iois: list | np.ndarray,
    rhythm_iois: list | np.ndarray,
    human_iois: list | np.ndarray,
    human_part: str,
    noise_sd: float,
    pass_threshold: float = 100.0,
    start_chunk: int = 0,
) -> dict:
    """
    Divide the reference into 4 non-overlapping chunks and run both checks on each.
    Chunk size is 4 for beat (4 beats per phrase) or 7 for rhythm.

    start_chunk indicates which reference chunk the human's first note belongs to.
    Chunks before start_chunk are scored 0% (human was silent); from start_chunk
    onward the human's notes are compared against the corresponding reference chunk.

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
    chunk_size = _chunk_size_for(human_part)
    beat_iois = np.asarray(beat_iois, dtype=float)
    rhythm_iois = np.asarray(rhythm_iois, dtype=float)
    human_iois = np.asarray(human_iois, dtype=float)

    # Tile each reference phrase to span all N_CHUNKS so chunks 1-3 have reference data.
    full_beat_iois = np.tile(beat_iois, _N_CHUNKS)[:_N_CHUNKS * _BEAT_CHUNK_SIZE]
    full_rhythm_iois = np.tile(rhythm_iois, _N_CHUNKS)[:_N_CHUNKS * _RHYTHM_CHUNK_SIZE]
    ref_iois = full_beat_iois if human_part == "beat" else full_rhythm_iois

    # Precompute absolute onset times for relational scoring.
    ref_onsets = np.concatenate([[0.0], np.cumsum(ref_iois)])

    direct_accs: list[float] = []
    relational_accs: list[float] = []
    all_per_onset: list[dict] = []

    for i in range(_N_CHUNKS):
        # Chunk 0 contributes chunk_size+1 entries (includes the sequence's first onset);
        # all later chunks contribute chunk_size (skip the boundary note shared with prev chunk).
        n_expected = chunk_size + (1 if i == 0 else 0)

        # Chunks before start_chunk: human was silent here.
        if i < start_chunk:
            direct_accs.append(0.0)
            relational_accs.append(0.0)
            all_per_onset.extend([{"not_played": True}] * n_expected)
            continue

        # Human chunk index within their actual played notes.
        human_chunk_idx = i - start_chunk
        h_start = human_chunk_idx * chunk_size
        h_chunk = human_iois[h_start : h_start + chunk_size]

        r_start = i * chunk_size
        r_chunk = ref_iois[r_start : r_start + chunk_size]

        if len(h_chunk) == 0 or len(r_chunk) == 0:
            direct_accs.append(0.0)
            relational_accs.append(0.0)
            all_per_onset.extend([{"not_played": True}] * n_expected)
            continue

        onset_offset = float(ref_onsets[r_start])
        d_acc, rel_acc, per_onset = _run_both_checks(
            h_chunk, full_beat_iois, full_rhythm_iois, r_chunk, human_part, noise_sd,
            onset_offset_ms=onset_offset,
        )
        direct_accs.append(d_acc)
        relational_accs.append(rel_acc)
        # Chunk 0: include onset[0] so the graph always starts at note 1.
        # Later chunks: skip onset[0] (shared boundary note with the previous chunk).
        entries: list[dict] = list(per_onset) if i == 0 else list(per_onset[1:])
        if len(entries) < n_expected:
            entries.extend([{"not_played": True}] * (n_expected - len(entries)))
        all_per_onset.extend(entries)

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
        "per_onset_relational": all_per_onset,
        "start_chunk": start_chunk,
    }


# ---------------------------------------------------------------------------
# Algorithm 2 — Sliding Window (size N, step 1)
# ---------------------------------------------------------------------------

def sliding_dual_window(
    beat_iois: list | np.ndarray,
    rhythm_iois: list | np.ndarray,
    human_iois: list | np.ndarray,
    human_part: str,
    noise_sd: float,
    pass_threshold: float = 100.0,
    start_chunk: int = 0,
) -> dict:
    """
    Slide an N-IOI window across the human array, where N=4 for beat and N=7
    for rhythm. The reference window is anchored at start_chunk so the human's
    notes are compared against the correct section of the reference.

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
    chunk_size = _chunk_size_for(human_part)
    beat_iois = np.asarray(beat_iois, dtype=float)
    rhythm_iois = np.asarray(rhythm_iois, dtype=float)
    human_iois = np.asarray(human_iois, dtype=float)
    ref_iois = beat_iois if human_part == "beat" else rhythm_iois

    ref_onsets = np.concatenate([[0.0], np.cumsum(ref_iois)])

    # Align the reference window to where the human started playing.
    ref_start_idx = start_chunk * chunk_size
    n_ref_available = max(0, len(ref_iois) - ref_start_idx)
    n = min(len(human_iois), n_ref_available)
    n_windows = max(0, n - chunk_size + 1)

    direct_accs: list[float] = []
    relational_accs: list[float] = []

    for i in range(n_windows):
        h_chunk = human_iois[i : i + chunk_size]
        ref_i = ref_start_idx + i
        r_chunk = ref_iois[ref_i : ref_i + chunk_size]
        onset_offset = float(ref_onsets[ref_i])

        d_acc, rel_acc, _ = _run_both_checks(
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


def print_chunk_dual_report(result: dict, pass_threshold: float) -> None:
    best_idx = result["best_chunk_index"]
    direct_accs = result["chunk_direct_accuracies"]
    rel_accs = result["chunk_relational_accuracies"]
    start_chunk = result.get("start_chunk", 0)

    print("\n=== Fixed Chunks — Dual-Channel ===")
    print(f"{'#':>4}  {'Direct%':>9}  {'Relational%':>12}  {'Avg%':>7}  {'':>6}")
    print("-" * 48)
    for i, (d, r) in enumerate(zip(direct_accs, rel_accs)):
        if i < start_chunk:
            print(f"{i+1:>4}  {'— NOT PLAYED —':>34}")
            continue
        avg = (d + r) / 2.0
        marker = "← best" if i == best_idx else ""
        print(f"{i+1:>4}  {d:>8.1f}%  {r:>11.1f}%  {avg:>6.1f}%  {marker}")
    print("-" * 48)
    print(f"Best Chunk {best_idx+1}: Direct={result['best_direct_accuracy']:.1f}%  "
          f"Relational={result['best_relational_accuracy']:.1f}%")
    print(f"Verdict: {result['verdict']}  (threshold: {pass_threshold:.0f}%)")
