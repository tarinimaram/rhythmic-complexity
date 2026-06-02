"""
chunk_dual_analyzer.py — Chunked Dual-Channel Accuracy

Applies onset-deviation scoring across 4 fixed chunks.  Each chunk is
assigned human notes by timing (nearest reference onset), so an extra or
missing note in one chunk does not corrupt subsequent chunks.

Rules:
  - Chunk i gets the human notes whose nearest reference onset lives in
    chunk i (greedy nearest-neighbor IOI assignment).
  - If the human note count in a chunk != expected (chunk_size): the chunk
    is DISCARDED — scored as None, shown as gray in the plot.
  - Discarded chunks do not affect the evaluation of later chunks.
  - Within a valid chunk each human onset is matched to the nearest
    (unassigned) reference onset ("before" or "after") to determine its
    deviation and pass/fail.
"""

from __future__ import annotations

import numpy as np

_BEAT_CHUNK_SIZE = 4
_RHYTHM_CHUNK_SIZE = 7
_N_CHUNKS = 4


def _chunk_size_for(human_part: str) -> int:
    return _BEAT_CHUNK_SIZE if human_part == "beat" else _RHYTHM_CHUNK_SIZE




def _match_nearest_greedy(
    human_onsets: np.ndarray,
    ref_onsets: np.ndarray,
) -> list[tuple[int, int]]:
    """
    Greedy nearest-neighbor matching: iterate human onsets in order, each
    time assigning it to the nearest still-available reference onset.
    Returns list of (human_idx, ref_idx) pairs.
    Assumes len(human_onsets) == len(ref_onsets).
    """
    available = list(range(len(ref_onsets)))
    pairs: list[tuple[int, int]] = []
    for h_idx in range(len(human_onsets)):
        h = human_onsets[h_idx]
        best = min(available, key=lambda r: abs(h - ref_onsets[r]))
        pairs.append((h_idx, best))
        available.remove(best)
    return pairs


# ---------------------------------------------------------------------------
# Algorithm 1 — Fixed 4×N Chunks (timing-based assignment)
# ---------------------------------------------------------------------------

def fixed_dual_chunks(
    beat_iois: list | np.ndarray,
    rhythm_iois: list | np.ndarray,
    human_iois: list | np.ndarray,
    human_part: str,
    noise_sd: float,
    pass_threshold: float = 100.0,
    start_chunk: int = 0,
    human_start_ms: float = 0.0,
) -> dict:
    """
    Divide the reference into 4 non-overlapping chunks and evaluate each one.
    Human notes are assigned to chunks by timing (nearest reference onset),
    so an extra or missing note in one chunk does not shift note assignments
    for subsequent chunks.

    A chunk is DISCARDED (scored None) when the number of human notes
    assigned to it differs from the expected chunk_size.

    Returns
    -------
    dict:
        chunk_direct_accuracies     : list[float | None]   (None = discarded)
        chunk_relational_accuracies : list[float | None]
        best_chunk_index            : int  (0-based, best non-discarded)
        best_direct_accuracy        : float
        best_relational_accuracy    : float
        verdict                     : "PASS" | "FAIL"
        per_onset_relational        : list[dict]
        start_chunk                 : int
    """
    chunk_size = _chunk_size_for(human_part)
    beat_iois = np.asarray(beat_iois, dtype=float)
    rhythm_iois = np.asarray(rhythm_iois, dtype=float)
    human_iois = np.asarray(human_iois, dtype=float)

    # Tile reference to span all N_CHUNKS.
    full_beat_iois = np.tile(beat_iois, _N_CHUNKS)[: _N_CHUNKS * _BEAT_CHUNK_SIZE]
    full_rhythm_iois = np.tile(rhythm_iois, _N_CHUNKS)[: _N_CHUNKS * _RHYTHM_CHUNK_SIZE]
    ref_iois = full_beat_iois if human_part == "beat" else full_rhythm_iois

    # Absolute onset times.
    ref_onsets = np.concatenate([[0.0], np.cumsum(ref_iois)])
    human_onsets_abs = human_start_ms + np.concatenate([[0.0], np.cumsum(human_iois)])

    # Start onsets of each IOI (all but the very last onset point).
    ioi_starts = human_onsets_abs[:-1] if len(human_onsets_abs) > 1 else np.array([], dtype=float)

    tol = 2.0 * noise_sd

    direct_accs: list[float | None] = []
    relational_accs: list[float | None] = []
    all_per_onset: list[dict] = []

    for i in range(_N_CHUNKS):
        chunk_ref_start = ref_onsets[i * chunk_size]
        chunk_ref_end = ref_onsets[(i + 1) * chunk_size]

        if i < start_chunk:
            # Human was silent here.
            direct_accs.append(0.0)
            relational_accs.append(0.0)
            all_per_onset.extend([{"not_played": True}] * chunk_size)
            continue

        # Assign IOIs to this chunk by time window.
        # A note belongs to chunk i if its absolute onset is in
        #   [chunk_ref_start - tol, chunk_ref_end - tol)
        # The -tol shift ensures notes within the tolerance band of the
        # chunk boundary are assigned to the chunk they're closest to
        # rather than spilling into the adjacent chunk.
        in_window = np.where(
            (ioi_starts >= chunk_ref_start - tol) & (ioi_starts < chunk_ref_end - tol)
        )[0]
        actual_count = len(in_window)

        if actual_count != chunk_size:
            # Extra or missing note — discard this chunk entirely.
            direct_accs.append(None)
            relational_accs.append(None)
            # Fill per_onset with discarded markers so x-axis positions stay consistent.
            n_markers = max(actual_count, chunk_size)
            all_per_onset.extend([{"discarded": True}] * n_markers)
            continue

        # Reference start onsets for this chunk.
        r_onsets_chunk = ref_onsets[i * chunk_size : (i + 1) * chunk_size]
        h_onsets_chunk = ioi_starts[in_window]

        # Match each human onset to the nearest (unassigned) reference onset.
        pairs = _match_nearest_greedy(h_onsets_chunk, r_onsets_chunk)

        entries: list[dict] = []
        n_pass = 0
        for h_idx, r_idx in pairs:
            h = float(h_onsets_chunk[h_idx])
            r = float(r_onsets_chunk[r_idx])
            dev = h - r
            passed = abs(dev) <= tol
            if passed:
                n_pass += 1
            entries.append({
                "actual_offset_ms": h,
                "expected_offset_ms": r,
                "pass": passed,
            })

        accuracy = (n_pass / chunk_size) * 100.0
        direct_accs.append(accuracy)
        relational_accs.append(accuracy)
        all_per_onset.extend(entries)

    # Any human notes that fell beyond the last reference chunk boundary.
    last_ref_onset = ref_onsets[_N_CHUNKS * chunk_size]
    extra_ioi_indices = [j for j, t in enumerate(ioi_starts) if t >= last_ref_onset]
    if extra_ioi_indices:
        r_phrase = ref_onsets[:chunk_size]
        for j in extra_ioi_indices:
            h = float(ioi_starts[j])
            r = float(r_phrase[j % chunk_size])
            all_per_onset.append({
                "actual_offset_ms": h,
                "expected_offset_ms": r,
                "pass": abs(h - r) <= tol,
                "extra": True,
            })

    # Best chunk: highest accuracy among non-discarded, non-silent chunks.
    valid = [
        (i, d)
        for i, (d, r) in enumerate(zip(direct_accs, relational_accs))
        if d is not None and i >= start_chunk
    ]
    if not valid:
        best_idx = start_chunk
        best_d = best_r = 0.0
    else:
        best_idx, best_d = max(valid, key=lambda x: x[1])
        best_r = relational_accs[best_idx]  # type: ignore[assignment]

    verdict = "PASS" if (best_d is not None and best_d >= pass_threshold
                         and best_r is not None and best_r >= pass_threshold) else "FAIL"

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
# Algorithm 2 — Sliding Window (size N, step 1) — unchanged
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
    for rhythm.  The reference window is anchored at start_chunk so the human's
    notes are compared against the correct section of the reference.
    """
    from dual_channel_analyzer import check_direct_ioi, check_relational

    chunk_size = _chunk_size_for(human_part)
    beat_iois = np.asarray(beat_iois, dtype=float)
    rhythm_iois = np.asarray(rhythm_iois, dtype=float)
    human_iois = np.asarray(human_iois, dtype=float)
    ref_iois = beat_iois if human_part == "beat" else rhythm_iois

    ref_onsets = np.concatenate([[0.0], np.cumsum(ref_iois)])

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

        direct = check_direct_ioi(h_chunk, r_chunk, noise_sd)
        relational = check_relational(
            h_chunk, beat_iois, rhythm_iois, human_part, noise_sd,
            human_onset_offset_ms=onset_offset,
        )
        direct_accs.append(direct["accuracy_pct"])
        relational_accs.append(relational["accuracy_pct"])

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

    print("\n=== Fixed Chunks — Onset-Deviation Scoring ===")
    print(f"{'#':>4}  {'Direct%':>9}  {'Relational%':>12}  {'Avg%':>7}  {'':>10}")
    print("-" * 52)
    for i, (d, r) in enumerate(zip(direct_accs, rel_accs)):
        if i < start_chunk:
            print(f"{i+1:>4}  {'— NOT PLAYED —':>38}")
            continue
        if d is None:
            print(f"{i+1:>4}  {'— DISCARDED (extra/missing note) —':>38}")
            continue
        avg = (d + r) / 2.0
        marker = "← best" if i == best_idx else ""
        print(f"{i+1:>4}  {d:>8.1f}%  {r:>11.1f}%  {avg:>6.1f}%  {marker}")
    print("-" * 52)
    bd = result["best_direct_accuracy"]
    br = result["best_relational_accuracy"]
    print(f"Best Chunk {best_idx+1}: Direct={bd:.1f}%  Relational={br:.1f}%")
    print(f"Verdict: {result['verdict']}  (threshold: {pass_threshold:.0f}%)")
