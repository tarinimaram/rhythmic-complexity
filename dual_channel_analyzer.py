"""
dual_channel_analyzer.py — Dual-Channel Accuracy Engine

Evaluates a human performance against a stereo reference with two independent checks:
  1. Direct IOI match — human IOIs vs. the same-part reference IOIs
  2. Cross-channel relational accuracy — human onset offsets vs. the partner channel
"""

from __future__ import annotations

import numpy as np


_TOLERANCE_MULT = 2.0


def _onsets_from_iois(iois_ms: np.ndarray) -> np.ndarray:
    """Absolute onset times (ms) from an IOI array (prepend 0, cumsum)."""
    return np.cumsum(np.concatenate([[0.0], iois_ms]))


def _nearest_idx(query: float, targets: np.ndarray) -> int:
    return int(np.argmin(np.abs(targets - query)))


# ---------------------------------------------------------------------------
# Check 1 — Direct IOI Match
# ---------------------------------------------------------------------------

def check_direct_ioi(
    human_iois: np.ndarray,
    ref_iois: np.ndarray,
    noise_sd: float,
) -> dict:
    """
    Compare human IOIs directly against the same-part reference IOIs.

    Returns
    -------
    dict:
        per_ioi          : list[dict] — per-interval correctness detail
        accuracy_pct     : float
        pass             : bool
    """
    tol = _TOLERANCE_MULT * noise_sd
    n = min(len(human_iois), len(ref_iois))
    human_trimmed = human_iois[:n]
    ref_trimmed = ref_iois[:n]

    per_ioi = []
    for i, (h, r) in enumerate(zip(human_trimmed, ref_trimmed)):
        deviation = abs(h - r)
        per_ioi.append({
            "index": i,
            "human_ioi_ms": float(h),
            "ref_ioi_ms": float(r),
            "deviation_ms": float(deviation),
            "tolerance_ms": float(tol),
            "pass": bool(deviation <= tol),
        })

    n_pass = sum(p["pass"] for p in per_ioi)
    accuracy_pct = (n_pass / n * 100.0) if n > 0 else 0.0

    return {
        "n_compared": n,
        "per_ioi": per_ioi,
        "accuracy_pct": accuracy_pct,
        "pass": n_pass == n,
    }


# ---------------------------------------------------------------------------
# Check 2 — Cross-Channel Relational Accuracy
# ---------------------------------------------------------------------------

def check_relational(
    human_iois: np.ndarray,
    beat_iois: np.ndarray,
    rhythm_iois: np.ndarray,
    human_part: str,
    noise_sd: float,
    human_onset_offset_ms: float = 0.0,
) -> dict:
    """
    For each human onset, compare its absolute timing against the corresponding
    reference onset at the same position in the sequence. The deviation is
    (human_onset - reference_onset): positive means the human was late, negative
    means early.

    human_onset_offset_ms positions the human's first note in the global
    reference timeline (e.g. ref_onsets[chunk_start]).

    Parameters
    ----------
    human_iois  : human IOI array (ms)
    beat_iois   : reference beat IOIs (ms)
    rhythm_iois : reference rhythm IOIs (ms)
    human_part  : "beat" or "rhythm"
    noise_sd    : tolerance SD (ms)

    Returns
    -------
    dict:
        per_onset        : list[dict] — per-onset deviation detail
        accuracy_pct     : float
        pass             : bool
    """
    if human_part not in ("beat", "rhythm"):
        raise ValueError(f"human_part must be 'beat' or 'rhythm', got '{human_part}'")

    tol = _TOLERANCE_MULT * noise_sd

    ref_iois = beat_iois if human_part == "beat" else rhythm_iois
    human_onsets = _onsets_from_iois(human_iois) + human_onset_offset_ms
    ref_onsets = _onsets_from_iois(ref_iois)

    # Anchor: find which reference onset the human's first note corresponds to.
    start_ref_idx = _nearest_idx(human_onsets[0], ref_onsets)
    n = min(len(human_onsets), len(ref_onsets) - start_ref_idx)

    per_onset = []
    for i in range(n):
        h_onset = human_onsets[i]
        r_onset = ref_onsets[start_ref_idx + i]
        signed_dev = h_onset - r_onset
        per_onset.append({
            "index": i,
            "human_onset_ms": float(h_onset),
            "actual_offset_ms": float(h_onset),
            "expected_offset_ms": float(r_onset),
            "relational_error_ms": float(abs(signed_dev)),
            "tolerance_ms": float(tol),
            "pass": bool(abs(signed_dev) <= tol),
        })

    n_pass = sum(p["pass"] for p in per_onset)
    accuracy_pct = (n_pass / n * 100.0) if n > 0 else 0.0

    return {
        "n_compared": n,
        "per_onset": per_onset,
        "accuracy_pct": accuracy_pct,
        "pass": n_pass == n,
    }


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def analyze_dual(
    beat_iois: list[float] | np.ndarray,
    rhythm_iois: list[float] | np.ndarray,
    human_iois: list[float] | np.ndarray,
    human_part: str,
    noise_sd: float,
    pass_threshold: float = 100.0,
    human_start_ms: float = 0.0,
) -> dict:
    """
    Evaluate each human onset against its matched reference onset.
    Deviation = human_onset_ms − matched_ref_onset_ms  (+ = late, − = early).

    Returns
    -------
    dict:
        per_onset    : list[dict] — per-note deviation detail
        accuracy_pct : float
        overall_pass : bool
    """
    beat_iois = np.asarray(beat_iois, dtype=float)
    rhythm_iois = np.asarray(rhythm_iois, dtype=float)
    human_iois = np.asarray(human_iois, dtype=float)

    ref_iois = beat_iois if human_part == "beat" else rhythm_iois
    tol = _TOLERANCE_MULT * noise_sd

    # IOI start onsets only (exclude the trailing endpoint).
    human_onsets = _onsets_from_iois(human_iois)[:-1] + human_start_ms
    ref_onsets = _onsets_from_iois(ref_iois)

    # Each human onset independently finds its nearest reference onset.
    # This keeps deviations accurate even when there is an extra or missing
    # note mid-sequence — adjacent notes are not dragged into wrong positions.
    per_onset = []
    for i, h in enumerate(human_onsets):
        nearest_ref_idx = int(np.argmin(np.abs(ref_onsets - h)))
        r = float(ref_onsets[nearest_ref_idx])
        dev = float(h) - r
        per_onset.append({
            "index": i,
            "human_onset_ms": float(h),
            "ref_onset_ms": r,
            "deviation_ms": dev,
            "tolerance_ms": float(tol),
            "pass": bool(abs(dev) <= tol),
        })

    n = len(per_onset)
    n_pass = sum(p["pass"] for p in per_onset)
    accuracy_pct = (n_pass / n * 100.0) if n > 0 else 0.0

    return {
        "per_onset": per_onset,
        "n_compared": n,
        "accuracy_pct": accuracy_pct,
        "overall_pass": accuracy_pct >= pass_threshold,
    }


def print_dual_report(result: dict, noise_sd: float, pass_threshold: float) -> None:
    tol = _TOLERANCE_MULT * noise_sd

    print("\n" + "=" * 65)
    print("  DUAL-CHANNEL ANALYSIS REPORT")
    print("=" * 65)
    print(f"  Tolerance: ±{tol:.1f} ms  (noise_sd={noise_sd:.1f} ms × 2)")
    print(f"  Pass threshold: {pass_threshold:.0f}%")

    print("\n--- Onset Deviation from Reference ---")
    print(f"{'#':>4}  {'Human (ms)':>12}  {'Ref (ms)':>10}  {'Dev (ms)':>10}  {'Pass':>5}")
    print("-" * 52)
    for p in result["per_onset"]:
        flag = "PASS" if p["pass"] else "FAIL"
        print(f"{p['index']+1:>4}  {p['human_onset_ms']:>11.1f}  {p['ref_onset_ms']:>9.1f}  "
              f"{p['deviation_ms']:>+10.1f}  {flag:>5}")
    print(f"  Accuracy: {result['accuracy_pct']:.1f}%  →  "
          f"{'PASS' if result['overall_pass'] else 'FAIL'}")

    print("\n" + "=" * 65)
    verdict = "PASS" if result["overall_pass"] else "FAIL"
    print(f"  OVERALL VERDICT: {verdict}")
    print("=" * 65 + "\n")
