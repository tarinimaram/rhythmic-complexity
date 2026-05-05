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
    For each human onset, compare its actual offset from the partner channel's
    nearest onset against the expected offset from the reference.

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
        per_onset        : list[dict] — per-onset relational error detail
        accuracy_pct     : float
        pass             : bool
    """
    if human_part not in ("beat", "rhythm"):
        raise ValueError(f"human_part must be 'beat' or 'rhythm', got '{human_part}'")

    tol = _TOLERANCE_MULT * noise_sd

    human_onsets = _onsets_from_iois(human_iois) + human_onset_offset_ms
    beat_onsets = _onsets_from_iois(beat_iois)
    rhythm_onsets = _onsets_from_iois(rhythm_iois)

    if human_part == "beat":
        ref_same = beat_onsets      # reference onsets for the part the human played
        ref_partner = rhythm_onsets # partner channel the human should align against
    else:
        ref_same = rhythm_onsets
        ref_partner = beat_onsets

    n = min(len(human_onsets), len(ref_same))

    per_onset = []
    for i in range(n):
        h_onset = human_onsets[i]

        # Actual offset of this human onset from nearest partner onset
        nearest_partner_idx = _nearest_idx(h_onset, ref_partner)
        actual_offset = h_onset - ref_partner[nearest_partner_idx]

        # Expected offset: find closest reference same-part onset, then its offset
        # from *its* nearest partner onset
        nearest_ref_same_idx = _nearest_idx(h_onset, ref_same)
        ref_same_onset = ref_same[nearest_ref_same_idx]
        nearest_ref_partner_idx = _nearest_idx(ref_same_onset, ref_partner)
        expected_offset = ref_same_onset - ref_partner[nearest_ref_partner_idx]

        relational_error = abs(actual_offset - expected_offset)
        per_onset.append({
            "index": i,
            "human_onset_ms": float(h_onset),
            "actual_offset_ms": float(actual_offset),
            "expected_offset_ms": float(expected_offset),
            "relational_error_ms": float(relational_error),
            "tolerance_ms": float(tol),
            "pass": bool(relational_error <= tol),
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
    pass_threshold: float = 70.0,
) -> dict:
    """
    Run both accuracy checks and return a combined verdict.

    Parameters
    ----------
    beat_iois       : reference beat IOIs (ms), 28 values
    rhythm_iois     : reference rhythm IOIs (ms), 28 values
    human_iois      : human-performed IOIs (ms); trimmed to match reference length
    human_part      : "beat" or "rhythm"
    noise_sd        : tolerance SD (ms)
    pass_threshold  : minimum accuracy % required for each check to PASS

    Returns
    -------
    dict:
        direct_result      : output of check_direct_ioi
        relational_result  : output of check_relational
        direct_pass        : bool
        relational_pass    : bool
        overall_pass       : bool
    """
    beat_iois = np.asarray(beat_iois, dtype=float)
    rhythm_iois = np.asarray(rhythm_iois, dtype=float)
    human_iois = np.asarray(human_iois, dtype=float)

    ref_iois = beat_iois if human_part == "beat" else rhythm_iois
    n = min(len(human_iois), len(ref_iois))
    human_iois = human_iois[:n]

    direct = check_direct_ioi(human_iois, ref_iois, noise_sd)
    relational = check_relational(human_iois, beat_iois, rhythm_iois, human_part, noise_sd)

    direct_pass = direct["accuracy_pct"] >= pass_threshold
    relational_pass = relational["accuracy_pct"] >= pass_threshold

    return {
        "direct_result": direct,
        "relational_result": relational,
        "direct_pass": direct_pass,
        "relational_pass": relational_pass,
        "overall_pass": direct_pass and relational_pass,
    }


def print_dual_report(result: dict, noise_sd: float, pass_threshold: float) -> None:
    tol = _TOLERANCE_MULT * noise_sd
    dr = result["direct_result"]
    rr = result["relational_result"]

    print("\n" + "=" * 65)
    print("  DUAL-CHANNEL ANALYSIS REPORT")
    print("=" * 65)
    print(f"  Tolerance: ±{tol:.1f} ms  (noise_sd={noise_sd:.1f} ms × 2)")
    print(f"  Pass threshold: {pass_threshold:.0f}%")

    print("\n--- Check 1: Direct IOI Match ---")
    print(f"{'#':>4}  {'Human':>9}  {'Ref':>9}  {'Dev':>9}  {'Pass':>5}")
    print("-" * 45)
    for p in dr["per_ioi"]:
        flag = "PASS" if p["pass"] else "FAIL"
        print(f"{p['index']+1:>4}  {p['human_ioi_ms']:>8.1f}  {p['ref_ioi_ms']:>8.1f}  "
              f"{p['deviation_ms']:>8.1f}  {flag:>5}")
    print(f"  Accuracy: {dr['accuracy_pct']:.1f}%  →  {'PASS' if result['direct_pass'] else 'FAIL'}")

    print("\n--- Check 2: Cross-Channel Relational Accuracy ---")
    print(f"{'#':>4}  {'Human Onset':>12}  {'Actual Off':>11}  {'Expect Off':>11}  {'Error':>8}  {'Pass':>5}")
    print("-" * 60)
    for p in rr["per_onset"]:
        flag = "PASS" if p["pass"] else "FAIL"
        print(f"{p['index']+1:>4}  {p['human_onset_ms']:>11.1f}  {p['actual_offset_ms']:>10.1f}  "
              f"{p['expected_offset_ms']:>10.1f}  {p['relational_error_ms']:>7.1f}  {flag:>5}")
    print(f"  Accuracy: {rr['accuracy_pct']:.1f}%  →  {'PASS' if result['relational_pass'] else 'FAIL'}")

    print("\n" + "=" * 65)
    verdict = "PASS" if result["overall_pass"] else "FAIL"
    print(f"  OVERALL VERDICT: {verdict}")
    print("=" * 65 + "\n")
