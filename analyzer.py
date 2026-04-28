"""
analyzer.py — Comparison Engine

Compares a reference rhythm sequence against a human-performed test sequence
using per-beat IOI deviation checks and cross-correlation (CCF).

Also provides chunk-based accuracy algorithms for evaluating a 28-IOI (4×7)
repeating rhythmic phrase: fixed phrase-boundary chunks and a sliding window.
"""

from __future__ import annotations

import numpy as np
from thebeat import Sequence
from thebeat.stats import ccf_df

_CHUNK_SIZE = 7
_N_CHUNKS = 4


def analyze(
    ref_sequence: Sequence,
    ref_iois: np.ndarray,
    noise_sd: float,
    test_sequence: Sequence,
    test_iois: np.ndarray,
    ccf_threshold: float = 0.85,
    tolerance_multiplier: float = 2.0,
    ccf_resolution: float = 10.0,
) -> dict:
    """
    Compare a test rhythm against a reference and return a structured verdict.

    The comparison has two independent checks that must both pass:

    1. **IOI deviation check** — each aligned IOI pair (reference vs. test) must
       fall within ±(tolerance_multiplier × noise_sd) ms. Sequences of unequal
       length are trimmed to the shorter one.

    2. **CCF check** — the peak value of the cross-correlation function between
       the two sequences must meet or exceed `ccf_threshold`.

    Parameters
    ----------
    ref_sequence : Sequence
        thebeat Sequence representing the ideal reference rhythm.
    ref_iois : np.ndarray
        Reference IOI array in milliseconds.
    noise_sd : float
        Standard deviation (ms) of acceptable IOI deviation.
    test_sequence : Sequence
        thebeat Sequence built from the human performance.
    test_iois : np.ndarray
        Extracted IOI array from the human performance in milliseconds.
    ccf_threshold : float
        Minimum peak CCF value required to pass the CCF check (0–1). Default 0.85.
    tolerance_multiplier : float
        Number of standard deviations defining the pass/fail window. Default 2.0.
    ccf_resolution : float
        Bin width in ms for the CCF histogram. Default 10.0.

    Returns
    -------
    result : dict
        {
            "n_beats_compared": int,
            "per_beat": [
                {
                    "beat": int,           # 1-indexed
                    "ref_ioi_ms": float,
                    "test_ioi_ms": float,
                    "deviation_ms": float,
                    "tolerance_ms": float, # tolerance_multiplier × noise_sd
                    "pass": bool,
                }
            ],
            "ioi_pass_rate": float,        # fraction of beats that passed
            "ioi_check_pass": bool,        # True if ALL beats passed
            "ccf_peak": float,
            "ccf_check_pass": bool,
            "overall_pass": bool,
        }
    """
    # --- IOI deviation check ---
    n = min(len(ref_iois), len(test_iois))
    ref_trimmed = ref_iois[:n]
    test_trimmed = test_iois[:n]
    tolerance = tolerance_multiplier * noise_sd

    per_beat = []
    for i, (r, t) in enumerate(zip(ref_trimmed, test_trimmed)):
        deviation = abs(t - r)
        per_beat.append(
            {
                "beat": i + 1,
                "ref_ioi_ms": float(r),
                "test_ioi_ms": float(t),
                "deviation_ms": float(deviation),
                "tolerance_ms": float(tolerance),
                "pass": bool(deviation <= tolerance),
            }
        )

    beats_passed = sum(b["pass"] for b in per_beat)
    ioi_pass_rate = beats_passed / n if n > 0 else 0.0
    ioi_check_pass = beats_passed == n

    # --- CCF check ---
    ccf_result = ccf_df(test_sequence, ref_sequence, resolution=ccf_resolution)
    # ccf_df returns a DataFrame with columns ['timestamp', 'correlation']
    if "correlation" in ccf_result.columns:
        ccf_peak = float(ccf_result["correlation"].abs().max())
    else:
        # Fallback: take the last column (assumed to be the correlation values)
        ccf_peak = float(ccf_result.iloc[:, -1].abs().max())

    ccf_check_pass = ccf_peak >= ccf_threshold

    return {
        "n_beats_compared": n,
        "per_beat": per_beat,
        "ioi_pass_rate": ioi_pass_rate,
        "ioi_check_pass": ioi_check_pass,
        "ccf_peak": ccf_peak,
        "ccf_check_pass": ccf_check_pass,
        "overall_pass": ioi_check_pass and ccf_check_pass,
    }


def print_report(result: dict, noise_sd: float) -> None:
    """
    Print a human-readable analysis report to stdout.

    Parameters
    ----------
    result : dict
        Output dict from :func:`analyze`.
    noise_sd : float
        Tolerance standard deviation used, shown in the header for context.
    """
    print("\n" + "=" * 60)
    print("  RHYTHM ANALYSIS REPORT")
    print("=" * 60)
    print(f"  Tolerance: ±{result['per_beat'][0]['tolerance_ms']:.1f} ms  "
          f"(noise_sd={noise_sd:.1f} ms × 2)")
    print(f"  Beats compared: {result['n_beats_compared']}")
    print()
    print(f"{'Beat':>5}  {'Ref IOI':>9}  {'Test IOI':>9}  {'Deviation':>10}  {'Pass':>5}")
    print("-" * 50)
    for b in result["per_beat"]:
        flag = "PASS" if b["pass"] else "FAIL"
        print(
            f"{b['beat']:>5}  {b['ref_ioi_ms']:>8.1f}ms  "
            f"{b['test_ioi_ms']:>8.1f}ms  "
            f"{b['deviation_ms']:>9.1f}ms  {flag:>5}"
        )
    print("-" * 50)
    print(f"  IOI pass rate : {result['ioi_pass_rate']:.0%}  "
          f"({'PASS' if result['ioi_check_pass'] else 'FAIL'})")
    print(f"  CCF peak      : {result['ccf_peak']:.3f}  "
          f"({'PASS' if result['ccf_check_pass'] else 'FAIL'})")
    print()
    verdict = "PASS" if result["overall_pass"] else "FAIL"
    print(f"  OVERALL VERDICT: {verdict}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Algorithm 1 — Fixed 7-Note Chunks (phrase-boundary aligned)
# ---------------------------------------------------------------------------

def fixed_chunk_accuracy(
    ref_iois: np.ndarray,
    test_iois: np.ndarray,
    noise_sd: float,
    pass_threshold: float = 70.0,
) -> dict:
    """
    Divide both 28-IOI arrays into 4 non-overlapping chunks of 7 and score each.

    Parameters
    ----------
    ref_iois : np.ndarray  shape (28,)
    test_iois : np.ndarray  shape (28,)
    noise_sd : float
        Tolerance SD in ms; an IOI is correct when |test-ref| <= 2*noise_sd.
    pass_threshold : float
        Minimum best-chunk accuracy (%) required to return verdict "PASS".

    Returns
    -------
    dict with keys:
        chunk_accuracies : list[float]  — accuracy % for each of the 4 chunks
        best_accuracy    : float
        best_index       : int          — 0-based index of the best chunk
        verdict          : "PASS" | "FAIL"
    """
    tolerance = 2.0 * noise_sd
    chunk_accuracies: list[float] = []
    for i in range(_N_CHUNKS):
        start = i * _CHUNK_SIZE
        end = start + _CHUNK_SIZE
        correct = int(np.sum(np.abs(test_iois[start:end] - ref_iois[start:end]) <= tolerance))
        chunk_accuracies.append(correct / _CHUNK_SIZE * 100.0)

    best_index = int(np.argmax(chunk_accuracies))
    best_accuracy = chunk_accuracies[best_index]
    return {
        "chunk_accuracies": chunk_accuracies,
        "best_accuracy": best_accuracy,
        "best_index": best_index,
        "verdict": "PASS" if best_accuracy >= pass_threshold else "FAIL",
    }


# ---------------------------------------------------------------------------
# Algorithm 2 — Sliding Window (any consecutive 7 correct IOIs)
# ---------------------------------------------------------------------------

def sliding_window_accuracy(
    ref_iois: np.ndarray,
    test_iois: np.ndarray,
    noise_sd: float,
    pass_threshold: float = 70.0,
) -> dict:
    """
    Slide a 7-IOI window across the test array; for each position find the
    best-matching reference window alignment.

    Parameters
    ----------
    ref_iois : np.ndarray  shape (28,)
    test_iois : np.ndarray  shape (28,)
    noise_sd : float
    pass_threshold : float

    Returns
    -------
    dict with keys:
        chunk_accuracies : list[float]  — best-match accuracy for each of the 22 test windows
        best_accuracy    : float
        best_index       : int          — start index of the best test window
        verdict          : "PASS" | "FAIL"
    """
    tolerance = 2.0 * noise_sd
    n = len(test_iois)
    n_windows = n - _CHUNK_SIZE + 1  # 22 windows for n=28

    window_accuracies: list[float] = []
    for i in range(n_windows):
        correct = int(np.sum(
            np.abs(test_iois[i : i + _CHUNK_SIZE] - ref_iois[i : i + _CHUNK_SIZE]) <= tolerance
        ))
        window_accuracies.append(correct / _CHUNK_SIZE * 100.0)

    best_index = int(np.argmax(window_accuracies))
    best_accuracy = window_accuracies[best_index]
    return {
        "chunk_accuracies": window_accuracies,
        "best_accuracy": best_accuracy,
        "best_index": best_index,
        "verdict": "PASS" if best_accuracy >= pass_threshold else "FAIL",
    }


# ---------------------------------------------------------------------------
# Pretty-print helpers for chunk algorithms
# ---------------------------------------------------------------------------

def print_chunk_report(result: dict, algorithm: str, pass_threshold: float) -> None:
    """
    Print a formatted report for fixed_chunk_accuracy or sliding_window_accuracy output.

    Parameters
    ----------
    result : dict
        Return value from fixed_chunk_accuracy or sliding_window_accuracy.
    algorithm : str
        "fixed" or "sliding".
    pass_threshold : float
        The threshold used, shown in the verdict line.
    """
    best_idx = result["best_index"]
    accs = result["chunk_accuracies"]

    if algorithm == "fixed":
        print("\n=== Algorithm 1: Fixed Chunks ===")
        for i, acc in enumerate(accs):
            start = i * _CHUNK_SIZE
            end = start + _CHUNK_SIZE - 1
            marker = "  ← best" if i == best_idx else ""
            print(f"Chunk {i + 1} (IOIs {start}–{end}):  {acc:.1f}%{marker}")
        print(
            f"Best chunk accuracy: {result['best_accuracy']:.1f}%"
            f" — {result['verdict']} (threshold: {pass_threshold:.0f}%)"
        )
    else:
        print("\n=== Algorithm 2: Sliding Window ===")
        # Show top 5 windows sorted by accuracy descending
        top5 = sorted(enumerate(accs), key=lambda x: -x[1])[:5]
        for idx, acc in top5:
            marker = "  ← best" if idx == best_idx else ""
            print(f"Window starting at IOI {idx}:  {acc:.1f}%{marker}")
        print(
            f"Best window accuracy: {result['best_accuracy']:.1f}%"
            f" — {result['verdict']} (threshold: {pass_threshold:.0f}%)"
        )
