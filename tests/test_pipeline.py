"""
tests/test_pipeline.py

End-to-end tests using synthetically generated sequences — no real WAV files needed.
We bypass the WAV-loading steps and test the analyzer logic directly.
"""

import sys
import os
import numpy as np
import pytest

# Allow importing from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from thebeat import Sequence
from analyzer import analyze, fixed_chunk_accuracy, sliding_window_accuracy


def make_sequence(iois_ms: np.ndarray) -> Sequence:
    """Helper: wrap an IOI array in a thebeat Sequence."""
    return Sequence(iois=iois_ms)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ISOCHRONOUS_IOI = 500.0  # ms — perfect metronome at 120 BPM
N_BEATS = 8
REF_IOIS = np.full(N_BEATS, ISOCHRONOUS_IOI)
NOISE_SD = 20.0  # ms tolerance


# ---------------------------------------------------------------------------
# IOI deviation check
# ---------------------------------------------------------------------------

class TestIOIDeviationCheck:
    def test_perfect_match_passes(self):
        """Identical sequences should pass with 100% beat pass rate."""
        ref_seq = make_sequence(REF_IOIS)
        test_seq = make_sequence(REF_IOIS.copy())
        result = analyze(
            ref_sequence=ref_seq,
            ref_iois=REF_IOIS,
            noise_sd=NOISE_SD,
            test_sequence=test_seq,
            test_iois=REF_IOIS.copy(),
            ccf_threshold=0.0,  # disable CCF gate for this test
        )
        assert result["ioi_pass_rate"] == 1.0
        assert result["ioi_check_pass"] is True

    def test_small_jitter_passes(self):
        """Jitter within ±1 SD should pass every beat."""
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-NOISE_SD * 0.9, NOISE_SD * 0.9, size=N_BEATS)
        test_iois = REF_IOIS + jitter
        ref_seq = make_sequence(REF_IOIS)
        test_seq = make_sequence(test_iois)
        result = analyze(
            ref_sequence=ref_seq,
            ref_iois=REF_IOIS,
            noise_sd=NOISE_SD,
            test_sequence=test_seq,
            test_iois=test_iois,
            ccf_threshold=0.0,
        )
        assert result["ioi_check_pass"] is True, (
            f"Expected all beats to pass with jitter < 1SD, "
            f"but pass rate was {result['ioi_pass_rate']:.0%}"
        )

    def test_large_jitter_fails(self):
        """Jitter far outside ±2 SD should cause at least one beat to fail."""
        # Every beat is 3× the tolerance off
        test_iois = REF_IOIS + NOISE_SD * 3
        ref_seq = make_sequence(REF_IOIS)
        test_seq = make_sequence(test_iois)
        result = analyze(
            ref_sequence=ref_seq,
            ref_iois=REF_IOIS,
            noise_sd=NOISE_SD,
            test_sequence=test_seq,
            test_iois=test_iois,
            ccf_threshold=0.0,
        )
        assert result["ioi_check_pass"] is False
        assert result["ioi_pass_rate"] == 0.0

    def test_boundary_exactly_at_tolerance(self):
        """Deviations exactly at 2×SD should pass (inclusive boundary)."""
        test_iois = REF_IOIS + (NOISE_SD * 2)  # exactly at the edge
        ref_seq = make_sequence(REF_IOIS)
        test_seq = make_sequence(test_iois)
        result = analyze(
            ref_sequence=ref_seq,
            ref_iois=REF_IOIS,
            noise_sd=NOISE_SD,
            test_sequence=test_seq,
            test_iois=test_iois,
            ccf_threshold=0.0,
        )
        assert result["ioi_check_pass"] is True, (
            "Deviation exactly at 2×SD should be treated as a pass."
        )

    def test_length_mismatch_trims_to_shorter(self):
        """If test has fewer beats than reference, only the shared portion is checked."""
        short_iois = REF_IOIS[:4]  # only half as many beats
        ref_seq = make_sequence(REF_IOIS)
        test_seq = make_sequence(short_iois)
        result = analyze(
            ref_sequence=ref_seq,
            ref_iois=REF_IOIS,
            noise_sd=NOISE_SD,
            test_sequence=test_seq,
            test_iois=short_iois,
            ccf_threshold=0.0,
        )
        assert result["n_beats_compared"] == 4
        assert len(result["per_beat"]) == 4


# ---------------------------------------------------------------------------
# CCF check
# ---------------------------------------------------------------------------

class TestCCFCheck:
    def test_identical_sequences_high_ccf(self):
        """CCF between identical isochronous sequences should exceed a high threshold."""
        ref_seq = make_sequence(REF_IOIS)
        test_seq = make_sequence(REF_IOIS.copy())
        result = analyze(
            ref_sequence=ref_seq,
            ref_iois=REF_IOIS,
            noise_sd=NOISE_SD,
            test_sequence=test_seq,
            test_iois=REF_IOIS.copy(),
            ccf_threshold=0.85,
        )
        assert result["ccf_check_pass"] is True, (
            f"Expected CCF peak ≥ 0.85 for identical sequences, "
            f"got {result['ccf_peak']:.3f}"
        )


# ---------------------------------------------------------------------------
# Overall verdict
# ---------------------------------------------------------------------------

class TestOverallVerdict:
    def test_pass_requires_both_checks(self):
        """overall_pass should be True only when both IOI and CCF checks pass."""
        ref_seq = make_sequence(REF_IOIS)
        test_seq = make_sequence(REF_IOIS.copy())
        result = analyze(
            ref_sequence=ref_seq,
            ref_iois=REF_IOIS,
            noise_sd=NOISE_SD,
            test_sequence=test_seq,
            test_iois=REF_IOIS.copy(),
            ccf_threshold=0.85,
        )
        assert result["overall_pass"] == (
            result["ioi_check_pass"] and result["ccf_check_pass"]
        )

    def test_high_ccf_threshold_fails_even_good_performance(self):
        """An impossibly high CCF threshold should force overall FAIL."""
        ref_seq = make_sequence(REF_IOIS)
        test_seq = make_sequence(REF_IOIS.copy())
        result = analyze(
            ref_sequence=ref_seq,
            ref_iois=REF_IOIS,
            noise_sd=NOISE_SD,
            test_sequence=test_seq,
            test_iois=REF_IOIS.copy(),
            ccf_threshold=1.1,  # impossible — CCF is bounded at 1.0
        )
        assert result["overall_pass"] is False


# ---------------------------------------------------------------------------
# Per-beat detail structure
# ---------------------------------------------------------------------------

class TestPerBeatStructure:
    def test_per_beat_fields_present(self):
        """Each per-beat entry should have all expected keys."""
        ref_seq = make_sequence(REF_IOIS)
        test_seq = make_sequence(REF_IOIS.copy())
        result = analyze(
            ref_sequence=ref_seq,
            ref_iois=REF_IOIS,
            noise_sd=NOISE_SD,
            test_sequence=test_seq,
            test_iois=REF_IOIS.copy(),
            ccf_threshold=0.0,
        )
        required_keys = {"beat", "ref_ioi_ms", "test_ioi_ms", "deviation_ms",
                         "tolerance_ms", "pass"}
        for b in result["per_beat"]:
            assert required_keys <= b.keys(), f"Missing keys in: {b}"

    def test_beat_numbers_are_one_indexed(self):
        ref_seq = make_sequence(REF_IOIS)
        test_seq = make_sequence(REF_IOIS.copy())
        result = analyze(
            ref_sequence=ref_seq,
            ref_iois=REF_IOIS,
            noise_sd=NOISE_SD,
            test_sequence=test_seq,
            test_iois=REF_IOIS.copy(),
            ccf_threshold=0.0,
        )
        beat_numbers = [b["beat"] for b in result["per_beat"]]
        assert beat_numbers[0] == 1
        assert beat_numbers[-1] == N_BEATS


# ---------------------------------------------------------------------------
# Fixed chunk accuracy
# ---------------------------------------------------------------------------

REF_28 = np.full(28, 500.0)
NOISE_SD_28 = 20.0  # tolerance = 40 ms


class TestFixedChunkAccuracy:
    def test_perfect_match_all_chunks_pass(self):
        """Identical 28-IOI arrays should give 100% on every chunk."""
        result = fixed_chunk_accuracy(REF_28, REF_28.copy(), NOISE_SD_28)
        assert result["chunk_accuracies"] == [100.0, 100.0, 100.0, 100.0]
        assert result["best_accuracy"] == 100.0
        assert result["verdict"] == "PASS"

    def test_only_iois_7_to_13_within_tolerance(self):
        """Only IOIs 7–13 within tolerance → chunk 2 (index 1) is best at 100%."""
        test = REF_28.copy()
        test[0:7] += NOISE_SD_28 * 3    # chunk 1 all fail
        test[14:28] += NOISE_SD_28 * 3  # chunks 3 & 4 all fail
        result = fixed_chunk_accuracy(REF_28, test, NOISE_SD_28, pass_threshold=70.0)
        assert result["best_index"] == 1
        assert result["best_accuracy"] == pytest.approx(100.0)
        assert result["verdict"] == "PASS"

    def test_cross_boundary_iois_9_to_15_does_not_achieve_full_chunk(self):
        """IOIs 9–15 span the boundary of chunks 2 and 3 — no chunk reaches 100%."""
        test = REF_28.copy()
        test += NOISE_SD_28 * 3         # all fail
        test[9:16] = REF_28[9:16]       # only indices 9–15 correct
        result = fixed_chunk_accuracy(REF_28, test, NOISE_SD_28, pass_threshold=80.0)
        # chunk 2 (7–13): 5/7 ≈ 71.4%;  chunk 3 (14–20): 2/7 ≈ 28.6%
        assert result["best_accuracy"] == pytest.approx(5 / 7 * 100, abs=0.1)
        assert result["verdict"] == "FAIL"

    def test_below_threshold_returns_fail(self):
        """When best chunk accuracy is below the threshold, verdict is FAIL."""
        test = REF_28.copy()
        test += NOISE_SD_28 * 3
        result = fixed_chunk_accuracy(REF_28, test, NOISE_SD_28, pass_threshold=70.0)
        assert result["verdict"] == "FAIL"

    def test_returns_four_chunks(self):
        result = fixed_chunk_accuracy(REF_28, REF_28.copy(), NOISE_SD_28)
        assert len(result["chunk_accuracies"]) == 4


# ---------------------------------------------------------------------------
# Sliding window accuracy
# ---------------------------------------------------------------------------

class TestSlidingWindowAccuracy:
    def test_perfect_match_all_windows_100(self):
        """Identical arrays → every window achieves 100% against its matching ref window."""
        result = sliding_window_accuracy(REF_28, REF_28.copy(), NOISE_SD_28)
        assert result["best_accuracy"] == 100.0
        assert result["verdict"] == "PASS"

    def test_cross_boundary_iois_9_to_15_found_by_sliding(self):
        """IOIs 9–15 within tolerance: sliding window finds 100%, fixed cannot."""
        test = REF_28.copy()
        test += NOISE_SD_28 * 3         # all fail
        test[9:16] = REF_28[9:16]       # only indices 9–15 correct
        result = sliding_window_accuracy(REF_28, test, NOISE_SD_28, pass_threshold=80.0)
        assert result["best_accuracy"] == pytest.approx(100.0)
        assert result["verdict"] == "PASS"

    def test_only_iois_7_to_13_within_tolerance(self):
        """When only IOIs 7–13 match, sliding window should also find 100%."""
        test = REF_28.copy()
        test[0:7] += NOISE_SD_28 * 3
        test[14:28] += NOISE_SD_28 * 3
        result = sliding_window_accuracy(REF_28, test, NOISE_SD_28, pass_threshold=70.0)
        assert result["best_accuracy"] == pytest.approx(100.0)
        assert result["verdict"] == "PASS"

    def test_returns_22_windows(self):
        """28-IOI input with window size 7 should produce 22 window positions."""
        result = sliding_window_accuracy(REF_28, REF_28.copy(), NOISE_SD_28)
        assert len(result["chunk_accuracies"]) == 22

    def test_below_threshold_returns_fail(self):
        test = REF_28.copy()
        test += NOISE_SD_28 * 3
        result = sliding_window_accuracy(REF_28, test, NOISE_SD_28, pass_threshold=70.0)
        assert result["verdict"] == "FAIL"
