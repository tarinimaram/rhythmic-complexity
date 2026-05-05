"""
tests/test_dual_analyzer.py — Synthetic tests for the dual-channel analysis modules.

Test A: perfect beat performance — both checks PASS
Test B: direct match but globally shifted (phase offset) → Check 1 PASS, Check 2 FAIL
Test C: only IOIs 7–13 within tolerance → fixed chunk 2 / sliding window at 7 best, PASS
Test D: rhythm part, relational correct but IOIs outside direct tolerance → Check 1 FAIL
"""

import numpy as np
import pytest

from dual_channel_analyzer import analyze_dual, check_direct_ioi, check_relational
from chunk_dual_analyzer import fixed_dual_chunks, sliding_dual_window


BEAT_IOIS = np.array([500.0] * 28)
RHYTHM_IOIS = np.array([300.0, 200.0] * 14)
NOISE_SD = 30.0
PASS_THRESHOLD = 70.0


# ---------------------------------------------------------------------------
# Test A — perfect beat: direct PASS + relational PASS
# ---------------------------------------------------------------------------

class TestA:
    def test_direct_pass(self):
        r = analyze_dual(BEAT_IOIS, RHYTHM_IOIS, BEAT_IOIS.copy(),
                         "beat", NOISE_SD, PASS_THRESHOLD)
        assert r["direct_pass"] is True

    def test_relational_pass(self):
        r = analyze_dual(BEAT_IOIS, RHYTHM_IOIS, BEAT_IOIS.copy(),
                         "beat", NOISE_SD, PASS_THRESHOLD)
        assert r["relational_pass"] is True

    def test_overall_pass(self):
        r = analyze_dual(BEAT_IOIS, RHYTHM_IOIS, BEAT_IOIS.copy(),
                         "beat", NOISE_SD, PASS_THRESHOLD)
        assert r["overall_pass"] is True


# ---------------------------------------------------------------------------
# Test B — direct PASS, phase offset breaks relational
#
# A performer starting 400 ms late plays the exact same IOIs (direct PASS),
# but their onset times are globally shifted so relational offsets are wrong.
# A phase offset is invisible in the IOI array itself; we model it via
# human_onset_offset_ms in check_relational.
# ---------------------------------------------------------------------------

class TestB:
    SHIFT = 400.0  # ms — well beyond 2*noise_sd = 60 ms

    def test_direct_pass(self):
        direct = check_direct_ioi(BEAT_IOIS, BEAT_IOIS, NOISE_SD)
        assert direct["pass"] is True

    def test_relational_fail(self):
        rel = check_relational(
            BEAT_IOIS, BEAT_IOIS, RHYTHM_IOIS, "beat", NOISE_SD,
            human_onset_offset_ms=self.SHIFT,
        )
        assert rel["accuracy_pct"] < PASS_THRESHOLD

    def test_combined_fail(self):
        direct = check_direct_ioi(BEAT_IOIS, BEAT_IOIS, NOISE_SD)
        rel = check_relational(
            BEAT_IOIS, BEAT_IOIS, RHYTHM_IOIS, "beat", NOISE_SD,
            human_onset_offset_ms=self.SHIFT,
        )
        assert direct["pass"] is True
        assert rel["accuracy_pct"] < PASS_THRESHOLD


# ---------------------------------------------------------------------------
# Test C — only IOIs 7–13 within tolerance
# ---------------------------------------------------------------------------

class TestC:
    """Chunk 2 (0-based index 1, positions 7–13) is the only accurate region."""

    def _human_iois(self):
        human = np.full(28, 999.0)
        human[7:14] = RHYTHM_IOIS[7:14]
        return human

    def test_fixed_best_chunk_is_index_1(self):
        result = fixed_dual_chunks(
            BEAT_IOIS, RHYTHM_IOIS, self._human_iois(), "rhythm", NOISE_SD, PASS_THRESHOLD
        )
        assert result["best_chunk_index"] == 1

    def test_fixed_verdict_pass(self):
        result = fixed_dual_chunks(
            BEAT_IOIS, RHYTHM_IOIS, self._human_iois(), "rhythm", NOISE_SD, PASS_THRESHOLD
        )
        assert result["verdict"] == "PASS"

    def test_sliding_finds_window_at_7(self):
        result = sliding_dual_window(
            BEAT_IOIS, RHYTHM_IOIS, self._human_iois(), "rhythm", NOISE_SD, PASS_THRESHOLD
        )
        assert result["best_chunk_index"] == 7

    def test_sliding_verdict_pass(self):
        result = sliding_dual_window(
            BEAT_IOIS, RHYTHM_IOIS, self._human_iois(), "rhythm", NOISE_SD, PASS_THRESHOLD
        )
        assert result["verdict"] == "PASS"

    def test_best_direct_accuracy_100(self):
        result = fixed_dual_chunks(
            BEAT_IOIS, RHYTHM_IOIS, self._human_iois(), "rhythm", NOISE_SD, PASS_THRESHOLD
        )
        assert result["best_direct_accuracy"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Test D — rhythm part: IOIs outside direct tolerance, combined FAIL
# ---------------------------------------------------------------------------

class TestD:
    """Human plays rhythm with each IOI stretched just past the direct tolerance."""

    def _human_iois(self):
        deviation = 2 * NOISE_SD + 10.0
        return RHYTHM_IOIS + deviation

    def test_direct_fail(self):
        r = analyze_dual(BEAT_IOIS, RHYTHM_IOIS, self._human_iois(),
                         "rhythm", NOISE_SD, PASS_THRESHOLD)
        assert r["direct_pass"] is False

    def test_overall_fail(self):
        r = analyze_dual(BEAT_IOIS, RHYTHM_IOIS, self._human_iois(),
                         "rhythm", NOISE_SD, PASS_THRESHOLD)
        assert r["overall_pass"] is False

    def test_best_direct_accuracy_zero(self):
        r = analyze_dual(BEAT_IOIS, RHYTHM_IOIS, self._human_iois(),
                         "rhythm", NOISE_SD, PASS_THRESHOLD)
        assert r["direct_result"]["accuracy_pct"] == pytest.approx(0.0)
