"""
visualizer_dual.py — Dual-Channel Visualizations

2×2 grid:
  Top-left    : both reference channel waveforms overlaid
  Top-right   : human onset times vs. both reference onset grids
  Bottom-left : per-chunk direct IOI accuracy bar chart
  Bottom-right: per-chunk relational accuracy bar chart
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_dual_all(
    beat_array: np.ndarray,
    rhythm_array: np.ndarray,
    sample_rate: int,
    human_onsets_ms: np.ndarray,
    beat_ref_onsets_ms: np.ndarray,
    rhythm_ref_onsets_ms: np.ndarray,
    fixed_result: dict | None = None,
    pass_threshold: float = 70.0,
) -> None:
    """
    Display the full 2×2 diagnostic grid.

    Parameters
    ----------
    beat_array           : left-channel audio samples
    rhythm_array         : right-channel audio samples
    sample_rate          : audio sample rate (Hz)
    human_onsets_ms      : human onset times in ms (absolute, from cumsum of IOIs)
    beat_ref_onsets_ms   : reference beat onset times in ms
    rhythm_ref_onsets_ms : reference rhythm onset times in ms
    fixed_result         : output dict from fixed_dual_chunks (for bottom row)
    pass_threshold       : threshold used for PASS/FAIL colouring
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Dual-Channel Rhythm Analysis", fontsize=14)

    _plot_overlaid_waveforms(axes[0, 0], beat_array, rhythm_array, sample_rate)
    _plot_onset_grid(axes[0, 1], human_onsets_ms, beat_ref_onsets_ms, rhythm_ref_onsets_ms)

    if fixed_result is not None:
        _plot_accuracy_bars(
            axes[1, 0],
            fixed_result["chunk_direct_accuracies"],
            pass_threshold,
            title="Fixed Chunks — Direct IOI Accuracy",
        )
        _plot_accuracy_bars(
            axes[1, 1],
            fixed_result["chunk_relational_accuracies"],
            pass_threshold,
            title="Fixed Chunks — Relational Accuracy",
        )
    else:
        for ax in (axes[1, 0], axes[1, 1]):
            ax.text(0.5, 0.5, "No fixed-chunk data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_axis_off()

    plt.tight_layout()
    plt.show()


def _plot_overlaid_waveforms(
    ax: plt.Axes,
    beat_array: np.ndarray,
    rhythm_array: np.ndarray,
    sr: int,
) -> None:
    t_beat = np.arange(len(beat_array)) / sr
    t_rhythm = np.arange(len(rhythm_array)) / sr
    ax.plot(t_beat, beat_array, color="steelblue", linewidth=0.4, alpha=0.8, label="Beat (L)")
    ax.plot(t_rhythm, rhythm_array, color="darkorange", linewidth=0.4, alpha=0.8, label="Rhythm (R)")
    ax.set_title("Reference Channels Overlaid")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", fontsize=8)


def _plot_onset_grid(
    ax: plt.Axes,
    human_ms: np.ndarray,
    beat_ms: np.ndarray,
    rhythm_ms: np.ndarray,
) -> None:
    duration = max(
        human_ms[-1] if len(human_ms) else 0,
        beat_ms[-1] if len(beat_ms) else 0,
        rhythm_ms[-1] if len(rhythm_ms) else 0,
    ) * 1.05

    def _ticks(times, y, color, label):
        for t in times:
            ax.vlines(t, y - 0.08, y + 0.08, color=color, linewidth=1.5)
        ax.plot([], [], color=color, label=label, linewidth=2)

    _ticks(beat_ms, 0.7, "steelblue", "Ref Beat")
    _ticks(rhythm_ms, 0.4, "darkorange", "Ref Rhythm")
    _ticks(human_ms, 0.1, "forestgreen", "Human")

    ax.set_xlim(0, duration)
    ax.set_ylim(-0.05, 0.85)
    ax.set_yticks([0.1, 0.4, 0.7])
    ax.set_yticklabels(["Human", "Ref Rhythm", "Ref Beat"], fontsize=8)
    ax.set_xlabel("Time (ms)")
    ax.set_title("Onset Alignment")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", linewidth=1.0, alpha=0.8, color="dimgray")
    ax.grid(True, which="minor", linestyle="--", linewidth=0.6, alpha=0.6, color="darkgray")


def plot_ioi_deviations(
    per_onset: list[dict],
    noise_sd: float,
    title: str = "Per-Onset Relational Deviation",
) -> None:
    """
    Show a bar for each onset indicating how far the human's relational offset
    was from the expected offset against the partner channel.
    Signed: positive = human onset was further from the beat than expected,
            negative = closer to the beat than expected.
    The shaded band marks the ±tolerance window; bars are green inside it and red outside.
    """
    tol = 2.0 * noise_sd
    n = len(per_onset)
    deviations = np.array([p["actual_offset_ms"] - p["expected_offset_ms"] for p in per_onset])
    passed = np.array([p["pass"] for p in per_onset])
    n_pass = int(np.sum(passed))

    fig, ax = plt.subplots(figsize=(max(10, n * 0.6), 5))

    x = np.arange(1, n + 1)
    colors = ["forestgreen" if p else "tomato" for p in passed]
    ax.bar(x, deviations, color=colors, edgecolor="black", linewidth=0.6, zorder=3)

    ax.axhspan(-tol, tol, alpha=0.12, color="limegreen", zorder=1)
    ax.axhline(tol, color="green", linestyle="--", linewidth=1.2, zorder=2)
    ax.axhline(-tol, color="green", linestyle="--", linewidth=1.2, zorder=2)
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)

    ax.set_xticks(x)
    ax.set_xlabel("Onset #")
    ax.set_ylabel("Relational offset error (ms)  [+ = further from beat than expected, − = closer]")
    ax.set_title(f"{title}   ({n_pass}/{n} passed)")

    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.7, color="dimgray", zorder=0)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.5, color="darkgray", zorder=0)
    ax.minorticks_on()
    ax.set_axisbelow(True)

    pass_patch = mpatches.Patch(color="forestgreen", label="PASS")
    fail_patch = mpatches.Patch(color="tomato", label="FAIL")
    tol_patch = mpatches.Patch(color="limegreen", alpha=0.4, label=f"Tolerance ±{tol:.0f} ms")
    ax.legend(handles=[pass_patch, fail_patch, tol_patch], fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.show()


def _plot_accuracy_bars(
    ax: plt.Axes,
    accuracies: list[float],
    pass_threshold: float,
    title: str,
) -> None:
    x = np.arange(len(accuracies))
    colors = ["forestgreen" if a >= pass_threshold else "tomato" for a in accuracies]
    ax.bar(x, accuracies, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(pass_threshold, color="black", linestyle="--", linewidth=1, label=f"Threshold {pass_threshold:.0f}%")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i+1}" for i in x], fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    pass_patch = mpatches.Patch(color="forestgreen", label="PASS")
    fail_patch = mpatches.Patch(color="tomato", label="FAIL")
    ax.legend(handles=[pass_patch, fail_patch], fontsize=8, loc="lower right")
