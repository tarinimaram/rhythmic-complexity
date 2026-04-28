"""
visualizer.py — Plotting

Four plots for inspecting and comparing a reference and test rhythm:
  1. Waveforms side-by-side
  2. Sequence event plots side-by-side
  3. Cross-correlation function
  4. Per-beat IOI deviation bar chart with ±2σ tolerance lines
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from thebeat import SoundStimulus, Sequence
from thebeat.stats import ccf_plot


def plot_waveforms(
    ref_stimulus: SoundStimulus,
    test_stimulus: SoundStimulus,
) -> plt.Figure:
    """
    Plot reference and test waveforms side-by-side.

    Parameters
    ----------
    ref_stimulus : SoundStimulus
        Reference audio stimulus.
    test_stimulus : SoundStimulus
        Test (human performance) audio stimulus.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    fig.suptitle("Waveforms", fontsize=14, fontweight="bold")

    ref_stimulus.plot_waveform(title="Reference", ax=axes[0])
    test_stimulus.plot_waveform(title="Human Performance", ax=axes[1])

    fig.tight_layout()
    return fig


def plot_sequences(
    ref_sequence: Sequence,
    test_sequence: Sequence,
) -> plt.Figure:
    """
    Plot reference and test event sequences side-by-side.

    Parameters
    ----------
    ref_sequence : Sequence
        Reference timing Sequence.
    test_sequence : Sequence
        Test timing Sequence from the human performance.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=False)
    fig.suptitle("Event Sequences", fontsize=14, fontweight="bold")

    ref_sequence.plot_sequence(title="Reference", ax=axes[0])
    test_sequence.plot_sequence(title="Human Performance", ax=axes[1])

    fig.tight_layout()
    return fig


def plot_ccf(
    ref_sequence: Sequence,
    test_sequence: Sequence,
    resolution: float = 10.0,
) -> plt.Figure:
    """
    Plot the cross-correlation function between test and reference sequences.

    Parameters
    ----------
    ref_sequence : Sequence
        Reference timing Sequence.
    test_sequence : Sequence
        Test timing Sequence.
    resolution : float
        Bin width in ms for the CCF histogram. Default 10.0.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ccf_plot(test_sequence, ref_sequence, resolution=resolution, ax=ax)
    ax.set_title("Cross-Correlation Function (Test vs. Reference)", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_ioi_deviations(
    per_beat: list[dict],
    noise_sd: float,
    tolerance_multiplier: float = 2.0,
) -> plt.Figure:
    """
    Bar chart of per-beat IOI deviations with ±(tolerance_multiplier × noise_sd) lines.

    Parameters
    ----------
    per_beat : list[dict]
        Per-beat dicts from :func:`analyzer.analyze` result["per_beat"].
    noise_sd : float
        Tolerance standard deviation in milliseconds.
    tolerance_multiplier : float
        Number of SDs that define the pass/fail band. Default 2.0.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    beats = [b["beat"] for b in per_beat]
    deviations = [b["deviation_ms"] for b in per_beat]
    colors = ["#4caf50" if b["pass"] else "#f44336" for b in per_beat]
    tolerance = tolerance_multiplier * noise_sd

    fig, ax = plt.subplots(figsize=(max(8, len(beats) * 0.6 + 2), 5))
    ax.bar(beats, deviations, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(
        tolerance,
        color="#e53935",
        linestyle="--",
        linewidth=1.5,
    )
    ax.set_xlabel("Beat number")
    ax.set_ylabel("Absolute IOI deviation (ms)")
    ax.set_title("Per-beat IOI Deviation", fontweight="bold")
    ax.set_xticks(beats)

    legend_elements = [
        Patch(facecolor="#4caf50", label="Pass"),
        Patch(facecolor="#f44336", label="Fail"),
        plt.Line2D([0], [0], color="#e53935", linestyle="--",
                   label=f"Tolerance ±{tolerance:.0f} ms"),
    ]
    ax.legend(handles=legend_elements)
    fig.tight_layout()
    return fig


def plot_fixed_chunks(
    result: dict,
    pass_threshold: float = 70.0,
) -> plt.Figure:
    """
    Bar chart of per-chunk accuracy for Algorithm 1 (fixed chunks).

    Parameters
    ----------
    result : dict
        Return value from analyzer.fixed_chunk_accuracy.
    pass_threshold : float
        Accuracy threshold drawn as a dashed horizontal line.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    accs = result["chunk_accuracies"]
    n = len(accs)
    labels = [f"Chunk {i + 1}\n(IOIs {i * 7}–{i * 7 + 6})" for i in range(n)]
    colors = ["#4caf50" if a >= pass_threshold else "#f44336" for a in accs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(n), accs, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(
        pass_threshold,
        color="#e53935",
        linestyle="--",
        linewidth=1.5,
    )
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Algorithm 1: Fixed Chunk Accuracy", fontweight="bold")

    legend_elements = [
        Patch(facecolor="#4caf50", label="Pass"),
        Patch(facecolor="#f44336", label="Fail"),
        plt.Line2D([0], [0], color="#e53935", linestyle="--",
                   label=f"Threshold {pass_threshold:.0f}%"),
    ]
    ax.legend(handles=legend_elements)
    fig.tight_layout()
    return fig


def plot_sliding_window(
    result: dict,
    pass_threshold: float = 70.0,
) -> plt.Figure:
    """
    Line chart of best-match accuracy for each sliding window position (Algorithm 2).

    Parameters
    ----------
    result : dict
        Return value from analyzer.sliding_window_accuracy.
    pass_threshold : float
        Accuracy threshold drawn as a dashed horizontal line.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    accs = result["chunk_accuracies"]
    best_idx = result["best_index"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(accs)), accs, color="#1976d2", linewidth=1.5, marker="o", markersize=4)
    ax.axhline(
        pass_threshold,
        color="#e53935",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold {pass_threshold:.0f}%",
    )
    ax.scatter(
        [best_idx], [accs[best_idx]],
        color="#ff6f00", zorder=5, s=80,
        label=f"Best (IOI {best_idx}: {accs[best_idx]:.1f}%)",
    )
    ax.set_xlabel("Window start index (IOI)")
    ax.set_ylabel("Best-match accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Algorithm 2: Sliding Window Accuracy", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig


def show_all_plots(
    ref_stimulus: SoundStimulus,
    test_stimulus: SoundStimulus,
    ref_sequence: Sequence,
    test_sequence: Sequence,
    per_beat: list[dict],
    noise_sd: float,
    ccf_resolution: float = 10.0,
) -> None:
    """
    Render and display all four diagnostic plots.

    Parameters
    ----------
    ref_stimulus : SoundStimulus
        Reference audio stimulus.
    test_stimulus : SoundStimulus
        Test audio stimulus.
    ref_sequence : Sequence
        Reference timing Sequence.
    test_sequence : Sequence
        Test timing Sequence.
    per_beat : list[dict]
        Per-beat results from :func:`analyzer.analyze`.
    noise_sd : float
        Tolerance standard deviation in milliseconds.
    ccf_resolution : float
        Bin width in ms for the CCF plot. Default 10.0.
    """
    plot_waveforms(ref_stimulus, test_stimulus)
    plot_sequences(ref_sequence, test_sequence)
    plot_ccf(ref_sequence, test_sequence, resolution=ccf_resolution)
    plot_ioi_deviations(per_beat, noise_sd)
    plt.show()
