"""
visualizer_dual.py — Dual-Channel Visualizations

2×2 grid (+ optional full-width bottom row):
  Top-left    : both reference channel waveforms overlaid
  Top-right   : human onset times vs. both reference onset grids
  Mid-left    : per-chunk direct IOI accuracy bar chart
  Mid-right   : per-chunk relational accuracy bar chart
  Bottom (opt): per-onset deviation from reference (spans full width)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator, FuncFormatter


def plot_default_graphs(
    human_onsets_ms: np.ndarray,
    beat_ref_onsets_ms: np.ndarray,
    rhythm_ref_onsets_ms: np.ndarray,
    fixed_result: dict | None = None,
    pass_threshold: float = 100.0,
    per_onset_relational: list[dict] | None = None,
    noise_sd: float = 30.0,
) -> None:
    """
    Display the three default diagnostic graphs:
      - Onset Alignment
      - Relational Accuracy
      - Per-Onset Relational Deviation (if data available)
    """
    has_ioi = bool(per_onset_relational)
    n_rows = 2 + (1 if has_ioi else 0)

    fig = plt.figure(figsize=(12, 4 * n_rows))
    gs = GridSpec(n_rows, 1, figure=fig, hspace=0.55)

    ax_align = fig.add_subplot(gs[0, 0])
    ax_rel = fig.add_subplot(gs[1, 0])

    fig.suptitle("Dual-Channel Rhythm Analysis", fontsize=14)

    _plot_onset_grid(ax_align, human_onsets_ms, beat_ref_onsets_ms, rhythm_ref_onsets_ms)

    if fixed_result is not None:
        _plot_accuracy_bars(
            ax_rel,
            fixed_result["chunk_relational_accuracies"],
            pass_threshold,
            title="Fixed Chunks — Relational Accuracy",
        )
    else:
        ax_rel.text(0.5, 0.5, "No fixed-chunk data", ha="center", va="center",
                    transform=ax_rel.transAxes, color="gray")
        ax_rel.set_axis_off()

    if has_ioi:
        ax_dev = fig.add_subplot(gs[2, 0])
        _plot_ioi_deviations_ax(ax_dev, per_onset_relational, noise_sd)

    plt.tight_layout()
    plt.show()


def plot_extra_graphs(
    beat_array: np.ndarray,
    rhythm_array: np.ndarray,
    sample_rate: int,
    fixed_result: dict | None = None,
    pass_threshold: float = 100.0,
) -> None:
    """
    Display additional diagnostic graphs (shown only with --plot flag):
      - Reference Channels Overlaid (waveforms)
      - Fixed Chunks — Direct IOI Accuracy
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Additional Diagnostics", fontsize=14)

    _plot_overlaid_waveforms(axes[0], beat_array, rhythm_array, sample_rate)

    if fixed_result is not None:
        _plot_accuracy_bars(
            axes[1],
            fixed_result["chunk_direct_accuracies"],
            pass_threshold,
            title="Fixed Chunks — Direct IOI Accuracy",
        )
    else:
        axes[1].text(0.5, 0.5, "No fixed-chunk data", ha="center", va="center",
                     transform=axes[1].transAxes, color="gray")
        axes[1].set_axis_off()

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Standalone IOI deviation figure (kept for external use)
# ---------------------------------------------------------------------------

def plot_ioi_deviations(
    per_onset: list[dict],
    noise_sd: float,
    title: str = "Per-Onset Deviation from Reference",
) -> None:
    fig, ax = plt.subplots(figsize=(max(10, len(per_onset) * 0.6), 5))
    _plot_ioi_deviations_ax(ax, per_onset, noise_sd, title=title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Private panel helpers
# ---------------------------------------------------------------------------

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
    ax.xaxis.set_major_locator(MultipleLocator(600))
    ax.xaxis.set_minor_locator(MultipleLocator(200))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}" if x % 1200 == 0 else ""))
    ax.grid(True, which="major", linestyle="--", linewidth=1.0, alpha=0.8, color="dimgray")
    ax.grid(True, which="minor", linestyle="--", linewidth=0.6, alpha=0.6, color="darkgray")


def _plot_ioi_deviations_ax(
    ax: plt.Axes,
    per_onset: list[dict],
    noise_sd: float,
    title: str = "Per-Onset Deviation from Reference",
) -> None:
    tol = 2.0 * noise_sd
    n = len(per_onset)
    x = np.arange(1, n + 1)

    played_entries = [(i, p) for i, p in enumerate(per_onset)
                      if not p.get("not_played") and not p.get("extra")]
    n_played = len(played_entries)
    n_pass = sum(p["pass"] for _, p in played_entries)

    played_x = [i + 1 for i, _ in played_entries]
    deviations = [p["actual_offset_ms"] - p["expected_offset_ms"] for _, p in played_entries]
    colors = ["forestgreen" if p["pass"] else "tomato" for _, p in played_entries]

    if played_x:
        ax.bar(played_x, deviations, color=colors, edgecolor="black", linewidth=0.6, zorder=3)

    # Extra notes beyond the reference — drawn in purple.
    extra_entries = [(i, p) for i, p in enumerate(per_onset) if p.get("extra")]
    has_extra = bool(extra_entries)
    if has_extra:
        extra_x = [i + 1 for i, _ in extra_entries]
        extra_devs = [p["actual_offset_ms"] - p["expected_offset_ms"] for _, p in extra_entries]
        ax.bar(extra_x, extra_devs, color="mediumpurple", edgecolor="black", linewidth=0.6, zorder=3)
        # Separator line between reference and extra regions.
        sep = extra_x[0] - 0.5
        ax.axvline(sep, color="black", linestyle="--", linewidth=1.2, zorder=4)

    # Shade not-played regions as contiguous spans.
    has_not_played = False
    run_start = None
    for i, p in enumerate(per_onset):
        if p.get("not_played"):
            has_not_played = True
            if run_start is None:
                run_start = i + 0.5
        else:
            if run_start is not None:
                ax.axvspan(run_start, i + 0.5, alpha=0.10, color="gray", zorder=0)
                run_start = None
    if run_start is not None:
        ax.axvspan(run_start, n + 0.5, alpha=0.10, color="gray", zorder=0)

    ax.axhspan(-tol, tol, alpha=0.12, color="limegreen", zorder=1)
    ax.axhline(tol, color="green", linestyle="--", linewidth=1.2, zorder=2)
    ax.axhline(-tol, color="green", linestyle="--", linewidth=1.2, zorder=2)
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)

    ax.set_xticks(x)
    ax.set_xlabel("Onset #")
    ax.set_ylabel("Deviation from reference (ms)  [+ = late, − = early]")
    extra_suffix = f"  +{len(extra_entries)} extra" if has_extra else ""
    ax.set_title(f"{title}   ({n_pass}/{n_played} passed{extra_suffix})")

    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.7, color="dimgray", zorder=0)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.5, color="darkgray", zorder=0)
    ax.minorticks_on()
    ax.set_axisbelow(True)

    pass_patch = mpatches.Patch(color="forestgreen", label="PASS")
    fail_patch = mpatches.Patch(color="tomato", label="FAIL")
    tol_patch = mpatches.Patch(color="limegreen", alpha=0.4, label=f"Tolerance ±{tol:.0f} ms")
    handles = [pass_patch, fail_patch, tol_patch]
    if has_not_played:
        handles.append(mpatches.Patch(color="gray", alpha=0.3, label="Not played"))
    if has_extra:
        handles.append(mpatches.Patch(color="mediumpurple", label="Extra (beyond reference)"))
    ax.legend(handles=handles, fontsize=8, loc="upper right")


def _plot_accuracy_bars(
    ax: plt.Axes,
    accuracies: list[float],
    pass_threshold: float,
    title: str,
) -> None:
    x = np.arange(len(accuracies))
    colors = ["forestgreen" if a >= pass_threshold else "tomato" for a in accuracies]
    ax.bar(x, accuracies, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(pass_threshold, color="black", linestyle="--", linewidth=1,
               label=f"Threshold {pass_threshold:.0f}%")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i+1}" for i in x], fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    pass_patch = mpatches.Patch(color="forestgreen", label="PASS")
    fail_patch = mpatches.Patch(color="tomato", label="FAIL")
    ax.legend(handles=[pass_patch, fail_patch], fontsize=8, loc="lower right")
