"""
visualizer_dual.py — Dual-Channel Visualizations

Each graph is shown in its own separate window so every panel can be
zoomed, panned, and resized independently using the matplotlib toolbar.

Default windows:
  1 — Onset Alignment
  2 — Fixed Chunks Accuracy
  3 — Per-Onset Deviation from Reference  (if data available)

Extra windows (--plot flag):
  4 — Reference Channels Overlaid (waveforms)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    Open each default diagnostic graph in its own window.
    """
    # --- Window 1: Onset Alignment ---
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    fig1.canvas.manager.set_window_title("Onset Alignment")
    _plot_onset_grid(ax1, human_onsets_ms, beat_ref_onsets_ms, rhythm_ref_onsets_ms)
    fig1.tight_layout()

    # --- Window 2: Chunk Accuracy ---
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    fig2.canvas.manager.set_window_title("Fixed Chunks — Accuracy")
    if fixed_result is not None:
        _plot_accuracy_bars(
            ax2,
            fixed_result["chunk_direct_accuracies"],
            pass_threshold,
            title="Fixed Chunks — Onset-Deviation Accuracy",
        )
    else:
        ax2.text(0.5, 0.5, "No fixed-chunk data", ha="center", va="center",
                 transform=ax2.transAxes, color="gray")
        ax2.set_axis_off()
    fig2.tight_layout()

    # --- Window 3: Per-Onset Deviation ---
    if per_onset_relational:
        n = len(per_onset_relational)
        fig3, ax3 = plt.subplots(figsize=(max(10, n * 0.55), 5))
        fig3.canvas.manager.set_window_title("Per-Onset Deviation from Reference")
        _plot_ioi_deviations_ax(ax3, per_onset_relational, noise_sd)
        fig3.tight_layout()

    plt.show()


def plot_extra_graphs(
    beat_array: np.ndarray,
    rhythm_array: np.ndarray,
    sample_rate: int,
    fixed_result: dict | None = None,
    pass_threshold: float = 100.0,
) -> None:
    """
    Open each extra diagnostic graph in its own window (--plot flag).
    """
    # --- Window 4: Reference Waveforms ---
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    fig4.canvas.manager.set_window_title("Reference Channels Overlaid")
    _plot_overlaid_waveforms(ax4, beat_array, rhythm_array, sample_rate)
    fig4.tight_layout()

    plt.show()


# ---------------------------------------------------------------------------
# Standalone IOI deviation figure (kept for external use)
# ---------------------------------------------------------------------------

def plot_ioi_deviations(
    per_onset: list[dict],
    noise_sd: float,
    title: str = "Per-Onset Deviation from Reference",
) -> None:
    n = len(per_onset)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.55), 5))
    fig.canvas.manager.set_window_title(title)
    _plot_ioi_deviations_ax(ax, per_onset, noise_sd, title=title)
    fig.tight_layout()
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
                      if not p.get("not_played") and not p.get("extra")
                      and not p.get("discarded")]
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
        sep = extra_x[0] - 0.5
        ax.axvline(sep, color="black", linestyle="--", linewidth=1.2, zorder=4)

    # Shade not-played and discarded regions as contiguous spans.
    has_not_played = False
    has_discarded = False
    run_start = None
    run_kind = None  # "not_played" or "discarded"
    for i, p in enumerate(per_onset):
        kind = "not_played" if p.get("not_played") else ("discarded" if p.get("discarded") else None)
        if kind is not None:
            if kind == "not_played":
                has_not_played = True
            else:
                has_discarded = True
            if run_start is None:
                run_start = i + 0.5
                run_kind = kind
        else:
            if run_start is not None:
                color = "gray" if run_kind == "not_played" else "darkorange"
                ax.axvspan(run_start, i + 0.5, alpha=0.18, color=color, zorder=0)
                run_start = None
                run_kind = None
    if run_start is not None:
        color = "gray" if run_kind == "not_played" else "darkorange"
        ax.axvspan(run_start, n + 0.5, alpha=0.18, color=color, zorder=0)

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
    if has_discarded:
        handles.append(mpatches.Patch(color="darkorange", alpha=0.3, label="Discarded (extra/missing note)"))
    if has_extra:
        handles.append(mpatches.Patch(color="mediumpurple", label="Extra (beyond reference)"))
    ax.legend(handles=handles, fontsize=8, loc="upper right")


def _plot_accuracy_bars(
    ax: plt.Axes,
    accuracies: list,
    pass_threshold: float,
    title: str,
) -> None:
    x = np.arange(len(accuracies))
    has_discarded = any(a is None for a in accuracies)

    bar_heights = [a if a is not None else 0.0 for a in accuracies]
    colors = []
    for a in accuracies:
        if a is None:
            colors.append("lightgray")
        elif a >= pass_threshold:
            colors.append("forestgreen")
        else:
            colors.append("tomato")

    bars = ax.bar(x, bar_heights, color=colors, edgecolor="black", linewidth=0.6)

    # Hatch discarded bars and add a label inside them.
    for bar, a in zip(bars, accuracies):
        if a is None:
            bar.set_hatch("//")
            cx = bar.get_x() + bar.get_width() / 2
            ax.text(cx, 5, "N/A", ha="center", va="bottom", fontsize=7, color="gray")

    ax.axhline(pass_threshold, color="black", linestyle="--", linewidth=1,
               label=f"Threshold {pass_threshold:.0f}%")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i+1}" for i in x], fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    pass_patch = mpatches.Patch(color="forestgreen", label="PASS")
    fail_patch = mpatches.Patch(color="tomato", label="FAIL")
    handles = [pass_patch, fail_patch]
    if has_discarded:
        handles.append(mpatches.Patch(facecolor="lightgray", hatch="//", label="Discarded"))
    ax.legend(handles=handles, fontsize=8, loc="lower right")
