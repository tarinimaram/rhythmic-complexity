"""
main_dual.py — Entry Point for Dual-Channel Analysis

CLI orchestrator:
  split stereo reference → extract human onsets → run dual-channel algorithms → report
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import librosa

from channel_splitter import split_channels
from dual_channel_analyzer import analyze_dual, print_dual_report, _onsets_from_iois
from chunk_dual_analyzer import fixed_dual_chunks, sliding_dual_window, print_chunk_dual_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dual-channel rhythm accuracy analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reference", required=True, metavar="WAV",
                        help="Stereo reference WAV to split into beat/rhythm channels.")
    parser.add_argument("--test", required=True, metavar="WAV",
                        help="Human solo mono recording.")
    parser.add_argument("--human_part", required=True, choices=["beat", "rhythm"],
                        help="Which part the human performed.")
    parser.add_argument("--beat_iois", required=True, metavar="MS,...",
                        help="Comma-separated reference beat IOIs in ms.")
    parser.add_argument("--rhythm_iois", required=True, metavar="MS,...",
                        help="Comma-separated reference rhythm IOIs in ms.")
    parser.add_argument("--noise_sd", type=float, default=30.0, metavar="MS",
                        help="Tolerance SD in ms (pass window = ±2×noise_sd).")
    parser.add_argument("--pass_threshold", type=float, default=100.0, metavar="PCT",
                        help="Minimum accuracy %% required for each check to PASS.")
    parser.add_argument("--algorithm", choices=["fixed", "sliding", "both"], default="both",
                        help="Chunking algorithm(s) to run.")
    parser.add_argument("--plot", action="store_true",
                        help="Show all visualizations after analysis.")
    return parser.parse_args(argv)


def _parse_iois(raw: str) -> np.ndarray:
    try:
        return np.array([float(x.strip()) for x in raw.split(",")])
    except ValueError as exc:
        raise ValueError(f"Could not parse IOI list: {raw!r}") from exc


def _extract_human_onsets(wav_path: str) -> tuple[np.ndarray, float]:
    """Return (iois_ms, first_onset_ms) where first_onset_ms is the silence before the first note."""
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, backtrack=True, units="frames")
    times_sec = librosa.frames_to_time(frames, sr=sr, hop_length=512)
    if len(times_sec) < 2:
        raise ValueError(
            f"Only {len(times_sec)} onset(s) detected in '{wav_path}'. "
            "Need at least 2 to compute IOIs."
        )
    return np.diff(times_sec) * 1000.0, float(times_sec[0] * 1000.0)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        beat_iois = _parse_iois(args.beat_iois)
        rhythm_iois = _parse_iois(args.rhythm_iois)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"\nSplitting stereo reference: {args.reference}")
    try:
        beat_array, rhythm_array, sr = split_channels(args.reference, plot=args.plot)
    except Exception as exc:
        print(f"ERROR splitting channels: {exc}")
        return 1

    print(f"\nExtracting onsets from: {args.test}")
    try:
        human_iois, silence_ms = _extract_human_onsets(args.test)
    except Exception as exc:
        print(f"ERROR extracting human onsets: {exc}")
        return 1
    print(f"  Detected {len(human_iois)+1} onsets → {len(human_iois)} IOIs")
    print(f"  First onset at {silence_ms:.1f} ms (silence before playing)")

    # Determine which reference phrase the human started at.
    # The phrase duration is the sum of one chunk's worth of reference IOIs.
    ref_iois_for_part = beat_iois if args.human_part == "beat" else rhythm_iois
    phrase_len = 4 if args.human_part == "beat" else 7
    phrase_duration_ms = float(np.sum(ref_iois_for_part[:phrase_len]))
    start_chunk = int(round(silence_ms / phrase_duration_ms))
    start_chunk = max(0, min(start_chunk, 3))  # clamp to [0, N_CHUNKS-1]
    print(f"  Phrase ≈ {phrase_duration_ms:.0f} ms → human starts at chunk {start_chunk + 1}")

    # Full dual analysis (no chunking)
    result = analyze_dual(
        beat_iois=beat_iois,
        rhythm_iois=rhythm_iois,
        human_iois=human_iois,
        human_part=args.human_part,
        noise_sd=args.noise_sd,
        pass_threshold=args.pass_threshold,
    )
    print_dual_report(result, args.noise_sd, args.pass_threshold)

    run_fixed = args.algorithm in ("fixed", "both")
    run_sliding = args.algorithm in ("sliding", "both")
    fixed_result = None
    sliding_result = None

    if run_fixed:
        fixed_result = fixed_dual_chunks(
            beat_iois, rhythm_iois, human_iois,
            args.human_part, args.noise_sd, args.pass_threshold,
            start_chunk=start_chunk,
        )
        print_chunk_dual_report(fixed_result, "fixed", args.pass_threshold)

    if run_sliding:
        sliding_result = sliding_dual_window(
            beat_iois, rhythm_iois, human_iois,
            args.human_part, args.noise_sd, args.pass_threshold,
            start_chunk=start_chunk,
        )
        print_chunk_dual_report(sliding_result, "sliding", args.pass_threshold)

    if args.plot:
        from visualizer_dual import plot_dual_all
        human_onsets_ms = silence_ms + _onsets_from_iois(human_iois)
        beat_ref_onsets_ms = _onsets_from_iois(beat_iois)
        rhythm_ref_onsets_ms = _onsets_from_iois(rhythm_iois)
        per_onset = fixed_result.get("per_onset_relational") if fixed_result else None
        plot_dual_all(
            beat_array=beat_array,
            rhythm_array=rhythm_array,
            sample_rate=sr,
            human_onsets_ms=human_onsets_ms,
            beat_ref_onsets_ms=beat_ref_onsets_ms,
            rhythm_ref_onsets_ms=rhythm_ref_onsets_ms,
            fixed_result=fixed_result,
            pass_threshold=args.pass_threshold,
            per_onset_relational=per_onset,
            noise_sd=args.noise_sd,
        )

    overall = result["overall_pass"]
    if fixed_result:
        overall = overall and (fixed_result["verdict"] == "PASS")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
