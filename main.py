"""
main.py — Entry Point

CLI orchestrator for the rhythm analysis pipeline:
  reference → onset extraction → analysis → (optional) visualisation → verdict
"""

import argparse
import sys

from reference import build_reference
from extractor import extract_onsets
from analyzer import (
    analyze, print_report,
    fixed_chunk_accuracy, sliding_window_accuracy, print_chunk_report,
)
from visualizer import show_all_plots, plot_fixed_chunks, plot_sliding_window



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a human rhythm performance against a reference recording.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reference", required=True, metavar="WAV",
        help="Path to the reference .wav file.",
    )
    parser.add_argument(
        "--test", required=True, metavar="WAV",
        help="Path to the human-performance .wav file.",
    )
    parser.add_argument(
        "--iois", required=True, metavar="MS,...",
        help="Comma-separated list of reference inter-onset intervals in milliseconds. "
             "Example: 500,500,500,500",
    )
    parser.add_argument(
        "--noise_sd", type=float, default=20.0, metavar="MS",
        help="Tolerance standard deviation in ms. A beat passes if its IOI deviation "
             "is within ±2×noise_sd.",
    )
    parser.add_argument(
        "--ccf_threshold", type=float, default=0.85, metavar="FLOAT",
        help="Minimum CCF peak required to pass the cross-correlation check (0–1).",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Show all diagnostic plots after analysis.",
    )
    parser.add_argument(
        "--algorithm", choices=["fixed", "sliding", "both"], default="both",
        help="Chunk-accuracy algorithm(s) to run after the main analysis.",
    )
    parser.add_argument(
        "--pass_threshold", type=float, default=70.0, metavar="PCT",
        help="Minimum best-chunk/window accuracy (%%) required for a PASS verdict.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        #iois_ms = [float(x.strip()) for x in args.iois.split(",")]
        iois_ms = [398, 209, 197, 198, 205, 191, 411, 398, 209, 197, 198, 205, 191, 411, 398, 209, 197, 198, 205, 191, 411, 398, 209, 197, 198, 205, 191, 411]
    except ValueError:
        print("ERROR: --iois must be a comma-separated list of numbers, e.g. 500,500,500")
        return 1

    print(f"\nLoading reference: {args.reference}")
    try:
        ref_stimulus, ref_sequence, ref_iois, noise_sd = build_reference(
            wav_path=args.reference,
            iois_ms=iois_ms,
            noise_sd=args.noise_sd,
        )
    except Exception as exc:
        print(f"ERROR building reference: {exc}")
        return 1

    print(f"Extracting onsets from test: {args.test}")
    try:
        test_stimulus, test_sequence, test_iois = extract_onsets(wav_path=args.test)
    except Exception as exc:
        print(f"ERROR extracting test onsets: {exc}")
        return 1

    print(f"  Detected {len(test_iois) + 1} onsets → {len(test_iois)} IOIs")

    result = analyze(
        ref_sequence=ref_sequence,
        ref_iois=ref_iois,
        noise_sd=noise_sd,
        test_sequence=test_sequence,
        test_iois=test_iois,
        ccf_threshold=args.ccf_threshold,
    )

    print_report(result, noise_sd)

    # --- Chunk-accuracy algorithms ---
    import numpy as np
    ref_arr = np.array(ref_iois)
    test_arr = np.array(test_iois[:28] if len(test_iois) >= 28 else test_iois)

    if len(ref_arr) == 28 and len(test_arr) == 28:
        run_fixed = args.algorithm in ("fixed", "both")
        run_sliding = args.algorithm in ("sliding", "both")

        fixed_result = None
        sliding_result = None

        if run_fixed:
            fixed_result = fixed_chunk_accuracy(
                ref_arr, test_arr, noise_sd, pass_threshold=args.pass_threshold
            )
            print_chunk_report(fixed_result, "fixed", args.pass_threshold)

        if run_sliding:
            sliding_result = sliding_window_accuracy(
                ref_arr, test_arr, noise_sd, pass_threshold=args.pass_threshold
            )
            print_chunk_report(sliding_result, "sliding", args.pass_threshold)

        if args.plot:
            if fixed_result is not None:
                plot_fixed_chunks(fixed_result, pass_threshold=args.pass_threshold)
            if sliding_result is not None:
                plot_sliding_window(sliding_result, pass_threshold=args.pass_threshold)
    else:
        print(
            f"  (Skipping chunk algorithms: need exactly 28 reference and test IOIs, "
            f"got ref={len(ref_arr)}, test={len(test_arr)})"
        )

    if args.plot:
        show_all_plots(
            ref_stimulus=ref_stimulus,
            test_stimulus=test_stimulus,
            ref_sequence=ref_sequence,
            test_sequence=test_sequence,
            per_beat=result["per_beat"],
            noise_sd=noise_sd,
        )

    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
