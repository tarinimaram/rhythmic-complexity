# Handoff — Rhythmic Complexity Project

## Goal

Build a CLI tool that measures how accurately a human performer matches a reference rhythm in a dual-channel (stereo) context. The reference WAV has the beat on the left channel and the rhythm on the right. The human records only one of those parts, and the tool scores their performance against both direct IOI matching and cross-channel relational timing accuracy.

---

## Architecture

| File | Role |
|---|---|
| `main_dual.py` | CLI entry point — parses args, orchestrates analysis, triggers plots |
| `channel_splitter.py` | Splits stereo reference WAV into left (beat) and right (rhythm) arrays |
| `dual_channel_analyzer.py` | Core accuracy engine: direct IOI check + cross-channel relational check |
| `chunk_dual_analyzer.py` | Applies both checks across fixed chunks (4×N) and a sliding window |
| `visualizer_dual.py` | All plotting — default graphs always shown, extra graphs behind `--plot` |
| `main.py` / `visualizer.py` | Older single-channel pipeline — not the active path, kept for reference |

### Key constants (chunk_dual_analyzer.py)
- Beat chunk size: 4 IOIs
- Rhythm chunk size: 7 IOIs
- Number of chunks: 4
- Human start chunk is inferred from silence before their first onset

---

## Current State (end of this session)

### What was changed this session

1. **`visualizer_dual.py`** — Replaced `plot_dual_all` with two functions:
   - `plot_default_graphs(...)` — stacked single-column figure: Onset Alignment → Relational Accuracy → Per-Onset Relational Deviation. Always runs.
   - `plot_extra_graphs(...)` — side-by-side: Reference Channels Overlaid + Direct IOI Accuracy. Only runs with `--plot`.

2. **`main_dual.py`** — Default graphs now always generated after analysis (no flag required). `--plot` triggers only the extra graphs. `split_channels` call hardcoded to `plot=False` so the stereo channel split waveform never displays.

### Graphs that always display (no flags)
- Onset Alignment (human vs. both reference channels on a timeline)
- Fixed Chunks — Relational Accuracy (bar chart, per chunk)
- Per-Onset Relational Deviation (bar chart with ±2σ tolerance band)

### Graphs only shown with `--plot`
- Reference Channels Overlaid (beat + rhythm waveforms)
- Fixed Chunks — Direct IOI Accuracy

### Graphs that no longer display at all
- Stereo channel split waveform (was in `channel_splitter._plot_channels`, now suppressed)

---

## How to Run

```bash
python main_dual.py \
  --reference "recordings/Recordings (rhythm + 12-8 beat R1).wav" \
  --test "recordings/R1 + 12:8 beat good example.wav" \
  --human_part beat \
  --beat_iois "500,500,500,500" \
  --rhythm_iois "398,209,197,198,205,191,411" \
  --noise_sd 30 \
  --pass_threshold 100
```

Add `--plot` to also show the extra diagnostic graphs.

---

## What Failed / Dead Ends

Nothing failed this session — both changes applied cleanly on the first attempt.

---

## Thebeat API Gotchas (from earlier sessions)

- `SoundSequence(sound=stimulus, sequence=seq)` — arg is `sound=`, not `sound_stimulus=`
- `SoundSequence` does NOT expose a `.sequence` attribute
- `ccf_df(test_seq, ref_seq, resolution)` — `resolution` (bin width ms) is required
- `ccf_df` returns `DataFrame` with `['timestamp', 'correlation']` columns
- `SoundStimulus.generate(freq=440, duration_ms=100)` — arg is `duration_ms`, not `duration`

---

## Suggested Next Steps

1. **Test the default graph output** end-to-end with one of the existing WAV pairs in `recordings/` to confirm layout looks right after the refactor.
2. **Tune chunk start inference** — the `start_chunk` logic uses silence before the first onset divided by phrase duration, clamped to [0, 3]. Worth verifying this works correctly with recordings where the human starts mid-sequence.
3. **Sliding window results** — `plot_extra_graphs` currently only shows fixed-chunk direct accuracy. If the sliding window output also needs visualization, that would be added there.
4. **`main.py` cleanup** — the single-channel pipeline in `main.py` has a hardcoded IOI list (line 68) overriding `--iois`. If that script is still in use, that should be fixed or removed.
