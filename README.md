# Rhythmic Complexity Analyzer

Compares a human rhythm performance against a reference recording and gives a **PASS/FAIL** verdict using two complementary checks:

1. **Per-beat IOI deviation** — each inter-onset interval must fall within ±2σ of the reference.
2. **Cross-correlation (CCF)** — the peak CCF between the sequences must meet a configurable threshold.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py \
  --reference "Recordings (rhythm alone R1).wav" \
  --test "R1 + 12:8 beat good example.wav" \
  --iois 500,500,500,500,500,500,500,500 \
  --noise_sd 30 \
  --ccf_threshold 0.80 \
  --plot
```

### CLI arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--reference` | yes | — | Path to the reference `.wav` file |
| `--test` | yes | — | Path to the human-performance `.wav` file |
| `--iois` | yes | — | Comma-separated reference IOIs in ms (e.g. `500,500,500,500`) |
| `--noise_sd` | no | `20.0` | Tolerance SD in ms; a beat passes if deviation ≤ 2×noise_sd |
| `--ccf_threshold` | no | `0.85` | Minimum CCF peak for the correlation check to pass (0–1) |
| `--plot` | no | off | Show all four diagnostic plots |

---

## File Overview

| File | Role |
|---|---|
| `reference.py` | Loads a reference WAV + known IOIs → `SoundSequence` |
| `extractor.py` | Detects onsets in a human WAV via librosa → IOIs → `SoundSequence` |
| `analyzer.py` | Runs IOI deviation + CCF checks, returns structured result dict |
| `visualizer.py` | Four diagnostic plots (waveforms, sequences, CCF, IOI deviations) |
| `main.py` | CLI entry point orchestrating the full pipeline |
| `tests/test_pipeline.py` | Pytest suite using synthetic sequences |

---

## Running Tests

```bash
cd "rhythmic complexity project"
pytest tests/ -v
```

---

## Notes on IOI values

If you don't know the exact IOIs of your reference recording, you can estimate them by running the reference through the extractor:

```python
from extractor import extract_onsets
_, iois = extract_onsets("your_reference.wav")
print(iois)  # use these as your --iois argument
```
