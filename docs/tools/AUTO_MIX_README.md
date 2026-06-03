# Reference-Guided Auto Mix MVP

This folder contains a Python 3.10+ command-line prototype for reference-track
style automatic mixing. It is inspired by Diff-MST's high-level idea:
use reference audio features to steer interpretable mixing-console parameters.

It does not train a neural network, download a model, or require a GPU. The MVP
is a rule system plus DSP effects and a lightweight optional parameter search.

## What It Does

Given a folder of stems and one reference track, the tool:

1. analyzes the reference track for loudness, crest factor, spectrum, stereo
   width, and imbalance;
2. analyzes every stem and infers a simple role from the filename;
3. derives gain, pan, EQ, compression, reverb send, and master settings;
4. optionally searches a small parameter grid on a 20-30 second preview;
5. renders a stereo WAV;
6. writes `mix_report.json` with all features, parameters, warnings, and final
   loudness.

## Install

```bash
cd auto_mix
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`pedalboard` and `pyloudnorm` are recommended. If they are unavailable, the
script falls back to simpler NumPy/SciPy DSP and approximate loudness.

## Usage

```bash
python auto_mix.py \
  --stems ./stems \
  --reference ./reference.wav \
  --out ./output/mix.wav \
  --style auto
```

The report is written next to the WAV by default:

```text
./output/mix_report.json
```

Disable the preview optimizer when you want a faster rule-only render:

```bash
python auto_mix.py --stems ./stems --reference ./reference.wav --out ./output/mix.wav --no-optimize
```

## Stem Role Inference

Filenames are mapped as follows:

- `vocal`, `vox`, `sing` -> `vocal`
- `drums`, `drum`, `beat`, `kick`, `snare` -> `drums`
- `bass` -> `bass`
- `guitar`, `gtr` -> `guitar`
- `keys`, `piano`, `synth`, `pad` -> `harmonic`
- anything else -> `other`

Unknown stems are not errors; they are mixed conservatively as `other`.

## Output Report

`mix_report.json` includes:

- `reference_features`
- `stem_features`
- `derived_params`
- `final_features`
- `feature_distance_before`
- `feature_distance_after`
- `warnings`

## Tests

```bash
python -m unittest discover -s tests -v
```

## Notes and TODO

- This is an MVP, not a replacement for a trained model or a human mix engineer.
- The optimizer uses a compact random/grid search over master gain, compression,
  stereo width, bass shift, and brightness shift.
- Large files are processed in memory for simplicity. TODO: add chunked stem
  processing and streaming loudness analysis for long sessions.
- Low-frequency stereo widening is intentionally avoided by keeping bass and
  drums centered and applying width mostly at the master side signal level.
