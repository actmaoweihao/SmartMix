# Diff-MST Integration Notes

SmartMix borrows Diff-MST's console-parameter approach rather than loading the full training stack.

## What Diff-MST Contributes

- It treats mixing style transfer as prediction of interpretable mixing-console controls.
- Its advanced console includes per-track gain, pan, parametric EQ, compressor, reverb send, plus master-bus EQ/compression/fader controls.
- It trains with paired synthetic mixes or unpaired reference mixes, then compares rendered output with reference style using differentiable audio features such as RMS, crest factor, stereo width, stereo imbalance, and Bark-spectrum shape.
- The open repository exposes training and inference scripts, but checkpoints are local paths in the scripts rather than a packaged model artifact.

## SmartMix Adaptation

The current implementation maps the idea onto SmartMix's four Demucs stems:

- `vocals`
- `drums`
- `bass`
- `other`

The frontend offers Diff-MST-inspired profiles in the stem debugger. Applying a profile sets per-stem relative gain and enables hidden EQ/compressor/pan/master-bus parameters. During preview, Web Audio applies those parameters. During export, the backend uses cached Demucs stems and applies the same `stemMixer` payload before crossfading.

If cached stems are missing, export falls back to the existing full-track mix path.
