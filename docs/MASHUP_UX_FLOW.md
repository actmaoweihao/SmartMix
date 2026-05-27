# Mashup Builder UX Flow

## Product Intent

Mashup Builder should feel like a guided music production workflow, not a pile
of DSP switches. The default path is Groove Vocal Handoff: one stable
drums/bass/other bed plus alternating vocal phrases from Song A and Song B.
Legacy section-to-section modes remain available for comparison, but they are
secondary.

## Primary User Flow

1. Upload and analyze at least two tracks.
2. Open Mashup Builder and choose Song A / Song B.
3. Keep the default mode, `groove_vocal_handoff`, unless the user intentionally
   wants a legacy section join.
4. Keep `useStems` enabled. If Demucs stems are missing, the system must say
   that clean groove mashup is unavailable instead of pretending full mixes are
   separated.
5. Click Analyze.
   The UI should show sections, minor sections, vocal phrases, groove bed
   candidates, safe cut points, and warnings.
6. Click Build Plan.
   The UI should show the chosen bed, vocal phrase timeline, score, warnings,
   and alternatives.
7. Click Render.
   The UI should show an audio player, download link, rendered layers, loudness,
   peak, duration, and render warnings.

## State Model

- Pick songs: blocked until two analyzed tracks are available.
- Analyze: extracts structure and candidate material.
- Build plan: chooses bed, vocal phrases, transitions, stretch/pitch policy.
- Render: creates the final WAV/MP3 preview and download.

Each state must expose what is missing next. Do not make the user infer whether
they need stems, analysis, a plan, or a render.

## Controls

Default-visible controls should be limited to:

- Song A
- Song B
- Mode
- Segment length for legacy modes
- useStems
- Vocal priority
- Groove bed preference
- allow hybrid bed
- allow vocal pitch
- max vocal stretch

Removed from the primary UI:

- Transition strictness
- Stem usage
- Energy curve

Those options were either not materially changing the groove result or were too
abstract for the current workflow. They can stay as backend-compatible optional
API fields, but they should not distract the first-screen experience.

## Quality Expectations

The result should not have long overlaps of two dry lead vocals. A phrase should
not be truncated just because its nominal bar window ended. If the system cannot
protect vocal phrases or cannot access stems, it must show a warning and lower
confidence rather than producing a misleading clean-sounding label.

## Failure States

- Missing stems: show `stems_required` or a fallback warning.
- Weak vocal phrase extraction: show phrase risk flags and avoid groove mode in
  auto if usable phrases are unavailable.
- Poor groove bed: show bed leakage/loopability warnings.
- Unsafe stretch/pitch: show per-event warnings and skip high-risk vocal events
  in conservative/balanced behavior.

## Recommended Next UX Improvements

- Add per-candidate preview buttons for vocal phrases and groove beds.
- Add a "Use this bed" and "Use this phrase" manual override.
- Move segmentation debug into a collapsible inspector after the default flow is
  stable.
- Add a short post-render checklist: "No long vocal overlap", "Bed uses stems",
  "Peak safe", "LUFS normalized".
