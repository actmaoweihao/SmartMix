# Smart Beat Handoff Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a professional Auto-DJ handoff flow that analyzes uploaded tracks, sorts them, and produces stem-aware beat handoff plans for smooth transitions across electronic, pop, rap, live, and slow songs.

**Architecture:** Use the academic Auto-DJ pattern as the backbone: offline analysis, transition planning, then audio rendering. The first implementation adds a deterministic backend planner that scores ordering and cue candidates, then feeds the existing timeline, teaching preview, seamless renderer, and export paths. Later phases upgrade cue detection and stem rendering without changing the public plan contract.

**Tech Stack:** FastAPI, Python, librosa-derived analysis metadata, existing Demucs stem cache, Vite frontend, Web Audio preview, existing `/api/transition-preview` and `/api/export` renderers.

---

## Product Definition

Smart Beat Handoff is not a low-bass crossfade. It is an Auto-DJ/Mashup transition system where each adjacent pair receives a concrete handoff plan:

- which song order should be used
- where A exits and B enters
- how many bars the handoff should last
- which rhythm bed should lead the transition
- whether to use drums, bass swap, vocal-safe bridge, or effect bridge
- what risks exist for pop/live/rap/slow tracks with unstable beat grids

The desired audible result is a shared transition groove: the listener should hear one coherent rhythmic bed through the handoff, with bass and vocals controlled so the transition does not feel like two unrelated songs overlapping.

## Research Backbone

The planning model follows the Auto-DJ system described in "From raw audio to a seamless mix": analyze beat/downbeat/structure/key/style, then choose tracks, cue points, time-stretching, beatmatching, and transition style. This project extends that idea beyond drum and bass by scoring beat-grid reliability and falling back to stem or bridge-based transitions when full beatmatching is unsafe.

Automatic cue detection follows DJ switch point research: good cues tend to sit at structural boundaries, novelty peaks, low vocal conflict areas, or stable groove entries. Later work such as CUE-DETR suggests a future ML path, but v1 should remain deterministic and explainable.

Stem-based transitions use the existing Demucs direction: drums, bass, vocals, and other are treated separately. The handoff should normally avoid two full-mix drum/bass/vocal stacks at the same time.

Useful references:

- Auto-DJ system: https://backoffice.biblio.ugent.be/download/8577611/8577612
- Automatic DJ cue/switch points: https://arxiv.org/abs/2007.08411
- CUE-DETR cue estimation: https://arxiv.org/abs/2407.06823
- Demucs source separation: https://github.com/facebookresearch/demucs
- librosa dynamic beat tracking: https://librosa.org/doc/main/auto_examples/plot_dynamic_beat.html

## v1 Data Contract

Create a backend endpoint:

```http
POST /api/auto-handoff/plan
```

Request:

```json
{
  "trackIds": ["id-a", "id-b"],
  "tracks": [{ "id": "id-a", "introPoint": 12.0, "outroPoint": 185.0 }],
  "settings": {
    "phraseBars": 8,
    "maxTempoChangePercent": 10,
    "preferStems": true,
    "targetEnergy": "arc"
  }
}
```

Response:

```json
{
  "orderedTrackIds": ["id-a", "id-b"],
  "score": 82.4,
  "transitions": [
    {
      "fromTrackId": "id-a",
      "toTrackId": "id-b",
      "type": "drum_bed_handoff",
      "score": 84.2,
      "barCount": 8,
      "durationSec": 16.0,
      "outgoingCue": { "time": 176.0, "role": "mix_out" },
      "incomingCue": { "time": 16.0, "role": "mix_in" },
      "rhythmBed": { "source": "A", "stem": "drums" },
      "automation": {
        "bassSwapAt": 0.55,
        "vocalDuck": true,
        "sharedDrumBed": true
      },
      "risk": "medium",
      "warnings": ["local beat grid only"],
      "explanation": "Use A drums as the rhythm bed..."
    }
  ]
}
```

## Transition Types

- `drum_bed_handoff`: A or B drums create the common rhythmic bed. Best for electronic, pop, rap, and locally stable songs.
- `bass_swap_handoff`: Similar tempo and strong low end; one bass owns the transition until the swap point.
- `percussive_loop_bridge`: For unstable/live/slow tracks; loop a stable drum/percussion area as a bridge.
- `vocal_safe_bridge`: Avoid long vocal overlap. Use other/drums as the bed and duck one vocal.
- `effect_tail_handoff`: Fallback for poor beat grid or large BPM/key mismatch. Use echo/filter/reverb tail instead of long beatmix.

## Cue Candidate Scoring

For every track, derive cue candidates from existing analysis:

- transition candidates intro/outro
- section boundaries
- bar/phrase grid
- energy curve
- vocal density curve
- drum and bass proxies from energy/low-frequency analysis

Score candidate cues with:

```text
cueScore =
  0.28 * phraseAlignment
+ 0.22 * vocalSafety
+ 0.18 * grooveStability
+ 0.14 * energyFit
+ 0.10 * structuralBoundary
+ 0.08 * durationRoom
```

For beat-grid unstable songs, lower `grooveStability` and prefer shorter bridges.

## Pair Scoring

For each directed pair:

```text
pairScore =
  0.24 * tempoCompatibility
+ 0.18 * cueCompatibility
+ 0.16 * rhythmBedQuality
+ 0.14 * vocalSafety
+ 0.12 * bassSafety
+ 0.10 * harmonicCompatibility
+ 0.06 * energyFlow
```

This intentionally shifts weight away from pure key/BPM similarity and toward whether the transition can actually be rendered cleanly.

## Task 1: Backend Planner

**Files:**
- Create: `backend/auto_handoff.py`
- Modify: `backend/main.py`

**Steps:**
1. Add `build_auto_handoff_plan(tracks, settings)`.
2. Compute cue candidates using existing `transition_candidates`, `bars`, `phrases`, and `sections`.
3. Score directed pairs and choose a greedy order.
4. Produce transition specs with type, cues, bar count, rhythm bed, automation, risks, warnings, and explanation.
5. Add `POST /api/auto-handoff/plan`.
6. Verify with `python -m compileall backend`.

## Task 2: Frontend Flow

**Files:**
- Modify: `src/main.js`
- Modify: `src/styles.css`

**Steps:**
1. Add `state.autoHandoff`.
2. Add a "Smart Beat Handoff" button and result panel.
3. Call `/api/auto-handoff/plan` with ready tracks.
4. Reorder tracks according to `orderedTrackIds`.
5. Apply outgoing/incoming cues and transition settings to the timeline.
6. Render transition cards with type, score, rhythm bed, cues, risk, and warnings.
7. Verify with `pnpm check`.

## Cue Detection v1.5 Upgrade

The first cue upgrade is deterministic and runs inside `backend/analysis.py` during upload analysis. It produces `transition_candidates.cue_candidates`, a ranked list of cue objects with:

- `role`: `mix_in`, `mix_out`, `drop`, `bridge`, `drum_loop`, or `vocal_safe`
- `score`: 0-100 suitability score
- `source`: detector or fallback source
- `components`: phrase alignment, vocal safety, groove stability, novelty, drum proxy, section boundary, and duration room
- `reasons`: short human-readable evidence such as `phrase aligned`, `low vocal conflict`, `stable local groove`, and `section boundary`

The detector uses existing bar features, but annotates each bar with novelty, local energy stability, vocal safety, drum proxy, and phrase alignment. `backend/auto_handoff.py` now prefers these analyzed cue candidates before falling back to section boundaries and raw bar anchors.

This makes Smart Beat Handoff less dependent on one guessed intro/outro point. Pop, live, rap, and slow tracks can now produce safer short-bridge or vocal-safe cues even when the full beat grid is not reliable enough for long beatmatching.

## CUE-DETR Optional ML Cue Provider

CUE-DETR is integrated as an optional local model provider in `backend/cue_detr.py`. It follows the public example pipeline from the CUE-DETR repository:

1. Load audio at 22050 Hz.
2. Convert it to a mel spectrogram.
3. Render the spectrogram as RGB image windows.
4. Run `DetrForObjectDetection.from_pretrained("disco-eth/cue-detr")`.
5. Convert object centers back to spectrogram frame times.
6. Merge predicted cues with the deterministic cue detector.

Install:

```bash
python -m pip install -r backend/requirements-cue-detr.txt
```

Enable:

```powershell
$env:SMARTMIX_ENABLE_CUE_DETR="1"
pnpm backend
```

The integration is deliberately optional because it downloads large Hugging Face assets and requires `transformers`. When disabled or unavailable, SmartMix continues using the deterministic cue detector.

## Real Handoff Audio Rendering

Smart Beat Handoff now separates planning from rendering. The first click generates ordering, cue points, transition type, and automation. The new "Render Real Handoff Audio" action then calls `/api/transition-preview` for each adjacent pair with stem separation enabled. Rendered previews are written back to the incoming track as `appliedTransitionPreview`, so browser preview and final export can reuse the same transition audio instead of falling back to a simple fade-like overlap.

## Task 3: Rendering Upgrade Follow-up

**Files:**
- Modify later: `backend/seamless.py`
- Modify later: `backend/mixing.py`

**Steps:**
1. Extend `/api/transition-preview` to accept `BeatHandoffPlan`.
2. Render drums/bass/vocals/other stems according to the selected transition type.
3. Add percussive loop bridge extraction for unstable beat grids.
4. Persist rendered previews on each incoming track.
5. Ensure final export reuses the rendered transition preview.

## Acceptance Criteria for v1

- Uploading at least two analyzed songs enables Smart Beat Handoff.
- Clicking the button returns a backend-generated order and transition list.
- The UI applies the order and cue points.
- The timeline updates to reflect the new overlap points.
- The transition panel explains why each handoff was selected.
- Backend and frontend syntax checks pass.
