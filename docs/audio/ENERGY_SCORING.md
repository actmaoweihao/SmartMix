# SmartMix Energy Scoring

## Why It Changed

The old `energy` value was based on average RMS and peak level:

```text
energy = min(1.0, avg_rms * 7 + peak * 1.5)
```

That made many modern mastered tracks display as 100. The new implementation treats energy as a multi-feature profile instead of a single loudness proxy.

## Implementation

- Backend analysis: `backend/analysis.py`, `_energy_metrics()`, `_energy_profile()`
- Pair matching: `backend/matching.py`, `energy_match_score()`
- Frontend fields: `track.energy`, `track.energy_profile`

## Energy Profile Fields

Each analyzed track now returns `energy_profile` with:

- `energy_index`
- `lufs`
- `true_peak_db`
- `rms_p10_db`, `rms_p50_db`, `rms_p85_db`, `rms_p95_db`
- `crest_factor_db`
- `low_frequency_ratio`
- `dynamic_range_db`
- `intro_relative_energy`
- `outro_relative_energy`
- `intro_delta_db`
- `outro_delta_db`
- `transition_contrast`

`energy` remains a 0-1 value for sorting and table display:

```text
energy = energy_index / 100
```

## Energy Index

```text
energy_index =
    0.30 * loudness
  + 0.25 * rms_body
  + 0.15 * crest_density
  + 0.15 * low_frequency
  + 0.10 * dynamic_motion
  + 0.05 * transition_contrast
```

Meaning:

- `loudness`: LUFS mapped to a usable 0-100 scale
- `rms_body`: RMS 85th percentile, representing body intensity
- `crest_density`: lower crest factor means denser, more compressed energy
- `low_frequency`: proportion of energy under 180Hz
- `dynamic_motion`: RMS P95-P10, representing energy movement
- `transition_contrast`: intro/outro energy difference from the full track

## Pair Match Energy Score

```text
energy_score =
    0.18 * energy_index_similarity
  + 0.18 * lufs_similarity
  + 0.14 * rms_body_similarity
  + 0.12 * crest_factor_similarity
  + 0.14 * low_frequency_similarity
  + 0.10 * dynamic_range_similarity
  + 0.14 * transition_shape_similarity
```

`transition_shape_similarity` is directional:

```text
previous track outro_relative_energy
vs
next track intro_relative_energy
```

So A -> B and B -> A can produce different energy scores.

## What You Should See

In the Pair Match panel, Energy now shows a summary like:

```text
index diff 31.3, LUFS diff 10.4, low diff 0.61
```

The API also returns detailed sub-scores:

```json
{
  "energy_index": 10.6,
  "lufs": 0.0,
  "rms_body": 12.9,
  "crest_factor": 59.1,
  "low_frequency": 0.0,
  "dynamic_range": 71.6,
  "transition_shape": 100.0
}
```

## Current Limitations

- LUFS uses `pyloudnorm` when available and falls back to RMS LUFS otherwise.
- Low-frequency ratio is spectrum-based, not stem-separated kick/bass detection.
- Browser fallback analysis cannot calculate a reliable low-frequency ratio yet, so backend analysis is preferred.

