from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import soundfile as sf

from auto_mix.auto_mix import (
    LoadedStem,
    analyze_audio,
    analyze_stems,
    derive_mix_params,
    feature_distance,
    load_audio,
    optimize_params,
    render_mix,
    sum_mix,
    write_report,
)

from .storage import EXPORT_DIR, STEM_DIR


STEM_NAMES = ("vocals", "drums", "bass", "other")
SAMPLE_RATE = 44100


def render_reference_mix(
    track_id: str,
    reference_track: dict[str, Any],
    *,
    style: str = "auto",
    optimize: bool = True,
    optimize_seconds: float = 30.0,
    optimize_trials: int = 18,
) -> dict[str, Any]:
    stem_paths = _stem_paths(track_id)
    missing = [name for name, path in stem_paths.items() if not path.exists()]
    if missing:
        raise ValueError(f"Missing stems: {', '.join(missing)}")

    reference_path = Path(reference_track.get("path") or "")
    if not reference_path.exists():
        raise ValueError("Reference track audio file not found")

    reference_audio, sr = load_audio(reference_path, SAMPLE_RATE)
    stems = []
    for name, path in stem_paths.items():
        audio, stem_sr = load_audio(path, sr)
        stems.append(LoadedStem(name=name, path=path, audio=audio, sr=stem_sr))
    min_len = min(len(stem.audio) for stem in stems)
    stems = [LoadedStem(stem.name, stem.path, stem.audio[:min_len], stem.sr) for stem in stems]

    reference_features = analyze_audio(reference_audio, sr)
    stem_features = analyze_stems(stems, sr)
    raw_mix = sum_mix([stem.audio for stem in stems])
    raw_features = analyze_audio(raw_mix, sr)
    raw_distance = feature_distance(raw_features, reference_features)
    params = derive_mix_params(reference_features, stem_features, style=style)
    before_distance = None
    after_distance = None
    if optimize:
        params, before_distance, after_distance = optimize_params(
            stems,
            sr,
            params,
            reference_features,
            seconds=max(5.0, min(float(optimize_seconds), 45.0)),
            trials=max(1, min(int(optimize_trials), 36)),
        )

    mix = render_mix(stems, sr, params)
    final_features = analyze_audio(mix, sr)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stem = uuid.uuid4().hex
    wav_path = EXPORT_DIR / f"{stem}_reference_mix.wav"
    raw_path = EXPORT_DIR / f"{stem}_raw_stem_sum.wav"
    report_path = EXPORT_DIR / f"{stem}_reference_mix_report.json"
    sf.write(raw_path, raw_mix, sr, subtype="PCM_24")
    sf.write(wav_path, mix, sr, subtype="PCM_24")

    report = {
        "track_id": track_id,
        "reference_track_id": reference_track.get("id"),
        "reference_track_name": reference_track.get("name"),
        "style": style,
        "reference_features": reference_features,
        "raw_features": raw_features,
        "stem_features": stem_features,
        "derived_params": params,
        "frontend_mixer": to_frontend_mixer(params),
        "final_features": final_features,
        "feature_distance_before": raw_distance,
        "optimizer_preview_distance_before": before_distance,
        "feature_distance_after": after_distance
        if after_distance is not None
        else feature_distance(final_features, reference_features),
        "warnings": [
            "Reference-guided mix uses rule-based DSP parameters, not a trained model.",
            "Large files are currently rendered in memory.",
        ],
    }
    write_report(report_path, report)

    return {
        "url": f"/api/exports/{wav_path.name}",
        "rawUrl": f"/api/exports/{raw_path.name}",
        "reportUrl": f"/api/exports/{report_path.name}",
        "filename": wav_path.name,
        "rawFilename": raw_path.name,
        "reportFilename": report_path.name,
        "referenceTrackId": reference_track.get("id"),
        "referenceTrackName": reference_track.get("name"),
        "params": params,
        "mixer": report["frontend_mixer"],
        "referenceFeatures": reference_features,
        "beforeFeatures": raw_features,
        "finalFeatures": final_features,
        "featureDistanceBefore": raw_distance,
        "optimizerPreviewDistanceBefore": before_distance,
        "featureDistanceAfter": report["feature_distance_after"],
        "summary": _summary(reference_features, raw_features, final_features),
    }


def to_frontend_mixer(params: dict[str, Any]) -> dict[str, Any]:
    stems = {}
    for name in STEM_NAMES:
        item = (params.get("stems") or {}).get(name) or {}
        eq = item.get("eq") or {}
        comp = item.get("compressor") or {}
        stems[name] = {
            "gainDb": round(float(item.get("gain_db", 0.0)), 3),
            "pan": round((float(item.get("pan", 0.0)) + 1.0) / 2.0, 3),
            "eqDb": {
                "low": round(float(eq.get("low_shelf_db", 0.0)), 3),
                "mid": round(float(eq.get("presence_db", 0.0)), 3),
                "high": round(float(eq.get("high_shelf_db", 0.0)), 3),
            },
            "compressor": {
                "thresholdDb": round(float(comp.get("threshold_db", 0.0)), 3),
                "ratio": round(float(comp.get("ratio", 1.0)), 3),
                "attackMs": round(float(comp.get("attack_ms", 25.0)), 3),
                "releaseMs": round(float(comp.get("release_ms", 100.0)), 3),
                "makeupGainDb": 0.0,
            },
            "reverbSend": round(float(item.get("reverb_send", 0.0)), 3),
            "role": item.get("role") or name,
        }
    master = params.get("master") or {}
    master_comp = master.get("master_compressor") or {}
    return {
        "source": "reference-guided-auto-mix",
        "stems": stems,
        "master": {
            "gainDb": round(float(master.get("master_gain_db", 0.0)), 3),
            "eqDb": {
                "low": round(float(master.get("bass_shift_db", 0.0)), 3),
                "mid": 0.0,
                "high": round(float(master.get("brightness_shift_db", 0.0)), 3),
            },
            "compressor": {
                "thresholdDb": round(float(master_comp.get("threshold_db", 0.0)), 3),
                "ratio": round(float(master_comp.get("ratio", 1.0)), 3),
                "attackMs": round(float(master_comp.get("attack_ms", 25.0)), 3),
                "releaseMs": round(float(master_comp.get("release_ms", 100.0)), 3),
                "makeupGainDb": 0.0,
            },
            "targetLufs": round(float(master.get("target_lufs", -14.0)), 3),
            "stereoWidth": round(float(master.get("stereo_width", 1.0)), 3),
        },
    }


def _stem_paths(track_id: str) -> dict[str, Path]:
    return {stem: STEM_DIR / track_id / "demucs_api" / f"{stem}.wav" for stem in STEM_NAMES}


def _summary(reference_features: dict[str, Any], raw_features: dict[str, Any], final_features: dict[str, Any]) -> dict[str, float]:
    return {
        "referenceLufs": round(float(reference_features.get("integrated_lufs") or 0.0), 2),
        "beforeLufs": round(float(raw_features.get("integrated_lufs") or 0.0), 2),
        "finalLufs": round(float(final_features.get("integrated_lufs") or 0.0), 2),
        "referenceWidth": round(float(reference_features.get("stereo_width") or 0.0), 3),
        "beforeWidth": round(float(raw_features.get("stereo_width") or 0.0), 3),
        "finalWidth": round(float(final_features.get("stereo_width") or 0.0), 3),
        "referenceCrestDb": round(float(reference_features.get("crest_factor_db") or 0.0), 2),
        "beforeCrestDb": round(float(raw_features.get("crest_factor_db") or 0.0), 2),
        "finalCrestDb": round(float(final_features.get("crest_factor_db") or 0.0), 2),
    }
