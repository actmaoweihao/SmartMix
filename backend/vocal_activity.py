from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


SAMPLE_RATE = 44100


def analyze_vocal_stem(path: str | Path, duration: float | None = None, sr: int = SAMPLE_RATE) -> dict[str, Any]:
    audio, source_sr = _load_mono(Path(path), sr)
    if audio.size == 0:
        return _empty_report("empty")

    total_duration = float(duration or librosa.get_duration(y=audio, sr=source_sr))
    hop = 1024
    frame = 2048
    rms = librosa.feature.rms(y=audio, frame_length=frame, hop_length=hop)[0]
    if rms.size == 0 or float(np.max(rms)) <= 1e-9:
        return _empty_report("silent", total_duration)

    energy = _smooth(_normalize(rms), 9)
    threshold = _adaptive_threshold(energy)
    active = energy >= threshold
    times = librosa.frames_to_time(np.arange(len(energy)), sr=source_sr, hop_length=hop)
    active = _fill_short_gaps(active, times, max_gap=0.45)
    active = _remove_short_regions(active, times, min_duration=0.36)

    regions = _regions_from_activity(times, energy, active, total_duration)
    entries = _entry_points(regions)
    releases = _release_points(regions, total_duration)
    curve = _activity_curve(regions, total_duration)
    coverage = sum(region["duration"] for region in regions) / max(total_duration, 1e-6)
    return {
        "source": "demucs_vocals",
        "stemsUsed": True,
        "duration": round(total_duration, 3),
        "threshold": round(float(threshold), 4),
        "coverage": round(float(max(0.0, min(1.0, coverage))), 4),
        "regionCount": len(regions),
        "regions": regions[:96],
        "entryPoints": entries[:96],
        "releasePoints": releases[:96],
        "activityCurve": curve,
    }


def merge_vocal_activity_into_analysis(analysis: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    transition_candidates = dict(analysis.get("transition_candidates") or {})
    transition_candidates["vocal_activity"] = report
    merged = {
        **analysis,
        "transition_candidates": transition_candidates,
        "vocal_activity": report,
        "vocal_regions": report.get("regions") or [],
        "vocal_release_points": report.get("releasePoints") or [],
        "vocal_entry_points": report.get("entryPoints") or [],
    }
    if report.get("activityCurve"):
        curve = [{"time": item["time"], "density": item["activity"]} for item in report["activityCurve"]]
        merged["vocal_density_curve"] = curve
        transition_candidates["vocal_density_curve"] = curve
    return merged


def _load_mono(path: Path, sr: int) -> tuple[np.ndarray, int]:
    try:
        audio, source_sr = sf.read(path, always_2d=True, dtype="float32")
        mono = np.mean(audio, axis=1)
        if source_sr != sr:
            mono = librosa.resample(mono, orig_sr=source_sr, target_sr=sr)
            source_sr = sr
        return np.asarray(mono, dtype=np.float32), int(source_sr)
    except Exception:
        audio, source_sr = librosa.load(path, sr=sr, mono=True)
        return np.asarray(audio, dtype=np.float32), int(source_sr)


def _empty_report(reason: str, duration: float = 0.0) -> dict[str, Any]:
    return {
        "source": "demucs_vocals",
        "stemsUsed": False,
        "reason": reason,
        "duration": round(float(duration), 3),
        "threshold": 0.0,
        "coverage": 0.0,
        "regionCount": 0,
        "regions": [],
        "entryPoints": [],
        "releasePoints": [],
        "activityCurve": [],
    }


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low = float(np.percentile(values, 12))
    high = float(np.percentile(values, 97))
    if high <= low + 1e-9:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < 3:
        return values
    width = max(3, min(window, values.size))
    kernel = np.ones(width, dtype=np.float32) / width
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def _adaptive_threshold(energy: np.ndarray) -> float:
    mean = float(np.mean(energy))
    p55 = float(np.percentile(energy, 55))
    p75 = float(np.percentile(energy, 75))
    return max(0.045, min(0.62, p55 * 0.64 + mean * 0.20 + p75 * 0.16))


def _fill_short_gaps(active: np.ndarray, times: np.ndarray, max_gap: float) -> np.ndarray:
    result = active.copy()
    spans = _boolean_spans(~result)
    for start_idx, end_idx in spans:
        if start_idx == 0 or end_idx >= len(result):
            continue
        duration = float(times[end_idx - 1] - times[start_idx]) if end_idx > start_idx else 0.0
        if duration <= max_gap:
            result[start_idx:end_idx] = True
    return result


def _remove_short_regions(active: np.ndarray, times: np.ndarray, min_duration: float) -> np.ndarray:
    result = active.copy()
    for start_idx, end_idx in _boolean_spans(result):
        duration = float(times[end_idx - 1] - times[start_idx]) if end_idx > start_idx else 0.0
        if duration < min_duration:
            result[start_idx:end_idx] = False
    return result


def _boolean_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if bool(value) and start is None:
            start = index
        elif not bool(value) and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def _regions_from_activity(times: np.ndarray, energy: np.ndarray, active: np.ndarray, duration: float) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    frame_step = float(np.median(np.diff(times))) if len(times) > 1 else 0.023
    for start_idx, end_idx in _boolean_spans(active):
        start = max(0.0, float(times[start_idx]) - frame_step * 0.5)
        end = min(duration, float(times[min(end_idx, len(times) - 1)]) + frame_step * 0.5)
        if end <= start:
            continue
        local = energy[start_idx:end_idx]
        regions.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "avgActivity": round(float(np.mean(local)) if local.size else 0.0, 3),
                "peakActivity": round(float(np.max(local)) if local.size else 0.0, 3),
            }
        )
    return _merge_nearby_regions(regions, max_gap=0.38)


def _merge_nearby_regions(regions: list[dict[str, Any]], max_gap: float) -> list[dict[str, Any]]:
    if not regions:
        return []
    merged = [dict(regions[0])]
    for region in regions[1:]:
        current = merged[-1]
        if float(region["start"]) - float(current["end"]) <= max_gap:
            current_duration = float(current["duration"])
            region_duration = float(region["duration"])
            total = current_duration + region_duration
            current["end"] = region["end"]
            current["duration"] = round(float(current["end"]) - float(current["start"]), 3)
            current["avgActivity"] = round((float(current["avgActivity"]) * current_duration + float(region["avgActivity"]) * region_duration) / max(total, 1e-6), 3)
            current["peakActivity"] = max(float(current["peakActivity"]), float(region["peakActivity"]))
        else:
            merged.append(dict(region))
    return merged


def _entry_points(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for region in regions:
        result.append({"time": region["start"], "confidence": region["peakActivity"], "regionEnd": region["end"]})
    return result


def _release_points(regions: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    result = []
    for index, region in enumerate(regions):
        next_start = float(regions[index + 1]["start"]) if index + 1 < len(regions) else duration
        gap = max(0.0, next_start - float(region["end"]))
        confidence = min(1.0, 0.35 + gap / 8.0 + float(region.get("peakActivity", 0.0)) * 0.25)
        result.append({"time": region["end"], "confidence": round(confidence, 3), "nextEntryInSec": round(gap, 3)})
    return result


def _activity_curve(regions: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    if duration <= 0:
        return []
    step = 0.5
    count = int(math.ceil(duration / step)) + 1
    points = []
    for index in range(count):
        time_value = min(duration, index * step)
        activity = _activity_at(regions, time_value, step)
        points.append({"time": round(float(time_value), 3), "activity": round(activity, 3)})
    return points


def _activity_at(regions: list[dict[str, Any]], time_value: float, window: float) -> float:
    start = time_value
    end = time_value + window
    active = 0.0
    peak = 0.0
    for region in regions:
        overlap = max(0.0, min(end, float(region["end"])) - max(start, float(region["start"])))
        if overlap <= 0:
            continue
        active += overlap / max(window, 1e-6)
        peak = max(peak, float(region.get("peakActivity", 0.0)))
    return max(0.0, min(1.0, active * 0.72 + peak * 0.28))
