from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
from scipy import signal

from .matching import camelot_key_distance


SAMPLE_RATE = 44100
BAND_NAMES = ("sub", "bass", "low_mid", "mid", "high_mid", "high")


def build_track_segments(
    track: dict[str, Any],
    source: str,
    bars_per_segment: int = 16,
    *,
    audio: np.ndarray | None = None,
    sr: int = SAMPLE_RATE,
) -> list[dict[str, Any]]:
    duration = float(track.get("duration") or 0.0)
    if duration <= 0:
        return []
    bpm = float(track.get("bpm") or 120.0)
    bars = normalized_bars(track, duration, bpm)
    if len(bars) < 2:
        return []

    bars_per_segment = int(np.clip(int(bars_per_segment or 16), 4, 32))
    starts = _segment_start_indices(track, bars, bars_per_segment)
    segments: list[dict[str, Any]] = []
    for index, bar_start in enumerate(starts):
        bar_end = min(bar_start + bars_per_segment, len(bars) - 1)
        if bar_end <= bar_start:
            continue
        start = float(bars[bar_start])
        end = float(bars[bar_end])
        if end - start < 2.0:
            continue
        metrics = segment_metrics(track, start, end, audio=audio, sr=sr)
        label = label_segment(metrics, start, end, duration)
        segments.append(
            {
                "id": f"{source}_seg_{len(segments) + 1:03d}",
                "trackId": track.get("id"),
                "source": source,
                "start": round(start, 3),
                "end": round(min(end, duration), 3),
                "barStart": int(bar_start),
                "barEnd": int(bar_end),
                "label": label,
                "energy": round(metrics["energy"], 3),
                "vocalDensity": round(metrics["vocalDensity"], 3),
                "bassEnergy": round(metrics["bassEnergy"], 3),
                "drumActivity": round(metrics["drumActivity"], 3),
                "brightness": round(metrics["brightness"], 3),
                "bpm": round(bpm, 3),
                "camelot": track.get("camelot"),
                "key": track.get("key"),
                "mixInScore": round(mix_boundary_score(track, start, metrics), 3),
                "mixOutScore": round(mix_boundary_score(track, end, metrics), 3),
            }
        )
    return segments


def normalized_bars(track: dict[str, Any], duration: float, bpm: float) -> list[float]:
    raw = [float(value) for value in (track.get("bars") or []) if _is_number(value)]
    raw = sorted(value for value in raw if 0 <= value < duration)
    if not raw:
        bar_seconds = 240.0 / max(float(bpm or 120.0), 1.0)
        raw = list(np.arange(0.0, duration, bar_seconds))
    if not raw or raw[0] > 0.05:
        raw.insert(0, 0.0)
    if raw[-1] < duration:
        raw.append(duration)
    cleaned = [raw[0]]
    for value in raw[1:]:
        if value - cleaned[-1] >= 0.25:
            cleaned.append(min(float(value), duration))
    return cleaned


def segment_metrics(
    track: dict[str, Any],
    start: float,
    end: float,
    *,
    audio: np.ndarray | None = None,
    sr: int = SAMPLE_RATE,
) -> dict[str, float]:
    energy = _curve_average(track.get("energy_curve") or [], "energy", start, end, fallback=track.get("energy", 0.45))
    vocal = _curve_average(track.get("vocal_density_curve") or [], "density", start, end, fallback=_section_vocal_hint(track, start, end))
    bass = 0.35 + energy * 0.35
    drums = 0.35 + energy * 0.35
    brightness = 0.45
    if audio is not None and audio.size:
        mono = _segment_mono(audio, start, end, sr)
        if mono.size:
            bands = band_energy(mono, sr)
            bass = _clamp01(bands["sub"] * 1.2 + bands["bass"] * 1.4)
            brightness = _clamp01(bands["high_mid"] * 1.2 + bands["high"] * 1.8)
            drums = drum_activity(mono, sr)
            rms = float(np.sqrt(np.mean(mono * mono) + 1e-9))
            energy = _clamp01(max(energy, min(1.0, rms * 8.0)))
    return {
        "energy": _clamp01(energy),
        "vocalDensity": _clamp01(vocal),
        "bassEnergy": _clamp01(bass),
        "drumActivity": _clamp01(drums),
        "brightness": _clamp01(brightness),
    }


def band_energy(mono: np.ndarray, sr: int) -> dict[str, float]:
    if mono.size < 64:
        return {name: 0.0 for name in BAND_NAMES}
    nperseg = min(4096, max(256, 2 ** int(np.floor(np.log2(mono.size)))))
    freqs, _, zxx = signal.stft(mono, fs=sr, nperseg=nperseg, noverlap=nperseg // 2, boundary=None, padded=False)
    power = np.mean(np.abs(zxx) ** 2, axis=1)
    total = float(np.sum(power[(freqs >= 20) & (freqs <= min(16000, sr / 2))]) + 1e-9)
    ranges = {
        "sub": (20, 60),
        "bass": (60, 250),
        "low_mid": (250, 500),
        "mid": (500, 2000),
        "high_mid": (2000, 6000),
        "high": (6000, 16000),
    }
    return {
        name: float(np.sum(power[(freqs >= low) & (freqs < min(high, sr / 2))]) / total)
        for name, (low, high) in ranges.items()
    }


def drum_activity(mono: np.ndarray, sr: int) -> float:
    if mono.size < 128:
        return 0.0
    onset = librosa.onset.onset_strength(y=mono, sr=sr)
    if onset.size == 0:
        return 0.0
    normalized = onset / (float(np.max(onset)) + 1e-9)
    return _clamp01(float(np.mean(normalized) * 1.8))


def label_segment(metrics: dict[str, float], start: float, end: float, duration: float) -> str:
    position = start / max(duration, 1e-6)
    energy = metrics["energy"]
    vocal = metrics["vocalDensity"]
    drums = metrics["drumActivity"]
    if position < 0.14 and energy < 0.55:
        return "intro_like"
    if position > 0.82:
        return "outro_like"
    if energy < 0.28 and vocal < 0.42:
        return "breakdown_like"
    if energy > 0.68 and drums > 0.45:
        return "drop_like"
    if vocal > 0.45 and energy > 0.48:
        return "chorus_like"
    return "verse_like"


def segment_compatibility(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    bpm = bpm_compatibility(float(left.get("bpm") or 0), float(right.get("bpm") or 0))
    camelot_eval = camelot_key_distance(left.get("camelot"), right.get("camelot"))
    camelot = _clamp01(float(camelot_eval.get("score") or 40.0) / 100.0)
    energy_flow = _clamp01(1.0 - abs(float(right.get("energy", 0)) - float(left.get("energy", 0))) * 1.35)
    vocal_conflict = _clamp01(1.0 - min(1.0, float(left.get("vocalDensity", 0)) * float(right.get("vocalDensity", 0)) * 1.7))
    timbre = timbre_similarity(left, right)
    score = 0.30 * bpm + 0.25 * camelot + 0.20 * energy_flow + 0.15 * vocal_conflict + 0.10 * timbre
    return {
        "score": round(score, 4),
        "score100": round(score * 100, 1),
        "components": {
            "bpmCompatibility": round(bpm, 4),
            "camelotCompatibility": round(camelot, 4),
            "energyFlow": round(energy_flow, 4),
            "vocalConflictAvoidance": round(vocal_conflict, 4),
            "timbreSimilarity": round(timbre, 4),
        },
        "bpmCompatibility": round(bpm, 4),
        "camelotCompatibility": round(camelot, 4),
        "energyFlow": round(energy_flow, 4),
        "vocalConflictAvoidance": round(vocal_conflict, 4),
        "timbreSimilarity": round(timbre, 4),
        "camelotRelation": camelot_eval.get("relation"),
    }


def bpm_compatibility(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.55
    ratio = min(a, b) / max(a, b)
    if ratio < 0.88:
        return _clamp01((ratio - 0.72) / 0.16 * 0.6)
    return _clamp01(1.0 - abs(1.0 - ratio) / 0.12 * 0.35)


def timbre_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    diffs = [
        abs(float(left.get("bassEnergy", 0)) - float(right.get("bassEnergy", 0))),
        abs(float(left.get("drumActivity", 0)) - float(right.get("drumActivity", 0))),
        abs(float(left.get("brightness", 0)) - float(right.get("brightness", 0))),
    ]
    return _clamp01(1.0 - float(np.mean(diffs)))


def best_compatible(source: dict[str, Any], candidates: list[dict[str, Any]], labels: set[str] | None = None) -> dict[str, Any] | None:
    pool = [item for item in candidates if not labels or item.get("label") in labels]
    if not pool:
        pool = candidates
    if not pool:
        return None
    return max(pool, key=lambda item: segment_compatibility(source, item)["score"] + float(item.get("mixInScore", 0)) * 0.08)


def _segment_start_indices(track: dict[str, Any], bars: list[float], bars_per_segment: int) -> list[int]:
    phrases = [float(value) for value in (track.get("phrases") or []) if _is_number(value)]
    phrase_indices = []
    for phrase in phrases:
        index = min(range(len(bars)), key=lambda item: abs(bars[item] - phrase))
        if index < len(bars) - 1:
            phrase_indices.append(index)
    if phrase_indices:
        starts = sorted(set(index for index in phrase_indices if index % max(1, bars_per_segment // 2) == 0 or True))
    else:
        starts = list(range(0, len(bars) - 1, bars_per_segment))
    if 0 not in starts:
        starts.insert(0, 0)
    return [index for index in starts if index < len(bars) - 1]


def _curve_average(points: list[dict[str, Any]], key: str, start: float, end: float, fallback: Any = 0.5) -> float:
    values = [float(item.get(key)) for item in points if _is_number(item.get(key)) and start <= float(item.get("time", -1)) < end]
    if values:
        return float(np.mean(values))
    try:
        return float(fallback)
    except (TypeError, ValueError):
        return 0.5


def _section_vocal_hint(track: dict[str, Any], start: float, end: float) -> float:
    for section in track.get("sections") or []:
        overlap = min(end, float(section.get("endTime", 0))) - max(start, float(section.get("startTime", 0)))
        if overlap <= 0:
            continue
        label = section.get("type")
        if label in {"chorus", "verse"}:
            return 0.55
        if label in {"intro", "outro", "breakdown"}:
            return 0.25
    return 0.45


def mix_boundary_score(track: dict[str, Any], time_value: float, metrics: dict[str, float]) -> float:
    candidates = track.get("transition_candidates") or {}
    intro = candidates.get("intro")
    outro = candidates.get("outro")
    anchor_bonus = 0.0
    for anchor in (intro, outro):
        if _is_number(anchor):
            anchor_bonus = max(anchor_bonus, 1.0 - min(1.0, abs(float(anchor) - time_value) / 16.0))
    low_vocal = 1.0 - metrics["vocalDensity"]
    moderate_energy = 1.0 - abs(metrics["energy"] - 0.45)
    return _clamp01(low_vocal * 0.48 + moderate_energy * 0.32 + anchor_bonus * 0.20)


def _segment_mono(audio: np.ndarray, start: float, end: float, sr: int) -> np.ndarray:
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim == 2 and y.shape[0] <= 8:
        y = np.mean(y, axis=0)
    elif y.ndim == 2:
        y = np.mean(y, axis=1)
    start_sample = max(0, int(start * sr))
    end_sample = max(start_sample, min(y.shape[-1], int(end * sr)))
    return np.ascontiguousarray(y[start_sample:end_sample], dtype=np.float32)


def _is_number(value: Any) -> bool:
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))

