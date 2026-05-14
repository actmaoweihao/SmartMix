from __future__ import annotations

import math
from pathlib import Path

import librosa
import numpy as np


KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def analyze_audio(path: Path) -> dict:
    y, sr = librosa.load(path, sr=44100, mono=True)
    if y.size == 0:
        raise ValueError("音频为空，无法分析")

    duration = float(librosa.get_duration(y=y, sr=sr))
    bpm = _estimate_bpm(y, sr)
    key = _estimate_key(y, sr)
    energy = _energy_metrics(y, sr)

    return {
        "duration": duration,
        "bpm": bpm,
        "key": key["label"],
        "key_index": key["index"],
        "mode": key["mode"],
        "energy": energy["energy"],
        "intro_low": energy["intro_low"],
        "outro_low": energy["outro_low"],
        "peaks": _waveform_peaks(y, 720),
    }


def _estimate_bpm(y: np.ndarray, sr: int) -> int | None:
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        value = float(np.asarray(tempo).reshape(-1)[0])
        if math.isfinite(value) and value > 0:
            return int(round(value))
    except Exception:
        return None
    return None


def _estimate_key(y: np.ndarray, sr: int) -> dict:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    profile = np.mean(chroma, axis=1)
    total = np.sum(profile)
    if total <= 0:
        return {"label": "未知", "index": None, "mode": None}
    profile = profile / total

    best_score = -np.inf
    best_index = 0
    best_mode = "major"
    for root in range(12):
        major_score = float(np.dot(np.roll(MAJOR_PROFILE, root), profile))
        minor_score = float(np.dot(np.roll(MINOR_PROFILE, root), profile))
        if major_score > best_score:
            best_score = major_score
            best_index = root
            best_mode = "major"
        if minor_score > best_score:
            best_score = minor_score
            best_index = root
            best_mode = "minor"

    return {
        "label": f"{KEY_NAMES[best_index]} {'Maj' if best_mode == 'major' else 'Min'}",
        "index": best_index,
        "mode": best_mode,
    }


def _energy_metrics(y: np.ndarray, sr: int) -> dict:
    frame_length = 2048
    hop_length = 1024
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    avg = float(np.mean(rms))
    peak = float(np.max(rms))
    threshold = max(avg * 0.55, peak * 0.08)
    frame_rate = sr / hop_length
    limit = max(1, int(len(rms) * 0.25))

    intro = 0
    for value in rms[:limit]:
        if value > threshold:
            break
        intro += 1

    outro = 0
    for value in rms[::-1][:limit]:
        if value > threshold:
            break
        outro += 1

    return {
        "energy": min(1.0, avg * 7 + peak * 1.5),
        "intro_low": float(intro / frame_rate),
        "outro_low": float(outro / frame_rate),
    }


def _waveform_peaks(y: np.ndarray, bins: int) -> list[float]:
    if y.size == 0:
        return []
    chunk = max(1, y.size // bins)
    trimmed = y[: chunk * bins]
    if trimmed.size == 0:
        return []
    peaks = np.max(np.abs(trimmed.reshape(bins, chunk)), axis=1)
    max_peak = float(np.max(peaks)) or 1.0
    return [round(float(value / max_peak), 4) for value in peaks]

