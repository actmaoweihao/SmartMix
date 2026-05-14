from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np

from .loudness import loudness_metrics
from .matching import key_label_to_camelot


KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def analyze_audio(path: Path) -> dict:
    y, sr = _load_audio(path)
    if y.size == 0:
        raise ValueError("audio is empty")

    duration = float(librosa.get_duration(y=y, sr=sr))
    beat_grid = _beat_grid(y, sr)
    bpm = beat_grid["bpm"] or _estimate_bpm(y, sr)
    key = _estimate_key(y, sr)
    energy = _energy_metrics(y, sr)
    loudness = loudness_metrics(y, sr)
    candidates = _transition_candidates(y, sr, duration, beat_grid["bars"], energy)

    return {
        "duration": duration,
        "bpm": bpm,
        "beats": beat_grid["beats"],
        "bars": beat_grid["bars"],
        "phrases": beat_grid["phrases"],
        "downbeat_offset": beat_grid["downbeat_offset"],
        "beat_confidence": beat_grid["confidence"],
        "key": key["label"],
        "camelot": key_label_to_camelot(key["label"], key["mode"]),
        "key_index": key["index"],
        "mode": key["mode"],
        "energy": energy["energy"],
        "intro_low": energy["intro_low"],
        "outro_low": energy["outro_low"],
        "loudness_lufs": loudness["lufs"],
        "true_peak_db": loudness["peak_db"],
        "transition_candidates": candidates,
        "peaks": _waveform_peaks(y, 720),
    }


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        return librosa.load(path, sr=44100, mono=True)
    except Exception as first_error:
        temp_path = None
        try:
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_path = Path(temp_file.name)
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "44100",
                    "-f",
                    "wav",
                    str(temp_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            return librosa.load(temp_path, sr=44100, mono=True)
        except Exception as fallback_error:
            raise ValueError(f"could not decode audio ({first_error}; ffmpeg fallback: {fallback_error})") from fallback_error
        finally:
            if temp_path:
                temp_path.unlink(missing_ok=True)


def _estimate_bpm(y: np.ndarray, sr: int) -> int | None:
    try:
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.feature.rhythm.tempo(onset_envelope=onset, sr=sr)
        value = float(np.asarray(tempo).reshape(-1)[0])
        if math.isfinite(value) and value > 0:
            return int(round(value))
    except Exception:
        pass
    return _estimate_bpm_from_envelope(y, sr)


def _beat_grid(y: np.ndarray, sr: int) -> dict:
    try:
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr, units="frames")
        bpm = float(np.asarray(tempo).reshape(-1)[0])
        beat_times = librosa.frames_to_time(beats, sr=sr).astype(float)
        if beat_times.size < 4:
            return {"bpm": int(round(bpm)) if bpm > 0 else None, "beats": [], "bars": [], "phrases": [], "downbeat_offset": 0, "confidence": 0.0}
        offset, confidence = _estimate_downbeat_offset(y, sr, beats)
        bars = beat_times[offset::4]
        phrases = bars[::4]
        return {
            "bpm": int(round(bpm)) if math.isfinite(bpm) and bpm > 0 else None,
            "beats": _round_times(beat_times),
            "bars": _round_times(bars),
            "phrases": _round_times(phrases),
            "downbeat_offset": int(offset),
            "confidence": round(float(confidence), 2),
        }
    except Exception:
        return {"bpm": None, "beats": [], "bars": [], "phrases": [], "downbeat_offset": 0, "confidence": 0.0}


def _estimate_downbeat_offset(y: np.ndarray, sr: int, beats: np.ndarray) -> tuple[int, float]:
    if len(beats) < 8:
        return 0, 0.0
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    beat_frames = librosa.time_to_frames(librosa.frames_to_time(beats, sr=sr), sr=sr, hop_length=512)
    beat_frames = np.clip(beat_frames, 0, len(rms) - 1)
    beat_strength = rms[beat_frames]
    if np.max(beat_strength) > 0:
        beat_strength = beat_strength / np.max(beat_strength)

    scores = []
    for offset in range(4):
        downbeats = beat_strength[offset::4]
        others = np.concatenate([beat_strength[i::4] for i in range(4) if i != offset])
        if downbeats.size == 0 or others.size == 0:
            scores.append(0.0)
            continue
        scores.append(float(np.mean(downbeats) - np.mean(others) * 0.35))

    best = int(np.argmax(scores))
    spread = float(max(scores) - np.mean(scores))
    confidence = max(0.0, min(0.95, 0.35 + spread))
    return best, confidence


def _estimate_bpm_from_envelope(y: np.ndarray, sr: int) -> int | None:
    hop = 1024
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    if rms.size < 16:
        return None
    flux = np.maximum(0, np.diff(rms, prepend=rms[0]))
    flux = np.maximum(0, flux - np.mean(flux))
    frame_rate = sr / hop
    best_bpm = None
    best_score = -np.inf
    for bpm in range(70, 181):
        lag = int(round((60 / bpm) * frame_rate))
        if lag <= 0 or lag >= flux.size:
            continue
        score = float(np.dot(flux[lag:], flux[:-lag]))
        if score > best_score:
            best_score = score
            best_bpm = bpm
    if best_bpm and best_score > 0:
        return int(best_bpm)
    return None


def _estimate_key(y: np.ndarray, sr: int) -> dict:
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    except Exception:
        return {"label": "Unknown", "index": None, "mode": None}
    profile = np.mean(chroma, axis=1)
    total = np.sum(profile)
    if total <= 0:
        return {"label": "Unknown", "index": None, "mode": None}
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


def _transition_candidates(y: np.ndarray, sr: int, duration: float, bars: list[float], energy: dict) -> dict:
    if not bars:
        intro = min(max(energy["intro_low"], 4.0), duration * 0.35)
        outro = max(intro + 1.0, duration - min(max(energy["outro_low"], 8.0), duration * 0.35))
        return {
            "intro": round(intro, 3),
            "outro": round(outro, 3),
            "confidence": 0.35,
            "intro_vocal_density": None,
            "outro_vocal_density": None,
            "method": "energy-fallback",
        }

    features = _bar_features(y, sr, bars, duration)
    intro_floor = max(energy["intro_low"], 2.0)
    intro = _pick_intro_bar(features, intro_floor, duration) or _first_at_or_after(bars, intro_floor) or bars[min(len(bars) - 1, 1)]
    outro_floor = max(duration * 0.55, duration - max(energy["outro_low"], 16.0) - 24.0)
    outro = _pick_outro_bar(features, outro_floor, duration) or _last_at_or_before(bars, duration - 1.0) or bars[-1]
    if outro <= intro:
        outro = _last_at_or_before(bars, duration - 1.0) or max(intro + 1.0, duration - 1.0)
    intro_feature = _nearest_bar_feature(features, intro)
    outro_feature = _nearest_bar_feature(features, outro)
    density_bonus = 0.0
    if intro_feature and outro_feature:
        density_bonus = (1 - intro_feature["vocal_density"]) * 0.08 + (1 - outro_feature["vocal_density"]) * 0.08
    confidence = min(0.92, 0.45 + min(len(bars), 32) / 90 + density_bonus)
    return {
        "intro": round(float(intro), 3),
        "outro": round(float(outro), 3),
        "confidence": round(confidence, 2),
        "intro_vocal_density": round(float(intro_feature["vocal_density"]), 3) if intro_feature else None,
        "outro_vocal_density": round(float(outro_feature["vocal_density"]), 3) if outro_feature else None,
        "intro_energy": round(float(intro_feature["energy"]), 3) if intro_feature else None,
        "outro_energy": round(float(outro_feature["energy"]), 3) if outro_feature else None,
        "method": "bar-vocal-energy",
    }


def _bar_features(y: np.ndarray, sr: int, bars: list[float], duration: float) -> list[dict]:
    hop = 1024
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    vocal_density = _vocal_density_curve(y, sr, hop, len(rms))
    max_rms = float(np.max(rms)) or 1.0
    features = []
    median_span = float(np.median(np.diff(bars))) if len(bars) > 1 else 2.0
    for index, start in enumerate(bars):
        end = bars[index + 1] if index + 1 < len(bars) else min(duration, start + median_span)
        mask = (frame_times >= start) & (frame_times < end)
        if not np.any(mask):
            continue
        energy_value = float(np.mean(rms[mask]) / max_rms)
        vocal_value = float(np.mean(vocal_density[mask]))
        position = float(start / max(duration, 1e-6))
        features.append(
            {
                "time": float(start),
                "energy": max(0.0, min(1.0, energy_value)),
                "vocal_density": max(0.0, min(1.0, vocal_value)),
                "position": position,
            }
        )
    return features


def _vocal_density_curve(y: np.ndarray, sr: int, hop: int, target_length: int) -> np.ndarray:
    try:
        harmonic, _ = librosa.effects.hpss(y)
        spectrum = np.abs(librosa.stft(harmonic, n_fft=2048, hop_length=hop))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        vocal_mask = (freqs >= 300) & (freqs <= 3400)
        total = np.sum(spectrum, axis=0) + 1e-9
        vocal = np.sum(spectrum[vocal_mask], axis=0) / total
        if vocal.size < target_length:
            vocal = np.pad(vocal, (0, target_length - vocal.size), mode="edge")
        return np.asarray(vocal[:target_length], dtype=np.float32)
    except Exception:
        return np.full(target_length, 0.5, dtype=np.float32)


def _pick_intro_bar(features: list[dict], intro_floor: float, duration: float) -> float | None:
    candidates = [item for item in features if intro_floor <= item["time"] <= duration * 0.45]
    if not candidates:
        return None
    scored = sorted(candidates, key=lambda item: item["vocal_density"] * 0.65 + item["energy"] * 0.25 + item["position"] * 0.1)
    return float(scored[0]["time"])


def _pick_outro_bar(features: list[dict], outro_floor: float, duration: float) -> float | None:
    candidates = [item for item in features if outro_floor <= item["time"] <= duration - 1.0]
    if not candidates:
        return None
    scored = sorted(
        candidates,
        key=lambda item: item["vocal_density"] * 0.6 + item["energy"] * 0.25 - item["position"] * 0.15,
    )
    return float(scored[0]["time"])


def _nearest_bar_feature(features: list[dict], time_value: float) -> dict | None:
    if not features:
        return None
    return min(features, key=lambda item: abs(item["time"] - time_value))


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


def _round_times(values: np.ndarray) -> list[float]:
    return [round(float(value), 4) for value in values.tolist()]


def _first_at_or_after(values: list[float], target: float) -> float | None:
    for value in values:
        if value >= target:
            return value
    return None


def _last_at_or_before(values: list[float], target: float) -> float | None:
    for value in reversed(values):
        if value <= target:
            return value
    return None
