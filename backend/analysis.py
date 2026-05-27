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
STYLE_LABELS = {
    "house": "House",
    "techno": "Techno",
    "drum_bass": "Drum & Bass",
    "hiphop": "Hip-Hop",
    "rnb": "R&B",
    "rock": "Rock",
    "pop": "Pop",
    "ambient": "Ambient",
    "electronic": "Electronic",
    "unknown": "Unknown",
}

def analyze_audio(path: Path) -> dict:
    y, sr = _load_audio(path)
    if y.size == 0:
        raise ValueError("audio is empty")

    duration = float(librosa.get_duration(y=y, sr=sr))
    beat_grid = _beat_grid(y, sr)
    bpm = beat_grid["bpm"] or _estimate_bpm(y, sr)
    key = _estimate_key(y, sr)
    energy = _energy_metrics(y, sr)
    style = _estimate_style(y, sr, bpm, energy, beat_grid["confidence"])
    loudness = loudness_metrics(y, sr)
    candidates = _transition_candidates(y, sr, duration, beat_grid["bars"], energy, bpm)

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
        "energy_profile": energy["energy_profile"],
        "style": style["primary"],
        "style_label": style["label"],
        "style_confidence": style["confidence"],
        "style_profile": style,
        "intro_low": energy["intro_low"],
        "outro_low": energy["outro_low"],
        "loudness_lufs": loudness["lufs"],
        "true_peak_db": loudness["peak_db"],
        "transition_candidates": candidates,
        "sections": candidates.get("sections", []),
        "vocal_density_curve": candidates.get("vocal_density_curve", []),
        "energy_curve": candidates.get("energy_curve", []),
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

    profile = _energy_profile(y, sr, rms, frame_length, hop_length)

    return {
        "energy": profile["energy_index"] / 100,
        "energy_profile": profile,
        "intro_low": float(intro / frame_rate),
        "outro_low": float(outro / frame_rate),
    }


def _energy_profile(y: np.ndarray, sr: int, rms: np.ndarray, frame_length: int, hop_length: int) -> dict:
    rms = np.asarray(rms, dtype=np.float32)
    rms_db = 20 * np.log10(np.maximum(rms, 1e-9))
    loudness = loudness_metrics(y, sr)
    peak = float(np.max(np.abs(y)) + 1e-12)
    rms_global = float(np.sqrt(np.mean(np.square(y)) + 1e-12))
    crest_db = float(20 * np.log10(peak / max(rms_global, 1e-12)))
    low_ratio = _low_frequency_ratio(y, sr, hop_length)

    p10 = float(np.percentile(rms_db, 10))
    p50 = float(np.percentile(rms_db, 50))
    p85 = float(np.percentile(rms_db, 85))
    p95 = float(np.percentile(rms_db, 95))
    dynamic_range_db = max(0.0, p95 - p10)

    intro_frames = max(1, min(len(rms), int(round(16 * sr / hop_length))))
    outro_frames = intro_frames
    full_rms = float(np.mean(rms) + 1e-12)
    intro_rms = float(np.mean(rms[:intro_frames]) + 1e-12)
    outro_rms = float(np.mean(rms[-outro_frames:]) + 1e-12)
    intro_relative = _ratio_to_unit(intro_rms / full_rms)
    outro_relative = _ratio_to_unit(outro_rms / full_rms)
    intro_delta_db = float(20 * np.log10(intro_rms / full_rms))
    outro_delta_db = float(20 * np.log10(outro_rms / full_rms))
    transition_contrast = min(1.0, (abs(intro_delta_db) + abs(outro_delta_db)) / 24)

    components = {
        "loudness": _clamp01((loudness["lufs"] + 30) / 18) * 100,
        "rms_body": _clamp01((p85 + 32) / 24) * 100,
        "crest_density": _clamp01((18 - crest_db) / 14) * 100,
        "low_frequency": _clamp01(low_ratio / 0.42) * 100,
        "dynamic_motion": _clamp01((dynamic_range_db - 3) / 14) * 100,
        "transition_contrast": transition_contrast * 100,
    }
    energy_index = (
        components["loudness"] * 0.30
        + components["rms_body"] * 0.25
        + components["crest_density"] * 0.15
        + components["low_frequency"] * 0.15
        + components["dynamic_motion"] * 0.10
        + components["transition_contrast"] * 0.05
    )

    return {
        "energy_index": round(float(energy_index), 1),
        "lufs": loudness["lufs"],
        "true_peak_db": loudness["peak_db"],
        "rms_p10_db": round(p10, 2),
        "rms_p50_db": round(p50, 2),
        "rms_p85_db": round(p85, 2),
        "rms_p95_db": round(p95, 2),
        "crest_factor_db": round(crest_db, 2),
        "low_frequency_ratio": round(float(low_ratio), 4),
        "dynamic_range_db": round(float(dynamic_range_db), 2),
        "intro_relative_energy": round(float(intro_relative), 4),
        "outro_relative_energy": round(float(outro_relative), 4),
        "intro_delta_db": round(intro_delta_db, 2),
        "outro_delta_db": round(outro_delta_db, 2),
        "transition_contrast": round(float(transition_contrast), 4),
        "components": {key: round(float(value), 1) for key, value in components.items()},
    }


def _estimate_style(y: np.ndarray, sr: int, bpm: float | None, energy: dict, beat_confidence: float = 0.0) -> dict:
    metrics = _style_metrics(y, sr, bpm, energy, beat_confidence)
    tempo = metrics["bpm"]
    half_tempo = tempo / 2 if tempo >= 130 else tempo
    low = metrics["low_frequency_ratio"]
    percussive = metrics["percussive_ratio"]
    vocal = metrics["vocal_density"]
    brightness = metrics["brightness"]
    zcr = metrics["zero_crossing_rate"]
    regularity = metrics["beat_regularity"]
    energy_value = metrics["energy"]
    dynamic = metrics["dynamic_range"]

    scores = {
        "house": (
            _range_affinity(tempo, 116, 132, 12) * 0.34
            + low * 0.19
            + percussive * 0.18
            + regularity * 0.16
            + (1 - vocal) * 0.06
            + energy_value * 0.07
        ),
        "techno": (
            _range_affinity(tempo, 124, 146, 14) * 0.33
            + percussive * 0.22
            + low * 0.18
            + regularity * 0.15
            + (1 - vocal) * 0.08
            + energy_value * 0.04
        ),
        "drum_bass": (
            _range_affinity(tempo, 160, 178, 10) * 0.42
            + percussive * 0.20
            + low * 0.18
            + brightness * 0.12
            + (1 - vocal) * 0.08
        ),
        "hiphop": (
            _range_affinity(half_tempo, 70, 104, 18) * 0.34
            + low * 0.22
            + vocal * 0.14
            + percussive * 0.13
            + (1 - brightness) * 0.10
            + (1 - regularity * 0.45) * 0.07
        ),
        "rnb": (
            _range_affinity(half_tempo, 64, 104, 16) * 0.30
            + vocal * 0.24
            + low * 0.15
            + (1 - zcr) * 0.13
            + (1 - percussive * 0.55) * 0.10
            + (1 - brightness * 0.5) * 0.08
        ),
        "rock": (
            _range_affinity(tempo, 86, 172, 24) * 0.24
            + brightness * 0.22
            + zcr * 0.18
            + dynamic * 0.14
            + vocal * 0.12
            + (1 - low) * 0.10
        ),
        "pop": (
            _range_affinity(tempo, 88, 132, 20) * 0.25
            + vocal * 0.22
            + (1 - abs(energy_value - 0.62) / 0.62) * 0.16
            + (1 - abs(brightness - 0.45) / 0.55) * 0.13
            + regularity * 0.12
            + (1 - abs(low - 0.22) / 0.5) * 0.12
        ),
        "ambient": (
            (1 - percussive) * 0.30
            + (1 - regularity) * 0.22
            + (1 - energy_value) * 0.18
            + dynamic * 0.12
            + (1 - vocal) * 0.10
            + (1 - zcr) * 0.08
        ),
        "electronic": (
            _range_affinity(tempo, 100, 150, 26) * 0.22
            + percussive * 0.20
            + regularity * 0.18
            + low * 0.16
            + brightness * 0.12
            + (1 - vocal) * 0.12
        ),
    }
    normalized_scores = {style: round(float(_clamp01(score)), 3) for style, score in scores.items()}
    ranked = sorted(normalized_scores.items(), key=lambda item: item[1], reverse=True)
    primary, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if top_score < 0.32:
        primary = "unknown"
    confidence = _clamp01(0.25 + top_score * 0.45 + max(0.0, top_score - second_score) * 1.3)
    if primary == "unknown":
        confidence = min(confidence, 0.25)
    return {
        "primary": primary,
        "label": STYLE_LABELS.get(primary, primary.title()),
        "confidence": round(float(confidence), 2),
        "scores": normalized_scores,
        "metrics": metrics,
        "method": "heuristic-audio-features",
    }


def _style_metrics(y: np.ndarray, sr: int, bpm: float | None, energy: dict, beat_confidence: float) -> dict:
    hop = 1024
    profile = energy.get("energy_profile") or {}
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    spectral_centroid = _safe_feature_mean(lambda: librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0], 1800.0)
    spectral_rolloff = _safe_feature_mean(lambda: librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop, roll_percent=0.85)[0], 4200.0)
    zcr = _safe_feature_mean(lambda: librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=hop)[0], 0.08)
    percussive_ratio = _percussive_ratio(y)
    vocal_density = float(np.mean(_vocal_density_curve(y, sr, hop, len(rms)))) if len(rms) else 0.5
    return {
        "bpm": round(float(bpm or 0), 2),
        "energy": round(float(energy.get("energy") or 0), 4),
        "low_frequency_ratio": round(float(profile.get("low_frequency_ratio") or 0), 4),
        "dynamic_range": round(_clamp01(float(profile.get("dynamic_range_db") or 0) / 18), 4),
        "brightness": round(_clamp01(spectral_centroid / 4200), 4),
        "spectral_rolloff": round(_clamp01(spectral_rolloff / 9000), 4),
        "zero_crossing_rate": round(_clamp01(zcr / 0.18), 4),
        "percussive_ratio": round(float(percussive_ratio), 4),
        "vocal_density": round(_clamp01(vocal_density), 4),
        "beat_regularity": round(_clamp01(float(beat_confidence or 0)), 4),
    }


def _safe_feature_mean(factory, fallback: float) -> float:
    try:
        values = np.asarray(factory(), dtype=np.float32)
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if finite.size else fallback
    except Exception:
        return fallback


def _percussive_ratio(y: np.ndarray) -> float:
    try:
        harmonic, percussive = librosa.effects.hpss(y)
        harmonic_energy = float(np.mean(np.square(harmonic)))
        percussive_energy = float(np.mean(np.square(percussive)))
        return _clamp01(percussive_energy / (harmonic_energy + percussive_energy + 1e-12))
    except Exception:
        return 0.5


def _range_affinity(value: float, low: float, high: float, grace: float) -> float:
    if value <= 0:
        return 0.35
    if low <= value <= high:
        return 1.0
    if value < low:
        return _clamp01(1 - (low - value) / max(grace, 1e-9))
    return _clamp01(1 - (value - high) / max(grace, 1e-9))


def _low_frequency_ratio(y: np.ndarray, sr: int, hop_length: int) -> float:
    try:
        spectrum = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        low = np.sum(spectrum[freqs <= 180])
        total = np.sum(spectrum) + 1e-12
        return float(low / total)
    except Exception:
        return 0.0


def _ratio_to_unit(value: float) -> float:
    return _clamp01(value / 2)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _transition_candidates(y: np.ndarray, sr: int, duration: float, bars: list[float], energy: dict, bpm: float | None = None) -> dict:
    if not bars:
        intro = min(max(energy["intro_low"], 4.0), duration * 0.35)
        outro = max(intro + 1.0, duration - min(max(energy["outro_low"], 8.0), duration * 0.35))
        return {
            "intro": round(intro, 3),
            "outro": round(outro, 3),
            "confidence": 0.35,
            "intro_vocal_density": None,
            "outro_vocal_density": None,
            "sections": _fallback_sections(duration, float(bpm or 120), intro, outro),
            "vocal_density_curve": [],
            "energy_curve": [],
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
        "sections": _structure_sections(duration, float(bpm or 120), bars, features, float(intro), float(outro)),
        "vocal_density_curve": _curve_points(features, "vocal_density"),
        "energy_curve": _curve_points(features, "energy"),
        "method": "bar-vocal-energy",
    }


def _curve_points(features: list[dict], key: str, limit: int = 48) -> list[dict]:
    if not features:
        return []
    step = max(1, int(np.ceil(len(features) / limit)))
    return [{"time": round(float(item["time"]), 3), "density" if key == "vocal_density" else "energy": round(float(item[key]), 3)} for item in features[::step]]


def _structure_sections(duration: float, bpm: float, bars: list[float], features: list[dict], intro: float, outro: float) -> list[dict]:
    if duration <= 0:
        return []
    intro_end = _clamp_time(intro, 2.0, duration * 0.35)
    outro_start = _clamp_time(outro, max(intro_end + 4.0, duration * 0.55), max(intro_end + 4.0, duration - 1.0))
    beat_seconds = 60 / max(bpm, 1)
    phrase = 32 * beat_seconds
    middle_features = [item for item in features if intro_end <= item["time"] <= outro_start]
    breakdown_feature = _pick_breakdown_feature(middle_features, duration)
    breakdown_start = breakdown_feature["time"] if breakdown_feature else _nearest_bar_time(bars, duration * 0.66)
    breakdown_start = _clamp_time(breakdown_start, intro_end + phrase, max(intro_end + phrase, outro_start - phrase))
    breakdown_end = min(outro_start, breakdown_start + phrase)
    chorus_start = _nearest_bar_time(bars, max(intro_end + phrase, duration * 0.38))
    chorus_end = min(breakdown_start, chorus_start + phrase * 2)
    drop_start = min(outro_start, breakdown_end)

    specs = [
        ("intro", 0.0, intro_end, 0.82),
        ("verse", intro_end, min(chorus_start, breakdown_start), 0.58),
        ("chorus", chorus_start, chorus_end, 0.72),
        ("bridge", chorus_end, breakdown_start, 0.5),
        ("breakdown", breakdown_start, breakdown_end, 0.64),
        ("drop", drop_start, outro_start, 0.62),
        ("outro", outro_start, duration, 0.78),
    ]
    return _clean_sections(specs, beat_seconds)


def _fallback_sections(duration: float, bpm: float, intro: float, outro: float) -> list[dict]:
    intro_end = _clamp_time(intro, 2.0, duration * 0.35)
    outro_start = _clamp_time(outro, duration * 0.55, max(duration - 1.0, 1.0))
    specs = [
        ("intro", 0.0, intro_end, 0.45),
        ("verse", intro_end, duration * 0.38, 0.35),
        ("chorus", duration * 0.38, duration * 0.58, 0.35),
        ("bridge", duration * 0.58, duration * 0.68, 0.3),
        ("breakdown", duration * 0.68, min(duration * 0.78, outro_start), 0.3),
        ("drop", min(duration * 0.78, outro_start), outro_start, 0.3),
        ("outro", outro_start, duration, 0.45),
    ]
    return _clean_sections(specs, 60 / max(bpm, 1))


def _pick_breakdown_feature(features: list[dict], duration: float) -> dict | None:
    candidates = [item for item in features if duration * 0.35 <= item["time"] <= duration * 0.82]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item["energy"] * 0.65 + item["vocal_density"] * 0.25 + abs(item["position"] - 0.68) * 0.1)


def _nearest_bar_time(bars: list[float], target: float) -> float:
    if not bars:
        return float(target)
    return float(min(bars, key=lambda value: abs(value - target)))


def _clean_sections(specs: list[tuple[str, float, float, float]], beat_seconds: float) -> list[dict]:
    sections = []
    for section_type, start, end, confidence in specs:
        start = max(0.0, float(start))
        end = max(start, float(end))
        if end - start < 1.0:
            continue
        sections.append(
            {
                "type": section_type,
                "startTime": round(start, 3),
                "endTime": round(end, 3),
                "startBeat": int(round(start / beat_seconds)),
                "endBeat": int(round(end / beat_seconds)),
                "confidence": round(float(confidence), 2),
            }
        )
    return sections


def _clamp_time(value: float, low: float, high: float) -> float:
    if high < low:
        return float(low)
    return float(max(low, min(high, value)))


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
