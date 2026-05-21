from __future__ import annotations

import math
from typing import Any

import librosa
import numpy as np
from scipy import signal

from .matching import camelot_key_distance


SAMPLE_RATE = 44100
BAND_NAMES = ("sub", "bass", "low_mid", "mid", "high_mid", "high")
SEGMENT_LABELS = {
    "intro_like",
    "verse_like",
    "pre_chorus_like",
    "chorus_like",
    "breakdown_like",
    "drop_like",
    "outro_like",
    "bridge_like",
}


def build_music_segments(
    track: dict[str, Any],
    source: str | None = None,
    bars_per_segment: int = 16,
    *,
    audio: np.ndarray | None = None,
    sr: int = SAMPLE_RATE,
) -> list[dict[str, Any]]:
    duration = _float(track.get("duration"), 0.0)
    if duration <= 0:
        return []
    source = source or str(track.get("source") or "A")
    try:
        from .segmentation import analyze_track_segmentation

        report = analyze_track_segmentation(track, source, audio=audio, sr=sr)
        sections = report.get("sections") or []
        if sections:
            return _segments_from_segmentation_report(track, source, sections, report)
    except Exception:
        pass
    bpm = _float(track.get("bpm"), 120.0)
    bars = normalized_bars(track, duration, bpm)
    phrases = normalized_phrases(track, bars, duration)
    if len(bars) < 2:
        return []

    bars_per_segment = int(np.clip(int(bars_per_segment or 16), 8, 32))
    boundaries = _candidate_boundaries(track, bars, phrases, bars_per_segment, duration)
    boundaries = [_move_boundary_off_vocals(track, bars, value) for value in boundaries]
    boundaries = _snap_boundaries(boundaries, bars, phrases, duration)
    ranges = _segment_ranges(boundaries, bars, duration, bars_per_segment)

    segments: list[dict[str, Any]] = []
    for start, end in ranges:
        if end <= start:
            continue
        metrics = segment_metrics(track, start, end, audio=audio, sr=sr)
        bar_start = nearest_index(bars, start)
        bar_end = nearest_index(bars, end)
        phrase_start = nearest_index(phrases, start) if phrases else 0
        phrase_end = nearest_index(phrases, end) if phrases else phrase_start
        label = label_segment(metrics, start, end, duration)
        mix_in = mix_boundary_score(track, start, metrics, "in")
        mix_out = mix_boundary_score(track, end, metrics, "out")
        risk_flags = segment_risk_flags(metrics, mix_in, mix_out)
        segment = {
            "id": f"{source}_seg_{len(segments) + 1:03d}",
            "trackId": track.get("id"),
            "source": source,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "barStart": int(bar_start),
            "barEnd": int(bar_end),
            "phraseStart": int(phrase_start),
            "phraseEnd": int(phrase_end),
            "downbeatTime": round(float(bars[bar_start]), 3),
            "label": label,
            "energy": round(metrics["energy"], 4),
            "energyStart": round(metrics["energyStart"], 4),
            "energyEnd": round(metrics["energyEnd"], 4),
            "energyDelta": round(metrics["energyDelta"], 4),
            "vocalDensity": round(metrics["vocalDensity"], 4),
            "vocalStart": round(metrics["vocalStart"], 4),
            "vocalEnd": round(metrics["vocalEnd"], 4),
            "bassEnergy": round(metrics["bassEnergy"], 4),
            "drumActivity": round(metrics["drumActivity"], 4),
            "brightness": round(metrics["brightness"], 4),
            "spectralChange": round(metrics["spectralChange"], 4),
            "bpm": round(bpm, 3),
            "camelot": track.get("camelot"),
            "key": track.get("key"),
            "mixInScore": round(mix_in, 4),
            "mixOutScore": round(mix_out, 4),
            "isCleanEntry": bool(mix_in >= 0.58 and metrics["vocalStart"] < 0.58),
            "isCleanExit": bool(mix_out >= 0.58 and metrics["vocalEnd"] < 0.58),
            "riskFlags": risk_flags,
        }
        segments.append(segment)
    return segments


def _segments_from_segmentation_report(track: dict[str, Any], source: str, sections: list[dict[str, Any]], report: dict[str, Any]) -> list[dict[str, Any]]:
    bars = [float(item["start"]) for item in report.get("barFeatures", []) if _is_number(item.get("start"))]
    if report.get("barFeatures"):
        bars.append(float(report["barFeatures"][-1]["end"]))
    phrases = [float(item.get("time")) for item in report.get("safeCutPoints", []) if item.get("type") in {"vocal_entry", "vocal_exit", "drop_entry", "breakdown_entry"} and _is_number(item.get("time"))]
    segments = []
    for section in sections:
        if int(section.get("bars", 0)) < 4:
            continue
        start = float(section.get("start", 0.0))
        end = float(section.get("end", start))
        if end <= start:
            continue
        bar_start = int(section.get("barStart", nearest_index(bars, start)))
        bar_end = int(section.get("barEnd", nearest_index(bars, end)))
        bar_features = report.get("barFeatures", [])[bar_start:bar_end]
        vocal_start = float(bar_features[0].get("vocalDensity", section.get("vocalDensity", 0.0))) if bar_features else float(section.get("vocalDensity", 0.0))
        vocal_end = float(bar_features[-1].get("vocalDensity", section.get("vocalDensity", 0.0))) if bar_features else float(section.get("vocalDensity", 0.0))
        energy_start = float(bar_features[0].get("energy", section.get("meanEnergy", 0.0))) if bar_features else float(section.get("meanEnergy", 0.0))
        energy_end = float(bar_features[-1].get("energy", section.get("meanEnergy", 0.0))) if bar_features else float(section.get("meanEnergy", 0.0))
        segment = {
            "id": f"{source}_seg_{len(segments) + 1:03d}",
            "sectionId": section.get("id"),
            "trackId": track.get("id"),
            "source": source,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "barStart": bar_start,
            "barEnd": bar_end,
            "phraseStart": nearest_index(phrases, start) if phrases else 0,
            "phraseEnd": nearest_index(phrases, end) if phrases else 0,
            "downbeatTime": round(start, 3),
            "label": section.get("label", "unknown"),
            "energy": round(float(section.get("meanEnergy", 0.0)), 4),
            "energyStart": round(energy_start, 4),
            "energyEnd": round(energy_end, 4),
            "energyDelta": round(float(np.clip(energy_end - energy_start, -1, 1)), 4),
            "vocalDensity": round(float(section.get("vocalDensity", 0.0)), 4),
            "vocalStart": round(vocal_start, 4),
            "vocalEnd": round(vocal_end, 4),
            "bassEnergy": round(float(section.get("bassEnergy", 0.0)), 4),
            "drumActivity": round(float(section.get("drumActivity", 0.0)), 4),
            "brightness": round(float(section.get("brightness", 0.0)), 4),
            "spectralChange": round(float(max((item.get("spectralFlux", 0.0) for item in bar_features), default=0.0)), 4),
            "bpm": round(_float(track.get("bpm"), 120.0), 3),
            "camelot": track.get("camelot"),
            "key": track.get("key"),
            "mixInScore": round(float(section.get("mixInScore", 0.0)), 4),
            "mixOutScore": round(float(section.get("mixOutScore", 0.0)), 4),
            "isCleanEntry": bool(section.get("entryClean", True)),
            "isCleanExit": bool(section.get("exitClean", True)),
            "riskFlags": list(section.get("riskFlags") or []),
            "segmentationConfidence": section.get("confidence"),
        }
        segments.append(segment)
    return segments


def build_track_segments(
    track: dict[str, Any],
    source: str,
    bars_per_segment: int = 16,
    *,
    audio: np.ndarray | None = None,
    sr: int = SAMPLE_RATE,
) -> list[dict[str, Any]]:
    return build_music_segments(track, source, bars_per_segment, audio=audio, sr=sr)


def normalized_bars(track: dict[str, Any], duration: float, bpm: float) -> list[float]:
    raw = sorted(float(value) for value in (track.get("bars") or []) if _is_number(value) and 0 <= float(value) <= duration)
    if not raw:
        bar_seconds = 240.0 / max(float(bpm or 120.0), 1.0)
        raw = list(np.arange(0.0, duration, bar_seconds))
    if not raw or raw[0] > 0.05:
        raw.insert(0, 0.0)
    if raw[-1] < duration - 0.05:
        raw.append(duration)
    cleaned = [max(0.0, raw[0])]
    for value in raw[1:]:
        value = min(float(value), duration)
        if value - cleaned[-1] >= 0.25:
            cleaned.append(value)
    if cleaned[-1] < duration:
        cleaned.append(duration)
    return cleaned


def normalized_phrases(track: dict[str, Any], bars: list[float], duration: float) -> list[float]:
    raw = sorted(float(value) for value in (track.get("phrases") or []) if _is_number(value) and 0 <= float(value) <= duration)
    if not raw and bars:
        raw = [bars[index] for index in range(0, max(1, len(bars) - 1), 8)]
    if raw and raw[0] > 0.05:
        raw.insert(0, 0.0)
    if raw and raw[-1] < duration - 0.05:
        raw.append(duration)
    return _dedupe_times(raw)


def segment_metrics(
    track: dict[str, Any],
    start: float,
    end: float,
    *,
    audio: np.ndarray | None = None,
    sr: int = SAMPLE_RATE,
) -> dict[str, float]:
    duration = max(0.1, end - start)
    head_end = min(end, start + min(2.0, duration * 0.28))
    tail_start = max(start, end - min(2.0, duration * 0.28))
    energy_curve = _energy_curve(track)
    vocal_curve = _vocal_density_curve(track)
    energy = curve_average(energy_curve, "energy", start, end, fallback=track.get("energy", 0.45))
    energy_start = curve_average(energy_curve, "energy", start, head_end, fallback=energy)
    energy_end = curve_average(energy_curve, "energy", tail_start, end, fallback=energy)
    vocal = curve_average(vocal_curve, "density", start, end, fallback=_section_vocal_hint(track, start, end))
    vocal_start = curve_average(vocal_curve, "density", start, head_end, fallback=vocal)
    vocal_end = curve_average(vocal_curve, "density", tail_start, end, fallback=vocal)
    bass = 0.32 + energy * 0.36
    drums = 0.34 + energy * 0.38
    brightness = 0.45
    spectral_change = abs(energy_end - energy_start) * 0.65
    if audio is not None and audio.size:
        mono = segment_mono(audio, start, end, sr)
        if mono.size >= 128:
            bands = band_energy(mono, sr)
            bass = _clamp01(bands["sub"] * 1.25 + bands["bass"] * 1.35)
            brightness = _clamp01(bands["high_mid"] * 1.15 + bands["high"] * 1.75)
            drums = drum_activity(mono, sr)
            rms = float(np.sqrt(np.mean(mono * mono) + 1e-9))
            energy = _clamp01(max(energy, rms * 8.0))
            spectral_change = max(spectral_change, spectral_flux_change(mono, sr))
    return {
        "energy": _clamp01(energy),
        "energyStart": _clamp01(energy_start),
        "energyEnd": _clamp01(energy_end),
        "energyDelta": float(np.clip(energy_end - energy_start, -1.0, 1.0)),
        "vocalDensity": _clamp01(vocal),
        "vocalStart": _clamp01(vocal_start),
        "vocalEnd": _clamp01(vocal_end),
        "bassEnergy": _clamp01(bass),
        "drumActivity": _clamp01(drums),
        "brightness": _clamp01(brightness),
        "spectralChange": _clamp01(spectral_change),
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
    return {name: float(np.sum(power[(freqs >= low) & (freqs < min(high, sr / 2))]) / total) for name, (low, high) in ranges.items()}


def drum_activity(mono: np.ndarray, sr: int) -> float:
    if mono.size < 128:
        return 0.0
    onset = librosa.onset.onset_strength(y=mono, sr=sr)
    if onset.size == 0:
        return 0.0
    normalized = onset / (float(np.max(onset)) + 1e-9)
    return _clamp01(float(np.mean(normalized) * 1.8))


def spectral_flux_change(mono: np.ndarray, sr: int) -> float:
    try:
        hop = 1024
        spectrum = np.abs(librosa.stft(mono, n_fft=2048, hop_length=hop))
        if spectrum.shape[1] < 3:
            return 0.0
        flux = np.mean(np.maximum(0.0, np.diff(spectrum, axis=1)), axis=0)
        return _clamp01(float(np.percentile(flux, 85) / (np.mean(spectrum) + 1e-9)) * 0.12)
    except Exception:
        return 0.0


def label_segment(metrics: dict[str, float], start: float, end: float, duration: float) -> str:
    position = start / max(duration, 1e-6)
    energy = metrics["energy"]
    vocal = metrics["vocalDensity"]
    drums = metrics["drumActivity"]
    bass = metrics["bassEnergy"]
    brightness = metrics["brightness"]
    delta = metrics["energyDelta"]
    spectral = metrics["spectralChange"]
    if position < 0.15 and energy < 0.58 and vocal < 0.45:
        return "intro_like"
    if position > 0.82 and (delta <= 0.08 or energy < 0.58):
        return "outro_like"
    if delta > 0.18 and vocal > 0.42 and energy < 0.72:
        return "pre_chorus_like"
    if drums < 0.30 and bass < 0.35 and brightness > 0.28:
        return "breakdown_like"
    if energy > 0.66 and drums > 0.45 and bass > 0.42:
        return "drop_like"
    if energy > 0.52 and vocal > 0.46 and brightness > 0.32:
        return "chorus_like"
    if spectral > 0.48:
        return "bridge_like"
    if vocal > 0.40:
        return "verse_like"
    return "bridge_like" if abs(delta) > 0.16 else "verse_like"


def score_segment_transition(
    seg_a: dict[str, Any],
    seg_b: dict[str, Any],
    mode: str = "auto",
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = context or {}
    use_stems = bool(context.get("useStems", context.get("use_stems", True)))
    bpm_score, stretch_ratio, bpm_warning = _score_bpm(seg_a.get("bpm"), seg_b.get("bpm"))
    camelot_score, camelot_warning, camelot_fix = _score_camelot(seg_a.get("camelot"), seg_b.get("camelot"))
    phrase_score = _score_phrase(seg_a, seg_b)
    energy_score, energy_warning = _score_energy(seg_a, seg_b, mode)
    vocal_score, vocal_warning, vocal_fix = _score_vocal_conflict(seg_a, seg_b, use_stems)
    bass_score, bass_warning, bass_fix = _score_bass_conflict(seg_a, seg_b, use_stems)
    timbre_score, timbre_warning, timbre_fix = _score_timbre(seg_a, seg_b)
    entry_exit_score = _score_entry_exit(seg_a, seg_b)
    components = {
        "bpm": bpm_score,
        "camelot": camelot_score,
        "phrase": phrase_score,
        "energyFlow": energy_score,
        "vocalConflict": vocal_score,
        "bassConflict": bass_score,
        "timbre": timbre_score,
        "entryExit": entry_exit_score,
    }
    score = (
        0.20 * bpm_score
        + 0.20 * camelot_score
        + 0.15 * phrase_score
        + 0.15 * energy_score
        + 0.15 * vocal_score
        + 0.05 * bass_score
        + 0.05 * timbre_score
        + 0.05 * entry_exit_score
    )
    warnings = [item for item in [bpm_warning, camelot_warning, energy_warning, vocal_warning, bass_warning, timbre_warning] if item]
    fixes = [item for item in [camelot_fix, vocal_fix, bass_fix, timbre_fix] if item]
    compatibility = {
        "score": round(float(np.clip(score, 0, 100)), 1),
        "components": {key: round(float(value), 1) for key, value in components.items()},
        "recommendedTransition": "crossfade",
        "warnings": warnings,
        "fixes": fixes,
        "stretchRatio": round(stretch_ratio, 4),
    }
    transition = choose_transition_type(seg_a, seg_b, compatibility, mode, use_stems)
    compatibility["recommendedTransition"] = transition["type"]
    compatibility["transitionSpec"] = transition
    compatibility["warnings"] = _dedupe([*warnings, *transition.get("warnings", [])])
    compatibility["fixes"] = _dedupe([*fixes, *transition.get("fixes", [])])
    return compatibility


def segment_compatibility(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    detailed = score_segment_transition(left, right)
    components = detailed["components"]
    score = detailed["score"] / 100.0
    return {
        "score": round(score, 4),
        "score100": detailed["score"],
        "components": {
            "bpmCompatibility": round(components["bpm"] / 100.0, 4),
            "camelotCompatibility": round(components["camelot"] / 100.0, 4),
            "energyFlow": round(components["energyFlow"] / 100.0, 4),
            "vocalConflictAvoidance": round(components["vocalConflict"] / 100.0, 4),
            "timbreSimilarity": round(components["timbre"] / 100.0, 4),
        },
        "bpmCompatibility": round(components["bpm"] / 100.0, 4),
        "camelotCompatibility": round(components["camelot"] / 100.0, 4),
        "energyFlow": round(components["energyFlow"] / 100.0, 4),
        "vocalConflictAvoidance": round(components["vocalConflict"] / 100.0, 4),
        "timbreSimilarity": round(components["timbre"] / 100.0, 4),
        "recommendedTransition": detailed["recommendedTransition"],
        "warnings": detailed["warnings"],
        "fixes": detailed["fixes"],
    }


def choose_transition_type(
    seg_a: dict[str, Any],
    seg_b: dict[str, Any],
    compatibility: dict[str, Any],
    mode: str = "auto",
    use_stems: bool = True,
) -> dict[str, Any]:
    components = compatibility.get("components", {})
    warnings: list[str] = []
    fixes: list[str] = []
    duration = _transition_duration(seg_a, seg_b)
    vocal_overlap = _float(seg_a.get("vocalEnd"), seg_a.get("vocalDensity", 0.0)) > 0.58 and _float(seg_b.get("vocalStart"), seg_b.get("vocalDensity", 0.0)) > 0.52
    bass_overlap = _float(seg_a.get("bassEnergy"), 0.0) > 0.58 and _float(seg_b.get("bassEnergy"), 0.0) > 0.58
    timbre_gap = abs(_float(seg_a.get("brightness"), 0.0) - _float(seg_b.get("brightness"), 0.0)) + abs(_float(seg_a.get("bassEnergy"), 0.0) - _float(seg_b.get("bassEnergy"), 0.0))

    if mode in {"a_vocal_b_instrumental", "b_vocal_a_instrumental"}:
        if use_stems:
            fixes.append("Render as vocal_over_instrumental with instrumental vocals muted.")
        else:
            warnings.append("Stems disabled; vocal_over_instrumental will fall back to full-mix blending.")
        return _transition_spec("vocal_over_instrumental", duration, warnings=warnings, fixes=fixes, vocal_duck=True)

    if compatibility["score"] < 58 and (seg_a.get("label") == "breakdown_like" or seg_b.get("label") == "breakdown_like"):
        return _transition_spec("breakdown_bridge", max(duration, 6.0), warnings=["Low direct transition score; inserted a short bridge."], fixes=["Use filter and reverb bridge."])
    if bass_overlap:
        fixes.append("Swap low end so both bass lines do not stack.")
        return _transition_spec("bass_swap", max(duration, 6.0), fixes=fixes)
    if vocal_overlap:
        if use_stems:
            fixes.append("Duck or mute the secondary vocal during overlap.")
        else:
            warnings.append("High vocal overlap without stems; using echo_out/mid duck fallback.")
        return _transition_spec("echo_out", duration, warnings=warnings, fixes=fixes, vocal_duck=True)
    if timbre_gap > 0.55 or components.get("timbre", 100) < 58:
        return _transition_spec("filter_sweep", duration, warnings=["Large timbre gap; filter sweep will mask the handoff."])
    if seg_a.get("label") in {"chorus_like", "breakdown_like"} and _float(seg_b.get("energyStart"), seg_b.get("energy", 0.0)) < 0.42:
        return _transition_spec("reverb_tail", duration, fixes=["Let outgoing tail decay under the low-energy entry."])
    if (
        components.get("bpm", 0) >= 86
        and components.get("camelot", 0) >= 82
        and components.get("phrase", 0) >= 82
        and components.get("entryExit", 0) >= 78
        and (seg_b.get("label") == "drop_like" or abs(_float(seg_b.get("energy"), 0.0) - _float(seg_a.get("energy"), 0.0)) < 0.22)
    ):
        return _transition_spec("hard_cut", 0.02, bars=0, fixes=["Use downbeat micro-fade hard cut."])
    return _transition_spec("crossfade", duration)


def best_compatible(source: dict[str, Any], candidates: list[dict[str, Any]], labels: set[str] | None = None, mode: str = "auto", use_stems: bool = True) -> dict[str, Any] | None:
    pool = [item for item in candidates if not labels or item.get("label") in labels] or candidates
    if not pool:
        return None
    return max(pool, key=lambda item: score_segment_transition(source, item, mode, {"useStems": use_stems})["score"] + _float(item.get("mixInScore"), 0.0) * 8)


def curve_average(points: list[dict[str, Any]], key: str, start: float, end: float, fallback: Any = 0.5) -> float:
    values = [float(item.get(key)) for item in points if _is_number(item.get(key)) and start <= float(item.get("time", -1)) < end]
    if values:
        return float(np.mean(values))
    return _float(fallback, 0.5)


def _energy_curve(track: dict[str, Any]) -> list[dict[str, Any]]:
    curve = track.get("energy_curve")
    if isinstance(curve, list) and curve:
        return curve
    profile = track.get("energy_profile")
    if isinstance(profile, list):
        return [item for item in profile if isinstance(item, dict) and _is_number(item.get("time")) and _is_number(item.get("energy"))]
    candidates = track.get("transition_candidates") or {}
    points = []
    for key, time_key in (("intro_energy", "intro"), ("outro_energy", "outro")):
        if _is_number(candidates.get(key)) and _is_number(candidates.get(time_key)):
            points.append({"time": float(candidates[time_key]), "energy": float(candidates[key])})
    return points


def _vocal_density_curve(track: dict[str, Any]) -> list[dict[str, Any]]:
    curve = track.get("vocal_density_curve")
    if isinstance(curve, list) and curve:
        return curve
    candidates = track.get("transition_candidates") or {}
    points = []
    for key, time_key in (("intro_vocal_density", "intro"), ("outro_vocal_density", "outro")):
        if _is_number(candidates.get(key)) and _is_number(candidates.get(time_key)):
            points.append({"time": float(candidates[time_key]), "density": float(candidates[key])})
    return points


def mix_boundary_score(track: dict[str, Any], time_value: float, metrics: dict[str, float], direction: str = "in") -> float:
    candidates = track.get("transition_candidates") or {}
    anchors = []
    if direction == "in":
        anchors.append(candidates.get("intro"))
    elif direction == "out":
        anchors.append(candidates.get("outro"))
    anchors.extend([candidates.get("intro"), candidates.get("outro")])
    anchor_bonus = 0.0
    for anchor in anchors:
        if _is_number(anchor):
            anchor_bonus = max(anchor_bonus, 1.0 - min(1.0, abs(float(anchor) - time_value) / 12.0))
    vocal_key = "vocalStart" if direction == "in" else "vocalEnd"
    energy_key = "energyStart" if direction == "in" else "energyEnd"
    low_vocal = 1.0 - metrics.get(vocal_key, metrics["vocalDensity"])
    moderate_energy = 1.0 - abs(metrics.get(energy_key, metrics["energy"]) - 0.45)
    return _clamp01(low_vocal * 0.48 + moderate_energy * 0.28 + anchor_bonus * 0.24)


def segment_mono(audio: np.ndarray, start: float, end: float, sr: int) -> np.ndarray:
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim == 2 and y.shape[0] <= 8:
        y = np.mean(y, axis=0)
    elif y.ndim == 2:
        y = np.mean(y, axis=1)
    start_sample = max(0, int(start * sr))
    end_sample = max(start_sample, min(y.shape[-1], int(end * sr)))
    return np.ascontiguousarray(y[start_sample:end_sample], dtype=np.float32)


def nearest_index(values: list[float], target: float) -> int:
    if not values:
        return 0
    return int(min(range(len(values)), key=lambda index: abs(values[index] - target)))


def _candidate_boundaries(track: dict[str, Any], bars: list[float], phrases: list[float], bars_per_segment: int, duration: float) -> list[float]:
    if phrases and len(phrases) >= 2:
        boundaries = list(phrases)
    else:
        boundaries = [bars[index] for index in range(0, len(bars) - 1, bars_per_segment)]
        boundaries.append(duration)

    candidates = track.get("transition_candidates") or {}
    for key in ("intro", "outro"):
        value = candidates.get(key)
        if _is_number(value) and _candidate_anchor_is_useful(candidates, key) and _anchor_has_min_bar_room(float(value), boundaries, bars):
            boundaries.append(float(value))
    return _dedupe_times([0.0, *boundaries, duration])


def _candidate_anchor_is_useful(candidates: dict[str, Any], key: str) -> bool:
    confidence = _float(candidates.get("confidence"), 0.0)
    if confidence >= 0.55:
        return True
    vocal = candidates.get(f"{key}_vocal_density")
    energy = candidates.get(f"{key}_energy")
    if _is_number(vocal) and float(vocal) <= 0.48:
        return True
    if _is_number(energy) and float(energy) <= 0.45:
        return True
    return confidence >= 0.35


def _anchor_has_min_bar_room(anchor: float, boundaries: list[float], bars: list[float], min_bars: int = 4) -> bool:
    if not bars:
        return True
    anchor_index = nearest_index(bars, anchor)
    existing = sorted(set(boundaries))
    previous = max((value for value in existing if value < anchor - 0.05), default=None)
    next_value = min((value for value in existing if value > anchor + 0.05), default=None)
    if previous is not None and anchor_index - nearest_index(bars, previous) < min_bars:
        return False
    if next_value is not None and nearest_index(bars, next_value) - anchor_index < min_bars:
        return False
    return True


def _move_boundary_off_vocals(track: dict[str, Any], bars: list[float], boundary: float) -> float:
    if boundary <= 0 or boundary >= bars[-1]:
        return boundary
    vocal_curve = _vocal_density_curve(track)
    local_vocal = curve_average(vocal_curve, "density", boundary - 1.5, boundary + 1.5, fallback=_boundary_vocal_hint(track, boundary))
    if local_vocal < 0.62:
        return boundary
    nearby = [bar for bar in bars if abs(bar - boundary) <= 8.0]
    if not nearby:
        return boundary
    return min(nearby, key=lambda bar: curve_average(vocal_curve, "density", bar - 1.5, bar + 1.5, fallback=_boundary_vocal_hint(track, bar)) + abs(bar - boundary) * 0.02)


def _boundary_vocal_hint(track: dict[str, Any], boundary: float) -> float:
    candidates = track.get("transition_candidates") or {}
    values = []
    for key, density_key in (("intro", "intro_vocal_density"), ("outro", "outro_vocal_density")):
        if _is_number(candidates.get(key)) and _is_number(candidates.get(density_key)):
            distance = abs(float(candidates[key]) - boundary)
            if distance <= 4.0:
                values.append((distance, float(candidates[density_key])))
    if values:
        return min(values, key=lambda item: item[0])[1]
    return _section_vocal_hint(track, max(0.0, boundary - 1.0), boundary + 1.0)


def _snap_boundaries(boundaries: list[float], bars: list[float], phrases: list[float], duration: float) -> list[float]:
    snapped = []
    snap_pool = sorted(set([*bars, *phrases, 0.0, duration]))
    for boundary in boundaries:
        snapped.append(float(min(snap_pool, key=lambda value: abs(value - boundary))))
    return _dedupe_times(snapped)


def _segment_ranges(boundaries: list[float], bars: list[float], duration: float, bars_per_segment: int) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    min_bars = 4
    max_bars = 32
    for start, end in zip(boundaries, boundaries[1:]):
        start_index = nearest_index(bars, start)
        end_index = nearest_index(bars, end)
        bar_count = max(1, end_index - start_index)
        if bar_count < min_bars:
            continue
        if bar_count > max_bars:
            cursor = start_index
            while cursor < end_index:
                next_index = min(end_index, cursor + bars_per_segment)
                if next_index - cursor >= min_bars:
                    ranges.append((bars[cursor], min(bars[next_index], duration)))
                cursor = next_index
        else:
            ranges.append((max(0.0, start), min(end, duration)))
    if not ranges and len(bars) > 1:
        step = min(max(bars_per_segment, min_bars), max_bars)
        for start_index in range(0, len(bars) - 1, step):
            end_index = min(len(bars) - 1, start_index + step)
            if end_index - start_index >= min_bars:
                ranges.append((bars[start_index], min(bars[end_index], duration)))
    return [(round(start, 4), round(end, 4)) for start, end in ranges if end - start > 0.1]


def _score_bpm(a: Any, b: Any) -> tuple[float, float, str | None]:
    bpm_a = _float(a, 0.0)
    bpm_b = _float(b, 0.0)
    if bpm_a <= 0 or bpm_b <= 0:
        return 55.0, 1.0, "Unknown BPM; rhythm score is neutral."
    ratio = bpm_a / bpm_b
    stretch = 1.0 / ratio
    distance = abs(1.0 - min(max(stretch, 1 / stretch), 10.0))
    if 0.97 <= stretch <= 1.03:
        return 100.0 - distance * 500, stretch, None
    if 0.94 <= stretch <= 1.06:
        return max(78.0, 94.0 - distance * 420), stretch, None
    if 0.88 <= stretch <= 1.12:
        return max(48.0, 78.0 - distance * 300), stretch, "BPM stretch is audible but still inside render limits."
    return 24.0, stretch, "BPM stretch exceeds 0.88x-1.12x; transition is high risk."


def _score_camelot(a: Any, b: Any) -> tuple[float, str | None, str | None]:
    result = camelot_key_distance(a, b)
    relation = result.get("relation")
    if relation == "same":
        return 100.0, None, None
    if relation == "adjacent":
        return 90.0, None, None
    if relation == "relative_major_minor":
        return 85.0, None, None
    if relation in {"energy_boost", "diagonal_mix", "mood_shifter", "jaws_mix"}:
        return 65.0, f"Camelot relation {relation} needs a shorter or more effected transition.", "Use short blend or harmonic tuning."
    if relation == "unknown":
        return 45.0, "Unknown Camelot key; avoid long harmonic blends.", None
    return 28.0, "Camelot clash; avoid long tonal overlap.", "Consider pitch/tuning or hard cut."


def _score_phrase(seg_a: dict[str, Any], seg_b: dict[str, Any]) -> float:
    score = 100.0
    if not seg_a.get("isCleanExit"):
        score -= 20.0
    if not seg_b.get("isCleanEntry"):
        score -= 20.0
    if int(seg_a.get("barEnd", 0)) <= int(seg_a.get("barStart", 0)):
        score -= 15.0
    if int(seg_b.get("barStart", 0)) < 0:
        score -= 10.0
    return max(0.0, score)


def _score_energy(seg_a: dict[str, Any], seg_b: dict[str, Any], mode: str) -> tuple[float, str | None]:
    delta = _float(seg_b.get("energyStart"), seg_b.get("energy", 0.0)) - _float(seg_a.get("energyEnd"), seg_a.get("energy", 0.0))
    if mode == "energy_build" or seg_b.get("label") == "drop_like":
        ideal = 0.18
        score = 100.0 - abs(delta - ideal) * 180.0
    elif delta < -0.35:
        return 38.0, "Energy drops abruptly at this transition."
    else:
        score = 100.0 - abs(delta - 0.02) * 150.0
    return float(np.clip(score, 0, 100)), None


def _score_vocal_conflict(seg_a: dict[str, Any], seg_b: dict[str, Any], use_stems: bool) -> tuple[float, str | None, str | None]:
    conflict = _float(seg_a.get("vocalEnd"), seg_a.get("vocalDensity", 0.0)) * _float(seg_b.get("vocalStart"), seg_b.get("vocalDensity", 0.0))
    score = 100.0 - conflict * 145.0
    if conflict > 0.34:
        if use_stems:
            return max(45.0, score), "High vocal overlap; will use stem duck/mute.", "Duck secondary vocals by about 12 dB."
        return max(22.0, score - 18.0), "High vocal overlap without stems.", "Use echo_out or mid duck fallback."
    return max(0.0, score), None, None


def _score_bass_conflict(seg_a: dict[str, Any], seg_b: dict[str, Any], use_stems: bool) -> tuple[float, str | None, str | None]:
    conflict = _float(seg_a.get("bassEnergy"), 0.0) * _float(seg_b.get("bassEnergy"), 0.0)
    score = 100.0 - conflict * 90.0
    if conflict > 0.35:
        return max(45.0, score), "Both segments have strong low end.", "Use bass_swap" if use_stems else "Use highpass/low-shelf bass swap simulation."
    return max(0.0, score), None, None


def _score_timbre(seg_a: dict[str, Any], seg_b: dict[str, Any]) -> tuple[float, str | None, str | None]:
    diff = abs(_float(seg_a.get("brightness"), 0.0) - _float(seg_b.get("brightness"), 0.0)) + abs(_float(seg_a.get("bassEnergy"), 0.0) - _float(seg_b.get("bassEnergy"), 0.0))
    score = 100.0 - diff * 90.0
    if diff > 0.55:
        return max(35.0, score), "Large timbre difference.", "Use filter_sweep or breakdown_bridge."
    return max(0.0, score), None, None


def _score_entry_exit(seg_a: dict[str, Any], seg_b: dict[str, Any]) -> float:
    score = 42.0
    score += _float(seg_a.get("mixOutScore"), 0.5) * 30.0
    score += _float(seg_b.get("mixInScore"), 0.5) * 30.0
    if seg_a.get("isCleanExit"):
        score += 8.0
    if seg_b.get("isCleanEntry"):
        score += 8.0
    return float(np.clip(score, 0, 100))


def _transition_spec(kind: str, duration: float, *, bars: int | None = None, warnings: list[str] | None = None, fixes: list[str] | None = None, vocal_duck: bool = False) -> dict[str, Any]:
    duration = max(0.02, float(duration))
    bars = int(bars if bars is not None else max(1, round(duration / 2.0)))
    automation = {
        "a_gain": [[0, 0], [1, -6 if kind != "hard_cut" else 0]],
        "b_gain": [[0, -8 if kind != "hard_cut" else 0], [1, 0]],
        "vocal_duck": bool(vocal_duck),
    }
    if kind == "bass_swap":
        automation.update(
            {
                "a_bass_gain": [[0, 0], [0.5, -12], [1, -60]],
                "b_bass_gain": [[0, -60], [0.5, -9], [1, 0]],
                "a_filter": "lowpass_down",
                "b_filter": "highpass_open",
            }
        )
    if kind == "filter_sweep":
        automation.update({"a_filter": "lowpass_down", "b_filter": "highpass_open"})
    if kind in {"echo_out", "reverb_tail", "breakdown_bridge"}:
        automation.update({"tail": kind, "tailWet": 0.22 if kind == "echo_out" else 0.18})
    return {
        "type": kind,
        "durationSec": round(duration, 3),
        "bars": bars,
        "automation": automation,
        "warnings": warnings or [],
        "fixes": fixes or [],
    }


def _transition_duration(seg_a: dict[str, Any], seg_b: dict[str, Any]) -> float:
    bpm = _float(seg_b.get("bpm"), _float(seg_a.get("bpm"), 120.0))
    bar_seconds = 240.0 / max(bpm, 1.0)
    return float(np.clip(bar_seconds * 2, 2.0, 8.0))


def segment_risk_flags(metrics: dict[str, float], mix_in: float, mix_out: float) -> list[str]:
    flags = []
    if metrics["vocalStart"] > 0.62:
        flags.append("vocal_entry")
    if metrics["vocalEnd"] > 0.62:
        flags.append("vocal_exit")
    if metrics["bassEnergy"] > 0.72:
        flags.append("heavy_bass")
    if mix_in < 0.42:
        flags.append("rough_entry")
    if mix_out < 0.42:
        flags.append("rough_exit")
    return flags


def _section_vocal_hint(track: dict[str, Any], start: float, end: float) -> float:
    best = 0.45
    best_overlap = 0.0
    hints = {"chorus": 0.62, "verse": 0.56, "bridge": 0.48, "drop": 0.24, "intro": 0.22, "outro": 0.24, "breakdown": 0.18}
    for section in track.get("sections") or []:
        overlap = min(end, _float(section.get("endTime"), 0.0)) - max(start, _float(section.get("startTime"), 0.0))
        if overlap > best_overlap:
            best_overlap = overlap
            best = hints.get(str(section.get("type") or "").lower(), 0.45)
    return best


def _dedupe_times(values: list[float]) -> list[float]:
    result: list[float] = []
    for value in sorted(float(item) for item in values if _is_number(item)):
        if not result or abs(value - result[-1]) > 0.05:
            result.append(value)
    return result


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _float(value: Any, fallback: Any = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else float(fallback)
    except (TypeError, ValueError):
        try:
            return float(fallback)
        except (TypeError, ValueError):
            return 0.0


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))
