from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from scipy import signal


SAMPLE_RATE = 44100
STEM_NAMES = ("vocals", "drums", "bass", "other")


@dataclass
class BarFeature:
    barIndex: int
    start: float
    end: float
    duration: float
    rms: float
    lufsApprox: float
    energy: float
    energyDelta: float
    chroma: list[float]
    mfcc: list[float]
    spectralCentroid: float
    spectralContrast: list[float]
    spectralFlux: float
    onsetStrength: float
    drumActivity: float
    bassEnergy: float
    vocalEnergy: float
    vocalDensity: float
    otherEnergy: float
    harmonicChange: float
    timbreChange: float
    rhythmChange: float
    isDownbeatStrong: bool
    isLikelyPickup: bool
    isLikelyVocalEntry: bool
    isLikelyVocalExit: bool


@dataclass
class BoundaryCandidate:
    time: float
    barIndex: int
    score: float
    sourceScores: dict[str, float]
    boundaryType: str
    riskFlags: list[str] = field(default_factory=list)
    snapReason: str = "bar"


@dataclass
class StructuralSection:
    id: str
    trackId: str
    source: str
    start: float
    end: float
    barStart: int
    barEnd: int
    bars: int
    label: str
    confidence: float
    meanEnergy: float
    energyShape: str
    vocalDensity: float
    drumActivity: float
    bassEnergy: float
    brightness: float
    loopability: float
    entryClean: bool
    exitClean: bool
    mixInScore: float
    mixOutScore: float
    grooveBedScore: float
    vocalPhraseScore: float
    similarSectionIds: list[str] = field(default_factory=list)
    riskFlags: list[str] = field(default_factory=list)
    level: str = "major"


@dataclass
class VocalPhrase:
    id: str
    trackId: str
    source: str
    sectionId: str
    start: float
    end: float
    barStart: int
    barEnd: int
    bars: int
    pickupStart: float | None
    tailEnd: float | None
    mainStart: float
    mainEnd: float
    vocalEnergy: float
    lyricDensity: float
    phraseCompleteness: float
    entryClean: bool
    exitClean: bool
    hasPickup: bool
    hasTail: bool
    score: float
    riskFlags: list[str] = field(default_factory=list)


@dataclass
class GrooveBedCandidate:
    id: str
    trackId: str
    source: str
    sectionId: str
    start: float
    end: float
    barStart: int
    barEnd: int
    bars: int
    usesStems: list[str]
    vocalLeakage: float
    drumActivity: float
    bassStability: float
    loopability: float
    energy: float
    camelot: str | None
    bpm: float
    score: float
    riskFlags: list[str] = field(default_factory=list)


def analyze_track_segmentation(
    track: dict[str, Any],
    source: str = "A",
    *,
    audio: np.ndarray | None = None,
    sr: int = SAMPLE_RATE,
    stems: dict[str, Any] | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    if audio is None:
        audio = _safe_load_audio(track.get("path"), sr)
    audio = _mono(audio)
    stem_audio = _load_stems(stems, sr)
    if not stem_audio:
        warnings.append("Vocals/drums/bass/other stems missing; segmentation confidence reduced.")

    bar_features = extract_bar_features(audio, sr, track, stem_audio or None)
    matrices = compute_similarity_matrices(bar_features)
    novelty = compute_novelty_curve(matrices["fused"], [4, 8, 16])
    boundaries = generate_boundary_candidates(bar_features, novelty, track)
    sections = build_hierarchical_sections(boundaries, bar_features, matrices, track=track, source=source)
    minor_sections = build_minor_sections(boundaries, bar_features, matrices, track=track, source=source)
    phrase_sections = minor_sections or sections
    vocal_phrases = extract_vocal_phrases_from_sections(phrase_sections, bar_features, stem_audio.get("vocals") if stem_audio else None, track)
    groove_beds = extract_groove_bed_candidates(sections, bar_features, stem_audio or None, track)
    safe_points = extract_transition_safe_points(sections, bar_features, track)
    return {
        "method": "multi_scale_ssm_novelty_stem_corrected",
        "barsDetected": len(bar_features) >= 2,
        "stemsUsed": bool(stem_audio),
        "boundaryCount": len(boundaries),
        "barFeatures": [asdict(item) for item in bar_features],
        "sections": sections,
        "minorSections": minor_sections,
        "vocalPhrases": vocal_phrases,
        "grooveBedCandidates": groove_beds,
        "safeCutPoints": safe_points,
        "warnings": warnings,
        "debug": {
            "noveltyPeaks": _novelty_peaks(novelty.get("fused_novelty", [])),
            "boundaryCandidates": boundaries[:24],
            "ssmShape": list(matrices["fused"].shape),
            "majorSectionCount": len(sections),
            "minorSectionCount": len(minor_sections),
        },
    }


def extract_bar_features(audio: np.ndarray | None, sr: int, analysis: dict[str, Any], stems: dict[str, Any] | None = None) -> list[BarFeature]:
    duration = float(analysis.get("duration") or (librosa.get_duration(y=audio, sr=sr) if audio is not None and audio.size else 0.0))
    if duration <= 0:
        return []
    bars = _normalized_bars(analysis, duration)
    audio = _fit_audio(_mono(audio), int(duration * sr))
    stem_audio = {name: _fit_audio(_mono(value), audio.size) for name, value in (stems or {}).items() if value is not None}
    features: list[BarFeature] = []
    prev: dict[str, Any] | None = None
    energy_curve = analysis.get("energy_curve") or analysis.get("energy_profile") or []
    vocal_curve = analysis.get("vocal_density_curve") or (analysis.get("transition_candidates") or {}).get("vocal_density_curve") or []
    for index in range(len(bars) - 1):
        start = float(bars[index])
        end = float(bars[index + 1])
        clip = _slice(audio, start, end, sr)
        rms = _rms(clip)
        energy = _clamp01(max(rms * 8.0, _curve_avg(energy_curve, "energy", start, end, rms * 8.0)))
        chroma = _mean_feature(lambda y: librosa.feature.chroma_stft(y=y, sr=sr, n_fft=min(4096, _n_fft(y))), clip, 12)
        mfcc = _mean_feature(lambda y: librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=min(4096, _n_fft(y))), clip, 13)
        contrast = _mean_feature(lambda y: librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=min(4096, _n_fft(y))), clip, 7)
        centroid = _scalar_feature(lambda y: librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=min(4096, _n_fft(y))), clip)
        onset = _scalar_feature(lambda y: librosa.onset.onset_strength(y=y, sr=sr), clip)
        flux = _spectral_flux(clip, sr)
        vocal_energy = _stem_energy(stem_audio.get("vocals"), start, end, sr)
        drum_energy = _stem_energy(stem_audio.get("drums"), start, end, sr)
        bass_energy = _low_band_energy(stem_audio.get("bass"), start, end, sr) if stem_audio.get("bass") is not None else _low_band_energy(audio, start, end, sr)
        other_energy = _stem_energy(stem_audio.get("other"), start, end, sr)
        vocal_density = _vocal_density(vocal_energy, rms, _curve_avg(vocal_curve, "density", start, end, 0.0), stem_audio.get("vocals") is not None)
        current = {
            "energy": energy,
            "chroma": chroma,
            "mfcc": mfcc,
            "onset": onset,
            "vocal": vocal_density,
            "drum": drum_energy,
            "bass": bass_energy,
        }
        harmonic_change = _distance(chroma, prev["chroma"]) if prev else 0.0
        timbre_change = _distance(mfcc, prev["mfcc"]) if prev else 0.0
        rhythm_change = abs(onset - prev["onset"]) if prev else 0.0
        energy_delta = energy - float(prev["energy"]) if prev else 0.0
        features.append(
            BarFeature(
                barIndex=index,
                start=round(start, 3),
                end=round(end, 3),
                duration=round(end - start, 3),
                rms=round(rms, 6),
                lufsApprox=round(20 * math.log10(max(rms, 1e-6)) - 0.691, 3),
                energy=round(float(energy), 4),
                energyDelta=round(float(np.clip(energy_delta, -1, 1)), 4),
                chroma=[round(float(v), 5) for v in chroma],
                mfcc=[round(float(v), 5) for v in mfcc],
                spectralCentroid=round(float(centroid), 3),
                spectralContrast=[round(float(v), 5) for v in contrast],
                spectralFlux=round(float(flux), 5),
                onsetStrength=round(float(onset), 5),
                drumActivity=round(float(_clamp01(max(drum_energy * 7.0, onset * 0.65))), 4),
                bassEnergy=round(float(_clamp01(bass_energy * 10.0)), 4),
                vocalEnergy=round(float(_clamp01(vocal_energy * 8.0)), 4),
                vocalDensity=round(float(vocal_density), 4),
                otherEnergy=round(float(_clamp01(other_energy * 8.0)), 4),
                harmonicChange=round(float(_clamp01(harmonic_change)), 4),
                timbreChange=round(float(_clamp01(timbre_change)), 4),
                rhythmChange=round(float(_clamp01(rhythm_change)), 4),
                isDownbeatStrong=bool(onset > 0.08 or index % 4 == 0),
                isLikelyPickup=False,
                isLikelyVocalEntry=False,
                isLikelyVocalExit=False,
            )
        )
        prev = current
    _mark_vocal_events(features)
    return features


def compute_similarity_matrices(bar_features: list[BarFeature]) -> dict[str, np.ndarray]:
    if not bar_features:
        empty = np.zeros((0, 0), dtype=np.float32)
        return {"harmonic": empty, "timbre": empty, "rhythm": empty, "energy": empty, "fused": empty}
    harmonic = _cosine_matrix(np.asarray([item.chroma for item in bar_features], dtype=np.float32))
    timbre = _cosine_matrix(np.asarray([item.mfcc + item.spectralContrast + [item.spectralCentroid / 8000.0] for item in bar_features], dtype=np.float32))
    rhythm = _cosine_matrix(np.asarray([[item.onsetStrength, item.drumActivity, item.rhythmChange] for item in bar_features], dtype=np.float32))
    energy = _cosine_matrix(np.asarray([[item.energy, item.energyDelta, item.bassEnergy, item.drumActivity, item.vocalDensity] for item in bar_features], dtype=np.float32))
    fused = harmonic * 0.30 + timbre * 0.25 + rhythm * 0.25 + energy * 0.20
    return {
        "harmonic": _smooth_matrix(harmonic),
        "timbre": _smooth_matrix(timbre),
        "rhythm": _smooth_matrix(rhythm),
        "energy": _smooth_matrix(energy),
        "fused": _smooth_matrix(fused),
    }


def compute_novelty_curve(fused_ssm: np.ndarray, kernel_sizes: list[int] | None = None) -> dict[str, list[float]]:
    kernel_sizes = kernel_sizes or [4, 8, 16]
    n = int(fused_ssm.shape[0]) if fused_ssm.ndim == 2 else 0
    result: dict[str, list[float]] = {}
    curves = []
    for size in kernel_sizes:
        curve = np.zeros(n, dtype=np.float32)
        half = max(1, int(size // 2))
        for index in range(half, max(half, n - half)):
            before = fused_ssm[index - half : index, index - half : index]
            after = fused_ssm[index : index + half, index : index + half]
            cross_a = fused_ssm[index - half : index, index : index + half]
            cross_b = fused_ssm[index : index + half, index - half : index]
            curve[index] = max(0.0, float(np.mean(before) + np.mean(after) - np.mean(cross_a) - np.mean(cross_b)))
        curve = _normalize_1d(_smooth_1d(curve, max(3, half | 1)))
        result[f"novelty_{size}"] = [round(float(v), 5) for v in curve]
        curves.append(curve)
    fused = _normalize_1d(np.mean(curves, axis=0)) if curves else np.zeros(n, dtype=np.float32)
    result["fused_novelty"] = [round(float(v), 5) for v in fused]
    return result


def generate_boundary_candidates(bar_features: list[BarFeature], novelty: dict[str, list[float]], analysis: dict[str, Any]) -> list[dict[str, Any]]:
    if not bar_features:
        return []
    bars = [item.start for item in bar_features] + [bar_features[-1].end]
    phrases = _normalized_phrases(analysis, bars)
    transition_candidates = analysis.get("transition_candidates") or {}
    fused = novelty.get("fused_novelty", [])
    candidates: list[BoundaryCandidate] = []
    for index in range(1, len(bar_features)):
        current = bar_features[index]
        previous = bar_features[index - 1]
        time = current.start
        novelty_score = float(fused[index]) if index < len(fused) else 0.0
        energy_change = abs(current.energy - previous.energy)
        vocal_change = max(0.0, current.vocalDensity - previous.vocalDensity, previous.vocalDensity - current.vocalDensity)
        drum_bass_change = 0.5 * abs(current.drumActivity - previous.drumActivity) + 0.5 * abs(current.bassEnergy - previous.bassEnergy)
        phrase_score = 1.0 if any(abs(time - phrase) <= 0.08 for phrase in phrases) else 0.0
        transition_score = _transition_candidate_score(time, transition_candidates)
        sustained_vocal = previous.vocalDensity > 0.55 and current.vocalDensity > 0.55 and vocal_change < 0.22
        score = (
            0.28 * novelty_score
            + 0.18 * energy_change
            + 0.18 * vocal_change
            + 0.16 * drum_bass_change
            + 0.12 * phrase_score
            + 0.08 * transition_score
        )
        risk_flags = []
        if sustained_vocal:
            score -= 0.22
            risk_flags.append("cuts_vocal_phrase")
        if not current.isDownbeatStrong and index % 4 != 0:
            risk_flags.append("weak_downbeat")
        score = float(np.clip(score, 0, 1))
        boundary_type = "major" if score >= 0.58 or novelty_score > 0.72 else "minor"
        if vocal_change > 0.38:
            boundary_type = "vocal_phrase"
        if transition_score > 0.6:
            boundary_type = "transition_safe"
        candidates.append(
            BoundaryCandidate(
                time=round(float(time), 3),
                barIndex=index,
                score=round(score, 4),
                sourceScores={
                    "novelty": round(novelty_score, 4),
                    "selfSimilarity": round(novelty_score, 4),
                    "energyChange": round(float(energy_change), 4),
                    "vocalEntryExit": round(float(vocal_change), 4),
                    "drumBassChange": round(float(drum_bass_change), 4),
                    "transitionCandidate": round(float(transition_score), 4),
                    "phraseBoundary": round(float(phrase_score), 4),
                },
                boundaryType=boundary_type,
                riskFlags=risk_flags,
                snapReason="transition_candidate" if transition_score > 0.6 else "phrase" if phrase_score else "bar",
            )
        )
    candidates.extend(_explicit_boundary_candidates(bar_features, phrases, transition_candidates))
    selected = _nms_boundaries(candidates, min_distance_bars=2)
    return [asdict(item) for item in selected]


def snap_boundary_to_musical_grid(candidate: dict[str, Any], bars: list[float], phrases: list[float], transition_candidates: dict[str, Any]) -> dict[str, Any]:
    time = float(candidate.get("time", 0.0))
    for key in ("intro", "outro", "drop"):
        value = transition_candidates.get(key)
        if _is_number(value) and abs(float(value) - time) <= 1.5 and float(transition_candidates.get("confidence", 0.0)) >= 0.5:
            candidate["time"] = round(float(value), 3)
            candidate["snapReason"] = "transition_candidate"
            return candidate
    if phrases:
        phrase = min(phrases, key=lambda value: abs(value - time))
        if abs(phrase - time) <= 1.0:
            candidate["time"] = round(float(phrase), 3)
            candidate["snapReason"] = "phrase"
            return candidate
    if bars:
        bar = min(bars, key=lambda value: abs(value - time))
        candidate["time"] = round(float(bar), 3)
        candidate["snapReason"] = "bar"
    return candidate


def build_hierarchical_sections(
    boundaries: list[dict[str, Any]],
    bar_features: list[BarFeature],
    similarity_matrices: dict[str, np.ndarray],
    min_bars: int = 4,
    preferred_bars: list[int] | None = None,
    *,
    track: dict[str, Any] | None = None,
    source: str = "A",
) -> list[dict[str, Any]]:
    preferred_bars = preferred_bars or [8, 16, 32]
    if not bar_features:
        return []
    track = track or {}
    n = len(bar_features)
    selected = [0, n]
    for item in boundaries:
        index = int(item.get("barIndex", 0))
        if index <= 0 or index >= n:
            continue
        score = float(item.get("score", 0.0))
        if score >= 0.42 or item.get("boundaryType") in {"major", "transition_safe", "vocal_phrase"}:
            selected.append(index)
    selected = _repair_boundaries(sorted(set(selected)), n, min_bars, preferred_bars)
    sections = []
    for start_idx, end_idx in zip(selected, selected[1:]):
        if end_idx - start_idx < min_bars:
            continue
        section = _make_section(track, source, bar_features, similarity_matrices, start_idx, end_idx, len(sections) + 1, "major")
        sections.append(section)
    if not sections:
        step = min(preferred_bars[0], max(min_bars, n))
        for start_idx in range(0, n, step):
            end_idx = min(n, start_idx + step)
            if end_idx - start_idx >= min_bars:
                sections.append(_make_section(track, source, bar_features, similarity_matrices, start_idx, end_idx, len(sections) + 1, "major"))
    return sections


def build_minor_sections(
    boundaries: list[dict[str, Any]],
    bar_features: list[BarFeature],
    similarity_matrices: dict[str, np.ndarray],
    min_bars: int = 2,
    *,
    track: dict[str, Any] | None = None,
    source: str = "A",
) -> list[dict[str, Any]]:
    """Build 2/4/8 bar micro sections for phrase and bed extraction.

    Major sections stay conservative for the UI and old segment API. These
    minor sections are intentionally more granular so the mashup engine can
    choose lyric phrases and loopable beds without falling back to whole
    verse/chorus-sized blocks.
    """
    if not bar_features:
        return []
    track = track or {}
    n = len(bar_features)
    points = {0, n}
    for item in boundaries:
        index = int(item.get("barIndex", 0))
        if 0 < index < n and (float(item.get("score", 0.0)) >= 0.30 or item.get("boundaryType") in {"vocal_phrase", "transition_safe"}):
            points.add(index)
    selected = sorted(points)
    repaired = [selected[0]]
    for value in selected[1:]:
        while value - repaired[-1] > 8:
            repaired.append(repaired[-1] + 4)
        if value - repaired[-1] >= min_bars:
            repaired.append(value)
    if repaired[-1] != n:
        repaired.append(n)
    sections = []
    for start_idx, end_idx in zip(repaired, repaired[1:]):
        if end_idx - start_idx < min_bars:
            continue
        sections.append(_make_section(track, source, bar_features, similarity_matrices, start_idx, end_idx, len(sections) + 1, "minor"))
    return sections


def detect_vocal_activity_curve(vocals_stem: Any, sr: int = SAMPLE_RATE) -> dict[str, Any]:
    audio = _mono(_load_audio_like(vocals_stem, sr))
    if audio.size == 0:
        return {"times": [], "energy": [], "active": [], "threshold": 0.0}
    hop = 512
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop)[0]
    energy = _normalize_1d(rms) * 0.82 + _normalize_1d(centroid) * 0.18
    energy = _smooth_1d(energy, 9)
    threshold = max(float(np.percentile(energy, 58)) * 0.75, float(np.mean(energy)) * 0.72, 0.03)
    active = energy > threshold
    times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop)
    return {"times": times.tolist(), "energy": energy.tolist(), "active": active.tolist(), "threshold": threshold}


def detect_vocal_phrase_boundaries(activity_curve: dict[str, Any], bars: list[float]) -> list[dict[str, Any]]:
    times = np.asarray(activity_curve.get("times") or [], dtype=np.float32)
    active = np.asarray(activity_curve.get("active") or [], dtype=bool)
    if times.size == 0 or not bars:
        return []
    result = []
    for index in range(len(bars) - 1):
        start = bars[index]
        end = bars[index + 1]
        mask = (times >= start) & (times < end)
        density = float(np.mean(active[mask])) if np.any(mask) else 0.0
        result.append({"barIndex": index, "start": start, "end": end, "density": density})
    return result


def extract_vocal_phrases_from_sections(
    sections: list[dict[str, Any]],
    bar_features: list[BarFeature],
    vocals_stem: Any,
    analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    if not sections or not bar_features:
        return []
    bars = [item.start for item in bar_features] + [bar_features[-1].end]
    activity = detect_vocal_activity_curve(vocals_stem) if vocals_stem is not None else {"times": [], "energy": [], "active": []}
    phrases: list[VocalPhrase] = []
    source = str(sections[0].get("source") or analysis.get("source") or "A")
    track_id = str(analysis.get("id") or "")
    for section in sections:
        if float(section.get("vocalDensity", 0.0)) < 0.18 and not _section_has_vocal_bars(bar_features, section):
            continue
        for start_bar, end_bar in _candidate_vocal_windows(section, bar_features, bars, activity):
            phrase = _score_vocal_phrase(activity, bar_features, bars, section, start_bar, end_bar, track_id, source, len(phrases) + 1)
            if phrase and phrase.score >= 38:
                phrases.append(phrase)
    phrases = _dedupe_phrases(phrases)
    return [asdict(item) for item in phrases[:32]]


def extract_groove_bed_candidates(
    sections: list[dict[str, Any]],
    bar_features: list[BarFeature],
    stems: dict[str, Any] | None,
    analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    if not stems or not all(name in stems for name in ("drums", "bass", "other")):
        return []
    candidates: list[GrooveBedCandidate] = []
    source = str(analysis.get("source") or (sections[0].get("source") if sections else "A"))
    track_id = str(analysis.get("id") or "")
    for section in sections:
        if int(section.get("bars", 0)) < 4:
            continue
        for start_idx, end_idx in _candidate_bed_windows(section, bar_features):
            candidate = _make_groove_bed_candidate(section, bar_features, analysis, source, track_id, start_idx, end_idx, len(candidates) + 1)
            if candidate.score >= 34:
                candidates.append(candidate)
    candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
    return [asdict(item) for item in candidates[:12]]


def extract_transition_safe_points(sections: list[dict[str, Any]], bar_features: list[BarFeature], analysis: dict[str, Any]) -> list[dict[str, Any]]:
    points = []
    transition_candidates = analysis.get("transition_candidates") or {}
    for index, bar in enumerate(bar_features):
        low_vocal = 1.0 - bar.vocalDensity
        downbeat = 1.0 if bar.isDownbeatStrong or index % 4 == 0 else 0.35
        energy_intent = min(1.0, abs(bar.energyDelta) * 1.7 + 0.45)
        phrase = 1.0 if any(abs(bar.start - section.get("start", -999)) < 0.05 or abs(bar.start - section.get("end", -999)) < 0.05 for section in sections) else 0.0
        tc = _transition_candidate_score(bar.start, transition_candidates)
        score = 0.35 * low_vocal + 0.22 * downbeat + 0.16 * energy_intent + 0.17 * phrase + 0.10 * tc
        risks = []
        if bar.vocalDensity > 0.55:
            risks.append("mid_vocal_cut")
            score -= 0.28
        if downbeat < 0.7:
            risks.append("weak_downbeat")
        score = float(np.clip(score, 0, 1))
        kind = "mix_in" if index == 0 else "mix_out"
        if bar.isLikelyVocalEntry:
            kind = "vocal_entry"
        elif bar.isLikelyVocalExit:
            kind = "vocal_exit"
        elif bar.energyDelta > 0.20 and bar.drumActivity > 0.55:
            kind = "drop_entry"
        elif bar.energyDelta < -0.20:
            kind = "breakdown_entry"
        points.append({"time": bar.start, "barIndex": index, "type": kind, "score": round(score, 4), "reasons": _safe_point_reasons(bar, score, tc, phrase), "riskFlags": risks})
    return sorted(points, key=lambda item: item["score"], reverse=True)[:32]


def _make_section(track: dict[str, Any], source: str, bar_features: list[BarFeature], matrices: dict[str, np.ndarray], start_idx: int, end_idx: int, number: int, level: str) -> dict[str, Any]:
    subset = bar_features[start_idx:end_idx]
    energies = np.asarray([item.energy for item in subset], dtype=np.float32)
    vocals = np.asarray([item.vocalDensity for item in subset], dtype=np.float32)
    drums = np.asarray([item.drumActivity for item in subset], dtype=np.float32)
    bass = np.asarray([item.bassEnergy for item in subset], dtype=np.float32)
    brightness = np.asarray([item.spectralCentroid / 8000.0 for item in subset], dtype=np.float32)
    loopability = _feature_loopability(subset, matrices)
    energy_shape = _energy_shape(energies)
    label, confidence = _label_section(start_idx, end_idx, len(bar_features), energies, vocals, drums, bass, brightness, energy_shape, loopability, matrices)
    entry_clean = subset[0].vocalDensity < 0.55
    exit_clean = subset[-1].vocalDensity < 0.55
    mix_in = _mix_score(subset[0], entry=True)
    mix_out = _mix_score(subset[-1], entry=False)
    risk_flags = []
    if not entry_clean:
        risk_flags.append("vocal_entry")
    if not exit_clean:
        risk_flags.append("vocal_exit")
    if loopability < 0.35:
        risk_flags.append("low_loopability")
    section = StructuralSection(
        id=f"{source}_sec_{number:03d}",
        trackId=str(track.get("id") or ""),
        source=source,
        start=round(subset[0].start, 3),
        end=round(subset[-1].end, 3),
        barStart=start_idx,
        barEnd=end_idx,
        bars=end_idx - start_idx,
        label=label,
        confidence=round(confidence, 4),
        meanEnergy=round(float(np.mean(energies)), 4),
        energyShape=energy_shape,
        vocalDensity=round(float(np.mean(vocals)), 4),
        drumActivity=round(float(np.mean(drums)), 4),
        bassEnergy=round(float(np.mean(bass)), 4),
        brightness=round(float(np.mean(brightness)), 4),
        loopability=round(loopability, 4),
        entryClean=entry_clean,
        exitClean=exit_clean,
        mixInScore=round(mix_in, 4),
        mixOutScore=round(mix_out, 4),
        grooveBedScore=round(float((1 - np.mean(vocals)) * 0.35 + np.mean(drums) * 0.25 + np.mean(bass) * 0.20 + loopability * 0.20), 4),
        vocalPhraseScore=round(float(np.mean(vocals) * 0.55 + entry_clean * 0.18 + exit_clean * 0.18 + min(0.09, len(subset) / 32)), 4),
        similarSectionIds=[],
        riskFlags=risk_flags,
        level=level,
    )
    return asdict(section)


def _label_section(start_idx: int, end_idx: int, total: int, energy: np.ndarray, vocal: np.ndarray, drums: np.ndarray, bass: np.ndarray, brightness: np.ndarray, shape: str, loopability: float, matrices: dict[str, np.ndarray]) -> tuple[str, float]:
    pos = start_idx / max(total, 1)
    mean_energy = float(np.mean(energy))
    mean_vocal = float(np.mean(vocal))
    mean_drums = float(np.mean(drums))
    mean_bass = float(np.mean(bass))
    mean_bright = float(np.mean(brightness))
    confidence = 0.52
    if pos < 0.15 and mean_vocal < 0.30 and mean_energy < 0.58:
        return "intro_like", 0.78
    if pos > 0.82 and mean_vocal < 0.35 and (shape == "falling" or mean_energy < 0.60):
        return "outro_like", 0.76
    if mean_drums > 0.58 and mean_bass > 0.55 and mean_energy > 0.58 and mean_vocal < 0.42:
        return "drop_like", 0.76
    if mean_vocal > 0.50 and mean_energy > 0.52 and mean_bright > 0.25:
        repetition = _section_repetition(start_idx, end_idx, matrices.get("fused"))
        return "chorus_like" if repetition > 0.58 else "verse_like", 0.70 + min(0.12, repetition * 0.1)
    if mean_vocal > 0.44 and shape == "rising":
        return "pre_chorus_like", 0.68
    if mean_drums < 0.34 and mean_bass < 0.38 and mean_energy < 0.50:
        return "breakdown_like", 0.70
    if float(np.mean([item for item in (np.std(energy), np.std(brightness))])) > 0.18:
        return "bridge_like", 0.62
    if confidence < 0.55:
        return "unknown", confidence
    return "verse_like" if mean_vocal >= 0.32 else "bridge_like", 0.58


def _repair_boundaries(selected: list[int], n: int, min_bars: int, preferred: list[int]) -> list[int]:
    cleaned = [selected[0]]
    for value in selected[1:]:
        if value - cleaned[-1] < min_bars:
            if value == n:
                cleaned[-1] = value
            continue
        while value - cleaned[-1] > max(preferred):
            cleaned.append(cleaned[-1] + preferred[-1])
        cleaned.append(value)
    if cleaned[-1] != n:
        cleaned.append(n)
    return sorted(set(cleaned))


def _candidate_vocal_windows(section: dict[str, Any], bar_features: list[BarFeature], bars: list[float], activity: dict[str, Any]) -> list[tuple[int, int]]:
    start_bar = int(section["barStart"])
    end_bar = int(section["barEnd"])
    if end_bar - start_bar < 2:
        return []
    densities = _bar_activity_densities(activity, bars, bar_features)
    local = densities[start_bar:end_bar]
    if not local:
        return []
    threshold = max(0.20, min(0.55, float(np.percentile(local, 55)) * 0.85))
    active = [value >= threshold or bar_features[start_bar + offset].vocalDensity >= 0.26 for offset, value in enumerate(local)]
    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(active):
        if not active[index]:
            index += 1
            continue
        run_start = index
        gap = 0
        index += 1
        while index < len(active):
            if active[index]:
                gap = 0
                index += 1
                continue
            if gap == 0 and index + 1 < len(active) and active[index + 1]:
                gap += 1
                index += 1
                continue
            break
        runs.append((start_bar + run_start, start_bar + index))
    windows: set[tuple[int, int]] = set()
    for run_start, run_end in runs:
        run_len = run_end - run_start
        if run_len <= 0:
            continue
        if run_len >= 2 and densities[run_start] < 0.75 and densities[run_start + 1] > densities[run_start] + 0.18:
            run_start += 1
            run_len = run_end - run_start
        cursor = run_start
        while cursor < run_end:
            remaining = run_end - cursor
            size = 8 if remaining >= 8 and _has_soft_boundary(densities, cursor + 8, run_end) else 4 if remaining >= 3 else 2
            if remaining > 8:
                size = 4
            if remaining == 3:
                size = 2
            phrase_end = min(run_end, cursor + size)
            if phrase_end - cursor < 2 and windows:
                break
            phrase_end = _round_phrase_end(cursor, phrase_end, end_bar)
            windows.add((cursor, phrase_end))
            cursor = phrase_end
    if not windows and float(section.get("vocalDensity", 0.0)) >= 0.22:
        cursor = start_bar
        while cursor + 2 <= end_bar:
            size = 4 if cursor + 4 <= end_bar else 2
            windows.add((cursor, cursor + size))
            cursor += size
    return sorted(windows)


def _bar_activity_densities(activity: dict[str, Any], bars: list[float], bar_features: list[BarFeature]) -> list[float]:
    times = np.asarray(activity.get("times") or [], dtype=np.float32)
    active = np.asarray(activity.get("active") or [], dtype=bool)
    if times.size and active.size:
        values = []
        for index in range(len(bars) - 1):
            mask = (times >= bars[index]) & (times < bars[index + 1])
            values.append(float(np.mean(active[mask])) if np.any(mask) else 0.0)
        return values
    return [float(item.vocalDensity) for item in bar_features]


def _has_soft_boundary(densities: list[float], boundary: int, run_end: int) -> bool:
    if boundary <= 0 or boundary >= len(densities) or boundary >= run_end:
        return False
    before = densities[max(0, boundary - 1)]
    after = densities[min(len(densities) - 1, boundary)]
    return min(before, after) < 0.38 or abs(before - after) > 0.22


def _round_phrase_end(start: int, end: int, section_end: int) -> int:
    length = end - start
    if length in {2, 4, 8}:
        return end
    for size in (2, 4, 8):
        if length <= size and start + size <= section_end:
            return start + size
    return end


def _score_vocal_phrase(activity: dict[str, Any], bar_features: list[BarFeature], bars: list[float], section: dict[str, Any], start_bar: int, end_bar: int, track_id: str, source: str, number: int) -> VocalPhrase | None:
    start = bars[start_bar]
    end = bars[end_bar]
    times = np.asarray(activity.get("times") or [], dtype=np.float32)
    energy = np.asarray(activity.get("energy") or [], dtype=np.float32)
    active = np.asarray(activity.get("active") or [], dtype=bool)
    if times.size:
        main_mask = (times >= start) & (times < end)
        vocal_energy = float(np.mean(energy[main_mask])) if np.any(main_mask) else 0.0
        lyric_density = float(np.mean(active[main_mask])) if np.any(main_mask) else 0.0
        pre_mask = (times >= max(0.0, start - 1.0)) & (times < start)
        post_mask = (times >= end) & (times < end + 1.5)
        pickup = bool(np.any(active[pre_mask])) and float(np.mean(active[pre_mask])) > 0.18
        tail = bool(np.any(active[post_mask])) and float(np.mean(active[post_mask])) > 0.16
    else:
        subset = bar_features[start_bar:end_bar]
        vocal_energy = float(np.mean([item.vocalEnergy for item in subset])) if subset else 0.0
        lyric_density = float(np.mean([item.vocalDensity for item in subset])) if subset else 0.0
        pickup = start_bar > 0 and bar_features[start_bar - 1].vocalDensity > 0.30
        tail = end_bar < len(bar_features) and bar_features[end_bar].vocalDensity > 0.30
    if lyric_density < 0.18 and vocal_energy < 0.12:
        return None
    entry_clean = not pickup and (start_bar == 0 or bar_features[start_bar - 1].vocalDensity < 0.45)
    exit_clean = not tail and (end_bar >= len(bar_features) or bar_features[end_bar].vocalDensity < 0.45)
    risk_flags = []
    if not entry_clean:
        risk_flags.append("pickup_or_busy_entry")
    if not exit_clean:
        risk_flags.append("tail_or_busy_exit")
    completeness = float(np.clip(0.32 + lyric_density * 0.28 + vocal_energy * 0.18 + entry_clean * 0.11 + exit_clean * 0.11 + ((end_bar - start_bar) in {2, 4, 8}) * 0.10, 0, 1))
    if not exit_clean and not tail:
        risk_flags.append("cuts_vocal_phrase")
    score = float(np.clip((0.30 * vocal_energy + 0.25 * lyric_density + 0.25 * completeness + 0.10 * entry_clean + 0.10 * exit_clean) * 100, 0, 100))
    return VocalPhrase(
        id=f"{source}_vp_{number:03d}",
        trackId=track_id,
        source=source,
        sectionId=str(section["id"]),
        start=round(max(0.0, start - (0.5 if pickup else 0.0)), 3),
        end=round(end + (0.5 if tail else 0.0), 3),
        barStart=start_bar,
        barEnd=end_bar,
        bars=end_bar - start_bar,
        pickupStart=round(max(0.0, start - 0.5), 3) if pickup else None,
        tailEnd=round(end + 0.5, 3) if tail else None,
        mainStart=round(start, 3),
        mainEnd=round(end, 3),
        vocalEnergy=round(vocal_energy, 4),
        lyricDensity=round(lyric_density, 4),
        phraseCompleteness=round(completeness, 4),
        entryClean=entry_clean,
        exitClean=exit_clean,
        hasPickup=pickup,
        hasTail=tail,
        score=round(score, 1),
        riskFlags=risk_flags,
    )


def _dedupe_phrases(phrases: list[VocalPhrase]) -> list[VocalPhrase]:
    phrases = sorted(phrases, key=lambda item: (item.score, item.bars in {2, 4}), reverse=True)
    selected: list[VocalPhrase] = []
    for phrase in phrases:
        overlap = False
        for existing in selected:
            bars_overlap = min(phrase.barEnd, existing.barEnd) - max(phrase.barStart, existing.barStart)
            if bars_overlap >= min(phrase.bars, existing.bars) * 0.75:
                overlap = True
                break
        if not overlap:
            selected.append(phrase)
    return sorted(selected, key=lambda item: (item.barStart, item.bars))


def _candidate_bed_windows(section: dict[str, Any], bar_features: list[BarFeature]) -> list[tuple[int, int]]:
    start = int(section["barStart"])
    end = int(section["barEnd"])
    windows: set[tuple[int, int]] = set()
    for size in (16, 8, 4):
        if end - start < size:
            continue
        step = max(4, size // 2)
        cursor = start
        while cursor + size <= end:
            subset = bar_features[cursor : cursor + size]
            vocal = float(np.mean([bar.vocalDensity for bar in subset])) if subset else 1.0
            drums = float(np.mean([bar.drumActivity for bar in subset])) if subset else 0.0
            if vocal <= 0.42 and drums >= 0.12:
                windows.add((cursor, cursor + size))
            cursor += step
    if not windows and end - start >= 4:
        windows.add((start, min(end, start + min(16, end - start))))
    return sorted(windows, key=lambda item: (item[1] - item[0], -item[0]), reverse=True)


def _make_groove_bed_candidate(
    section: dict[str, Any],
    bar_features: list[BarFeature],
    analysis: dict[str, Any],
    source: str,
    track_id: str,
    start_idx: int,
    end_idx: int,
    number: int,
) -> GrooveBedCandidate:
    subset = bar_features[start_idx:end_idx]
    vocal_leakage = float(np.mean([bar.vocalDensity for bar in subset])) if subset else 1.0
    drum = float(np.mean([bar.drumActivity for bar in subset])) if subset else 0.0
    bass_values = [bar.bassEnergy for bar in subset] or [0.0]
    bass_stability = float(np.clip(1.0 - np.std(bass_values) * 1.8, 0, 1))
    loopability = _feature_loopability(subset, {})
    energy_values = [bar.energy for bar in subset] or [0.0]
    consistency = float(np.clip(1.0 - np.std(energy_values) * 1.8, 0, 1))
    harmonic = 1.0 if analysis.get("camelot") else 0.65
    score = (0.25 * (1 - vocal_leakage) + 0.20 * drum + 0.20 * bass_stability + 0.15 * loopability + 0.10 * consistency + 0.10 * harmonic) * 100
    risks = []
    if vocal_leakage > 0.28:
        risks.append("vocal_leakage")
    if loopability < 0.45:
        risks.append("one_shot_bed")
    if end_idx - start_idx < int(section.get("bars", 0)):
        risks.append("section_subwindow")
    return GrooveBedCandidate(
        id=f"{source}_bed_{number:03d}",
        trackId=track_id,
        source=source,
        sectionId=str(section["id"]),
        start=round(bar_features[start_idx].start, 3),
        end=round(bar_features[end_idx - 1].end, 3),
        barStart=start_idx,
        barEnd=end_idx,
        bars=end_idx - start_idx,
        usesStems=["drums", "bass", "other"],
        vocalLeakage=round(vocal_leakage, 4),
        drumActivity=round(drum, 4),
        bassStability=round(bass_stability, 4),
        loopability=round(loopability, 4),
        energy=round(float(np.mean(energy_values)), 4),
        camelot=analysis.get("camelot"),
        bpm=float(analysis.get("bpm") or 120),
        score=round(float(np.clip(score, 0, 100)), 1),
        riskFlags=risks,
    )


def _section_loopability(section: dict[str, Any], bar_features: list[BarFeature], stems: dict[str, Any] | None) -> float:
    subset = bar_features[int(section["barStart"]) : int(section["barEnd"])]
    return _feature_loopability(subset, {})


def _feature_loopability(subset: list[BarFeature], matrices: dict[str, np.ndarray]) -> float:
    if len(subset) < 2:
        return 0.0
    first = subset[0]
    last = subset[-1]
    rms_diff = abs(first.rms - last.rms) / max(first.rms, last.rms, 1e-6)
    low_diff = abs(first.bassEnergy - last.bassEnergy)
    centroid_diff = abs(first.spectralCentroid - last.spectralCentroid) / 8000.0
    chroma_sim = _cosine(np.asarray(first.chroma), np.asarray(last.chroma))
    internal_repeat = _internal_repeat(subset)
    return float(np.clip(1.0 - rms_diff * 0.25 - low_diff * 0.25 - centroid_diff * 0.20 + chroma_sim * 0.18 + internal_repeat * 0.12, 0, 1))


def _internal_repeat(subset: list[BarFeature]) -> float:
    if len(subset) < 4:
        return 0.35
    half = len(subset) // 2
    left = np.asarray([[bar.energy, bar.bassEnergy, bar.drumActivity] for bar in subset[:half]], dtype=np.float32)
    right = np.asarray([[bar.energy, bar.bassEnergy, bar.drumActivity] for bar in subset[-half:]], dtype=np.float32)
    return float(np.mean([_cosine(a, b) for a, b in zip(left, right)]))


def _section_has_vocal_bars(bar_features: list[BarFeature], section: dict[str, Any]) -> bool:
    subset = bar_features[int(section["barStart"]) : int(section["barEnd"])]
    return any(item.vocalDensity > 0.32 for item in subset)


def _safe_point_reasons(bar: BarFeature, score: float, transition: float, phrase: float) -> list[str]:
    reasons = []
    if bar.vocalDensity < 0.35:
        reasons.append("low vocal activity")
    if bar.isDownbeatStrong:
        reasons.append("strong downbeat")
    if transition > 0.5:
        reasons.append("near transition candidate")
    if phrase > 0.5:
        reasons.append("section/phrase boundary")
    if abs(bar.energyDelta) > 0.15:
        reasons.append("intentional energy change")
    return reasons or ["bar-grid fallback"]


def _mark_vocal_events(features: list[BarFeature]) -> None:
    for index, item in enumerate(features):
        prev_v = features[index - 1].vocalDensity if index > 0 else 0.0
        next_v = features[index + 1].vocalDensity if index + 1 < len(features) else 0.0
        item.isLikelyVocalEntry = item.vocalDensity - prev_v > 0.25 and item.vocalDensity > 0.32
        item.isLikelyVocalExit = item.vocalDensity - next_v > 0.25 and item.vocalDensity > 0.32
        item.isLikelyPickup = prev_v > 0.22 and item.isLikelyVocalEntry


def _explicit_boundary_candidates(bar_features: list[BarFeature], phrases: list[float], transition_candidates: dict[str, Any]) -> list[BoundaryCandidate]:
    result = []
    starts = [item.start for item in bar_features]
    for phrase in phrases:
        index = _nearest_index(starts, phrase)
        if 0 < index < len(bar_features):
            result.append(BoundaryCandidate(time=round(starts[index], 3), barIndex=index, score=0.46, sourceScores={"novelty": 0, "selfSimilarity": 0, "energyChange": 0, "vocalEntryExit": 0, "drumBassChange": 0, "transitionCandidate": 0, "phraseBoundary": 1}, boundaryType="minor", snapReason="phrase"))
    for key in ("intro", "outro"):
        value = transition_candidates.get(key)
        if _is_number(value):
            index = _nearest_index(starts, float(value))
            if 0 < index < len(bar_features):
                result.append(BoundaryCandidate(time=round(starts[index], 3), barIndex=index, score=0.62, sourceScores={"novelty": 0, "selfSimilarity": 0, "energyChange": 0, "vocalEntryExit": 0, "drumBassChange": 0, "transitionCandidate": 1, "phraseBoundary": 0}, boundaryType="transition_safe", snapReason="transition_candidate"))
    return result


def _nms_boundaries(candidates: list[BoundaryCandidate], min_distance_bars: int) -> list[BoundaryCandidate]:
    selected: list[BoundaryCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if any(abs(candidate.barIndex - item.barIndex) < min_distance_bars for item in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: item.barIndex)


def _transition_candidate_score(time: float, candidates: dict[str, Any]) -> float:
    score = 0.0
    confidence = float(candidates.get("confidence") or 0.0)
    for key in ("intro", "outro", "drop"):
        value = candidates.get(key)
        if _is_number(value):
            score = max(score, (1.0 - min(1.0, abs(float(value) - time) / 4.0)) * max(0.5, confidence))
    return float(np.clip(score, 0, 1))


def _normalized_bars(analysis: dict[str, Any], duration: float) -> list[float]:
    raw = sorted(float(value) for value in (analysis.get("bars") or []) if _is_number(value) and 0 <= float(value) <= duration)
    if len(raw) < 2:
        beats = sorted(float(value) for value in (analysis.get("beats") or []) if _is_number(value) and 0 <= float(value) <= duration)
        raw = beats[::4] if len(beats) >= 8 else []
    if len(raw) < 2:
        bpm = float(analysis.get("bpm") or 120.0)
        bar_seconds = 240.0 / max(bpm, 1.0)
        raw = list(np.arange(0.0, duration, bar_seconds))
    if not raw or raw[0] > 0.05:
        raw.insert(0, 0.0)
    if raw[-1] < duration - 0.05:
        raw.append(duration)
    cleaned = [raw[0]]
    for value in raw[1:]:
        if value - cleaned[-1] >= 0.2:
            cleaned.append(min(value, duration))
    return cleaned


def _normalized_phrases(analysis: dict[str, Any], bars: list[float]) -> list[float]:
    duration = bars[-1] if bars else 0.0
    phrases = sorted(float(value) for value in (analysis.get("phrases") or []) if _is_number(value) and 0 <= float(value) <= duration)
    if not phrases and bars:
        phrases = [bars[index] for index in range(0, len(bars), 8)]
    return phrases


def _safe_load_audio(path: Any, sr: int) -> np.ndarray:
    if not path:
        return np.zeros(0, dtype=np.float32)
    try:
        y, _ = librosa.load(Path(path), sr=sr, mono=True)
        return np.ascontiguousarray(y, dtype=np.float32)
    except Exception:
        return np.zeros(0, dtype=np.float32)


def _load_stems(stems: dict[str, Any] | None, sr: int) -> dict[str, np.ndarray]:
    result = {}
    for name, value in (stems or {}).items():
        if name not in STEM_NAMES:
            continue
        audio = _load_audio_like(value, sr)
        if audio.size:
            result[name] = _mono(audio)
    return result if all(name in result for name in STEM_NAMES) else result


def _load_audio_like(value: Any, sr: int) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=np.float32)
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value, dtype=np.float32)
    try:
        y, _ = librosa.load(Path(value), sr=sr, mono=False)
        return np.ascontiguousarray(y, dtype=np.float32)
    except Exception:
        return np.zeros(0, dtype=np.float32)


def _mono(audio: np.ndarray | None) -> np.ndarray:
    if audio is None:
        return np.zeros(0, dtype=np.float32)
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim == 2 and y.shape[0] <= 8:
        y = np.mean(y, axis=0)
    elif y.ndim == 2:
        y = np.mean(y, axis=1)
    return np.ascontiguousarray(y, dtype=np.float32)


def _fit_audio(audio: np.ndarray, samples: int) -> np.ndarray:
    if samples <= 0:
        return audio
    if audio.size >= samples:
        return audio[:samples]
    return np.pad(audio, (0, samples - audio.size))


def _slice(audio: np.ndarray, start: float, end: float, sr: int) -> np.ndarray:
    start_i = max(0, min(audio.size, int(start * sr)))
    end_i = max(start_i, min(audio.size, int(end * sr)))
    return np.ascontiguousarray(audio[start_i:end_i], dtype=np.float32)


def _stem_energy(audio: np.ndarray | None, start: float, end: float, sr: int) -> float:
    if audio is None:
        return 0.0
    return _rms(_slice(audio, start, end, sr))


def _low_band_energy(audio: np.ndarray | None, start: float, end: float, sr: int) -> float:
    if audio is None:
        return 0.0
    clip = _slice(audio, start, end, sr)
    if clip.size < 128:
        return 0.0
    try:
        sos = signal.butter(3, 250, "lowpass", fs=sr, output="sos")
        low = signal.sosfilt(sos, clip)
        return _rms(low)
    except Exception:
        return _rms(clip)


def _mean_feature(fn: Any, clip: np.ndarray, size: int) -> list[float]:
    if clip.size < 32:
        return [0.0] * size
    try:
        values = np.asarray(fn(clip), dtype=np.float32)
        if values.ndim == 1:
            result = values
        else:
            result = np.mean(values, axis=1)
        return _pad_list(result, size)
    except Exception:
        return [0.0] * size


def _scalar_feature(fn: Any, clip: np.ndarray) -> float:
    if clip.size < 32:
        return 0.0
    try:
        values = np.asarray(fn(clip), dtype=np.float32)
        return float(np.mean(values))
    except Exception:
        return 0.0


def _spectral_flux(clip: np.ndarray, sr: int) -> float:
    if clip.size < 128:
        return 0.0
    try:
        stft = np.abs(librosa.stft(clip, n_fft=min(2048, _n_fft(clip)), hop_length=512))
        if stft.shape[1] < 2:
            return 0.0
        return float(np.mean(np.maximum(0, np.diff(stft, axis=1))))
    except Exception:
        return 0.0


def _n_fft(y: np.ndarray) -> int:
    return max(64, min(4096, 2 ** int(np.floor(np.log2(max(64, y.size))))))


def _vocal_density(vocal_energy: float, full_rms: float, curve_hint: float, has_stem: bool) -> float:
    if has_stem:
        relative = vocal_energy / max(full_rms, 1e-5)
        return float(np.clip(vocal_energy * 5.0 + relative * 0.18 + curve_hint * 0.18, 0, 1))
    return float(np.clip(curve_hint, 0, 1))


def _cosine_matrix(features: np.ndarray) -> np.ndarray:
    x = _normalize_features(features)
    return np.clip(x @ x.T, 0, 1).astype(np.float32)


def _normalize_features(features: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim != 2:
        x = x.reshape(len(x), -1)
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True) + 1e-6
    x = (x - mean) / std
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-6
    return x / norm


def _smooth_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    try:
        return signal.medfilt2d(matrix, kernel_size=3).astype(np.float32)
    except Exception:
        return matrix.astype(np.float32)


def _smooth_1d(values: np.ndarray, kernel: int) -> np.ndarray:
    if values.size < kernel or kernel <= 1:
        return values.astype(np.float32)
    kernel = kernel if kernel % 2 else kernel + 1
    return signal.medfilt(values.astype(np.float32), kernel_size=kernel).astype(np.float32)


def _normalize_1d(values: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low = float(np.min(x)) if x.size else 0.0
    high = float(np.max(x)) if x.size else 0.0
    if high - low < 1e-8:
        return np.zeros_like(x)
    return (x - low) / (high - low)


def _curve_avg(curve: list[dict[str, Any]], key: str, start: float, end: float, fallback: float) -> float:
    values = [float(item[key]) for item in curve if isinstance(item, dict) and _is_number(item.get(key)) and start <= float(item.get("time", -1)) < end]
    return float(np.mean(values)) if values else float(fallback)


def _distance(a: Any, b: Any) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(aa - bb) / max(np.sqrt(aa.size), 1.0))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12)) if audio.size else 0.0


def _pad_list(values: np.ndarray, size: int) -> list[float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size < size:
        arr = np.pad(arr, (0, size - arr.size))
    return [float(v) for v in arr[:size]]


def _energy_shape(values: np.ndarray) -> str:
    if values.size < 2:
        return "flat"
    delta = float(values[-1] - values[0])
    middle = float(np.mean(values[len(values) // 3 : max(len(values) // 3 + 1, len(values) * 2 // 3)]))
    if delta > 0.16:
        return "rising"
    if delta < -0.16:
        return "falling"
    if middle > max(values[0], values[-1]) + 0.10:
        return "peak"
    if middle < min(values[0], values[-1]) - 0.10:
        return "valley"
    return "flat"


def _section_repetition(start: int, end: int, matrix: np.ndarray | None) -> float:
    if matrix is None or matrix.size == 0 or end <= start:
        return 0.0
    inside = matrix[start:end, start:end]
    outside_left = matrix[start:end, :start]
    outside_right = matrix[start:end, end:]
    outside = np.concatenate([outside_left.reshape(-1), outside_right.reshape(-1)]) if outside_left.size or outside_right.size else np.asarray([0.0])
    return float(np.clip(np.percentile(outside, 90) - np.mean(inside) * 0.15 + 0.5, 0, 1))


def _mix_score(bar: BarFeature, entry: bool) -> float:
    low_vocal = 1.0 - bar.vocalDensity
    moderate = 1.0 - abs(bar.energy - 0.45)
    downbeat = 1.0 if bar.isDownbeatStrong else 0.55
    return float(np.clip(low_vocal * 0.44 + moderate * 0.28 + downbeat * 0.28, 0, 1))


def _nearest_index(values: list[float], target: float) -> int:
    if not values:
        return 0
    return int(min(range(len(values)), key=lambda index: abs(values[index] - target)))


def _novelty_peaks(values: list[float]) -> list[dict[str, Any]]:
    peaks = []
    arr = np.asarray(values, dtype=np.float32)
    if arr.size < 3:
        return peaks
    threshold = max(0.28, float(np.percentile(arr, 72)))
    for index in range(1, arr.size - 1):
        if arr[index] >= threshold and arr[index] >= arr[index - 1] and arr[index] >= arr[index + 1]:
            peaks.append({"barIndex": index, "score": round(float(arr[index]), 4)})
    return peaks


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
