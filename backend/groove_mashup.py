from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from .arrangement import build_music_segments, normalized_bars
from .loudness import loudness_metrics, normalize_loudness
from .matching import camelot_key_distance
from .mixing import SAMPLE_RATE
from .storage import STEM_DIR
from .segmentation import analyze_track_segmentation


STEM_NAMES = ("vocals", "drums", "bass", "other")
GROOVE_MODES = {
    "groove_vocal_handoff",
    "a_vocal_on_b_groove",
    "b_vocal_on_a_groove",
    "call_response_groove",
    "hook_exchange_groove",
}


def build_groove_mashup_plan(
    track_a: dict[str, Any],
    track_b: dict[str, Any],
    *,
    mode: str = "groove_vocal_handoff",
    target_duration_sec: float = 120.0,
    use_stems: bool = True,
    vocal_priority: str = "auto",
    bed_preference: str = "auto",
    allow_hybrid_bed: bool = True,
    allow_vocal_pitch_shift: bool = False,
    max_vocal_stretch: float = 1.06,
    return_alternatives: bool = True,
) -> dict[str, Any]:
    warnings: list[str] = []
    stems_a = _cached_stem_paths(str(track_a.get("id")))
    stems_b = _cached_stem_paths(str(track_b.get("id")))
    if use_stems and (not stems_a or not stems_b):
        message = "stems_required: Groove vocal handoff needs Demucs vocals/drums/bass/other for both tracks."
        return {
            "plan": [],
            "items": [],
            "score": 0,
            "mode": mode,
            "warnings": [message],
            "groovePlan": {
                "status": "stems_required",
                "globalWarnings": [message],
                "bed": None,
                "vocalEvents": [],
                "qualityReport": {"warnings": [message], "summary": message},
            },
            "downloadPreviewUrl": None,
        }
    if not use_stems:
        warnings.append("fallback_warning: Groove vocal handoff is not clean without stems; no fake full-mix groove plan was generated.")
        return {
            "plan": [],
            "items": [],
            "score": 0,
            "mode": mode,
            "warnings": warnings,
            "groovePlan": {
                "status": "stems_required",
                "globalWarnings": warnings,
                "bed": None,
                "vocalEvents": [],
                "qualityReport": {"warnings": warnings, "summary": warnings[0]},
            },
            "downloadPreviewUrl": None,
        }

    segmentation_a = analyze_track_segmentation({**track_a, "source": "A"}, "A", audio=_safe_load_mono(track_a), sr=SAMPLE_RATE, stems=stems_a)
    segmentation_b = analyze_track_segmentation({**track_b, "source": "B"}, "B", audio=_safe_load_mono(track_b), sr=SAMPLE_RATE, stems=stems_b)
    segments_a = build_music_segments(track_a, "A", 8, audio=_safe_load_mono(track_a), sr=SAMPLE_RATE)
    segments_b = build_music_segments(track_b, "B", 8, audio=_safe_load_mono(track_b), sr=SAMPLE_RATE)
    beds = [
        *[_bed_from_segmentation_candidate(candidate, track_a) for candidate in segmentation_a.get("grooveBedCandidates", [])],
        *[_bed_from_segmentation_candidate(candidate, track_b) for candidate in segmentation_b.get("grooveBedCandidates", [])],
    ]
    if allow_hybrid_bed:
        beds.extend(find_candidate_groove_beds(track_a, track_b, stems_a, stems_b, {"segments": segments_a}, {"segments": segments_b}, allow_hybrid_bed=True)[:6])
    beds = sorted([bed for bed in beds if bed], key=lambda item: item["score"], reverse=True)
    phrases_a = [_phrase_from_segmentation_candidate(item, track_a) for item in segmentation_a.get("vocalPhrases", [])]
    phrases_b = [_phrase_from_segmentation_candidate(item, track_b) for item in segmentation_b.get("vocalPhrases", [])]
    if not phrases_a:
        phrases_a = extract_vocal_phrases({**track_a, "source": "A"}, stems_a.get("vocals") if stems_a else None, {"segments": segments_a})
    if not phrases_b:
        phrases_b = extract_vocal_phrases({**track_b, "source": "B"}, stems_b.get("vocals") if stems_b else None, {"segments": segments_b})
    if mode == "a_vocal_on_b_groove":
        bed_preference = "B"
    elif mode == "b_vocal_on_a_groove":
        bed_preference = "A"
    if bed_preference == "A":
        beds = sorted(beds, key=lambda bed: 0 if bed.get("source") == "A" else 1)
    elif bed_preference == "B":
        beds = sorted(beds, key=lambda bed: 0 if bed.get("source") == "B" else 1)
    candidates = []
    for bed in beds[:6]:
        arrangement = build_vocal_handoff_arrangement(
            bed,
            phrases_a,
            phrases_b,
            mode,
            target_duration_sec,
            vocal_priority=vocal_priority,
            allow_vocal_pitch_shift=allow_vocal_pitch_shift,
            max_vocal_stretch=max_vocal_stretch,
        )
        if arrangement["vocalEvents"]:
            candidates.append(arrangement)
    if not candidates:
        message = "No usable groove bed plus vocal phrase arrangement found."
        return {"plan": [], "items": [], "score": 0, "mode": mode, "warnings": [message], "groovePlan": {"status": "no_plan", "globalWarnings": [message], "qualityReport": {"summary": message, "warnings": [message]}}, "downloadPreviewUrl": None}
    candidates.sort(key=lambda item: item["qualityReport"]["score"], reverse=True)
    selected = candidates[0]
    selected["globalWarnings"] = _dedupe([*warnings, *selected.get("globalWarnings", [])])
    ui_items = _groove_ui_items(selected)
    return {
        "plan": ui_items,
        "items": ui_items,
        "score": selected["qualityReport"]["score"],
        "mode": mode,
        "warnings": selected["globalWarnings"],
        "qualityReport": selected["qualityReport"],
        "targetBpm": selected["targetBpm"],
        "targetCamelot": selected["targetCamelot"],
        "groovePlan": selected,
        "segmentationReport": {
            "trackA": _compact_segmentation_report(segmentation_a),
            "trackB": _compact_segmentation_report(segmentation_b),
        },
        "alternativePlans": [
            {
                "plan": _groove_ui_items(item),
                "items": _groove_ui_items(item),
                "groovePlan": item,
                "score": item["qualityReport"]["score"],
                "mode": mode,
                "warnings": item.get("globalWarnings", []),
                "qualityReport": item["qualityReport"],
                "targetBpm": item["targetBpm"],
                "targetCamelot": item["targetCamelot"],
            }
            for item in candidates[1:4]
        ] if return_alternatives else [],
        "downloadPreviewUrl": None,
    }


def _bed_from_segmentation_candidate(candidate: dict[str, Any], track: dict[str, Any]) -> dict[str, Any]:
    source = candidate.get("source") or track.get("source") or "A"
    return {
        "id": candidate.get("id"),
        "source": source,
        "drumsSource": source,
        "bassSource": source,
        "otherSource": source,
        "drumsTrackId": candidate.get("trackId") or track.get("id"),
        "bassTrackId": candidate.get("trackId") or track.get("id"),
        "otherTrackId": candidate.get("trackId") or track.get("id"),
        "sourceStart": float(candidate.get("start", 0.0)),
        "sourceEnd": float(candidate.get("end", candidate.get("start", 0.0))),
        "drumsStart": float(candidate.get("start", 0.0)),
        "drumsEnd": float(candidate.get("end", candidate.get("start", 0.0))),
        "bassStart": float(candidate.get("start", 0.0)),
        "bassEnd": float(candidate.get("end", candidate.get("start", 0.0))),
        "otherStart": float(candidate.get("start", 0.0)),
        "otherEnd": float(candidate.get("end", candidate.get("start", 0.0))),
        "bars": int(candidate.get("bars", 0)),
        "bpm": float(candidate.get("bpm") or track.get("bpm") or 120),
        "camelot": candidate.get("camelot") or track.get("camelot"),
        "energy": float(candidate.get("energy", 0.5)),
        "bassEnergy": float(candidate.get("bassStability", 0.5)),
        "drumActivity": float(candidate.get("drumActivity", 0.5)),
        "vocalLeakage": float(candidate.get("vocalLeakage", 0.5)),
        "loopability": float(candidate.get("loopability", 0.0)),
        "score": float(candidate.get("score", 0.0)),
        "warnings": list(candidate.get("riskFlags") or []),
        "reason": "Selected by multi-scale segmentation as a low-vocal, loopable drums/bass/other bed.",
    }


def _phrase_from_segmentation_candidate(candidate: dict[str, Any], track: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": candidate.get("id"),
        "source": candidate.get("source") or track.get("source") or "A",
        "trackId": candidate.get("trackId") or track.get("id"),
        "sourceStart": float(candidate.get("start", 0.0)),
        "sourceEnd": float(candidate.get("end", candidate.get("start", 0.0))),
        "barStart": int(candidate.get("barStart", 0)),
        "barEnd": int(candidate.get("barEnd", 0)),
        "bars": int(candidate.get("bars", 0)),
        "bpm": float(track.get("bpm") or 120),
        "camelot": track.get("camelot"),
        "vocalEnergy": float(candidate.get("vocalEnergy", 0.0)),
        "lyricDensity": float(candidate.get("lyricDensity", 0.0)),
        "hasPickup": bool(candidate.get("hasPickup", False)),
        "hasTail": bool(candidate.get("hasTail", False)),
        "tailDuration": max(0.0, float(candidate.get("tailEnd") or candidate.get("end", 0.0)) - float(candidate.get("mainEnd", candidate.get("end", 0.0)))) if candidate.get("hasTail") else 0.0,
        "phraseCompleteness": float(candidate.get("phraseCompleteness", 0.0)),
        "downbeatOffset": max(0.0, float(candidate.get("mainStart", candidate.get("start", 0.0))) - float(candidate.get("start", 0.0))),
        "entryClean": bool(candidate.get("entryClean", True)),
        "exitClean": bool(candidate.get("exitClean", True)),
        "score": float(candidate.get("score", 0.0)),
        "warnings": list(candidate.get("riskFlags") or []),
    }


def _compact_segmentation_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": report.get("method"),
        "barsDetected": report.get("barsDetected"),
        "stemsUsed": report.get("stemsUsed"),
        "boundaryCount": report.get("boundaryCount"),
        "sections": report.get("sections", []),
        "minorSections": report.get("minorSections", []),
        "vocalPhrases": report.get("vocalPhrases", []),
        "grooveBedCandidates": report.get("grooveBedCandidates", []),
        "safeCutPoints": report.get("safeCutPoints", []),
        "warnings": report.get("warnings", []),
        "debug": report.get("debug", {}),
    }


def find_candidate_groove_beds(
    track_a: dict[str, Any],
    track_b: dict[str, Any],
    stems_a: dict[str, Path] | None = None,
    stems_b: dict[str, Path] | None = None,
    analysis_a: dict[str, Any] | None = None,
    analysis_b: dict[str, Any] | None = None,
    *,
    allow_hybrid_bed: bool = True,
) -> list[dict[str, Any]]:
    segments_a = analysis_a.get("segments") if analysis_a else None
    segments_b = analysis_b.get("segments") if analysis_b else None
    if not segments_a:
        segments_a = build_music_segments(track_a, "A", 8)
    if not segments_b:
        segments_b = build_music_segments(track_b, "B", 8)
    beds: list[dict[str, Any]] = []
    for source, track, stems, segments in (("A", track_a, stems_a, segments_a), ("B", track_b, stems_b, segments_b)):
        for segment in _best_bed_segments(segments)[:4]:
            beds.append(_make_bed(source, source, source, source, track, track, track, segment, segment, segment, stems))
    if allow_hybrid_bed:
        for seg_a in _best_bed_segments(segments_a)[:2]:
            for seg_b in _best_bed_segments(segments_b)[:2]:
                beds.extend(
                    [
                        _make_bed("hybrid", "A", "B", "B", track_a, track_b, track_b, seg_a, seg_b, seg_b, stems_a),
                        _make_bed("hybrid", "B", "A", "A", track_b, track_a, track_a, seg_b, seg_a, seg_a, stems_b),
                        _make_bed("hybrid", "A", "A", "B", track_a, track_a, track_b, seg_a, seg_a, seg_b, stems_a),
                        _make_bed("hybrid", "B", "B", "A", track_b, track_b, track_a, seg_b, seg_b, seg_a, stems_b),
                    ]
                )
    return sorted(beds, key=lambda bed: bed["score"], reverse=True)


def extract_vocal_phrases(track: dict[str, Any], vocals_stem: str | Path | None = None, analysis: dict[str, Any] | None = None, phrase_bars: list[int] | None = None) -> list[dict[str, Any]]:
    phrase_bars = phrase_bars or [2, 4, 8]
    duration = float(track.get("duration") or 0.0)
    bars = normalized_bars(track, duration, float(track.get("bpm") or 120))
    if len(bars) < 3:
        return []
    source = track.get("source") or "A"
    vocal_audio = _load_stereo(Path(vocals_stem)) if vocals_stem and Path(vocals_stem).exists() else None
    phrases: list[dict[str, Any]] = []
    for size in sorted(phrase_bars):
        step = max(1, size)
        for start_index in range(0, max(1, len(bars) - size - 1), step):
            end_index = start_index + size
            if end_index >= len(bars):
                continue
            start = bars[start_index]
            end = bars[end_index]
            vocal_energy = _vocal_energy(track, vocal_audio, start, end)
            lyric_density = _vocal_density(track, start, end, fallback=vocal_energy)
            if vocal_energy < 0.08 and lyric_density < 0.20:
                continue
            pickup = _vocal_energy(track, vocal_audio, max(0.0, start - 1.0), start) > vocal_energy * 0.35
            tail_duration = _estimate_tail(track, vocal_audio, end, min(duration, end + 1.5), vocal_energy)
            source_start = max(0.0, start - (0.5 if pickup else 0.0))
            entry_clean = _vocal_energy(track, vocal_audio, max(0.0, start - 0.35), start + 0.1) < max(0.2, vocal_energy * 0.75)
            exit_clean = tail_duration <= 1.2
            phrase_completeness = float(np.clip(0.55 + (0.18 if entry_clean else -0.08) + (0.18 if exit_clean else -0.05) + min(0.2, vocal_energy * 0.2), 0, 1))
            score = (
                0.30 * vocal_energy
                + 0.20 * phrase_completeness
                + 0.15 * float(entry_clean)
                + 0.15 * float(exit_clean)
                + 0.10 * (1.0 - abs(lyric_density - 0.62))
                + 0.10 * (1.0 if track.get("camelot") else 0.55)
            ) * 100
            phrases.append(
                {
                    "id": f"{source}_vocal_phrase_{len(phrases) + 1:03d}",
                    "source": source,
                    "trackId": track.get("id"),
                    "sourceStart": round(source_start, 3),
                    "sourceEnd": round(min(duration, end + tail_duration), 3),
                    "barStart": start_index,
                    "barEnd": end_index,
                    "bars": size,
                    "bpm": float(track.get("bpm") or 120),
                    "camelot": track.get("camelot"),
                    "vocalEnergy": round(float(np.clip(vocal_energy, 0, 1)), 4),
                    "lyricDensity": round(float(np.clip(lyric_density, 0, 1)), 4),
                    "hasPickup": bool(pickup),
                    "hasTail": bool(tail_duration > 0.2),
                    "tailDuration": round(tail_duration, 3),
                    "phraseCompleteness": round(phrase_completeness, 4),
                    "downbeatOffset": round(start - source_start, 3),
                    "entryClean": bool(entry_clean),
                    "exitClean": bool(exit_clean),
                    "score": round(float(np.clip(score, 0, 100)), 1),
                    "warnings": [] if entry_clean and exit_clean else ["phrase entry/exit needs fade or tail treatment"],
                }
            )
    phrases = sorted(phrases, key=lambda item: (item["score"], item["bars"] in {2, 4}), reverse=True)[:16]
    return sorted(phrases, key=lambda item: (item["barStart"], item["bars"]))


def build_vocal_handoff_arrangement(
    bed: dict[str, Any],
    phrases_a: list[dict[str, Any]],
    phrases_b: list[dict[str, Any]],
    mode: str,
    target_duration: float,
    *,
    vocal_priority: str = "auto",
    allow_vocal_pitch_shift: bool = False,
    max_vocal_stretch: float = 1.06,
) -> dict[str, Any]:
    target_bpm = choose_groove_target_bpm(bed, phrases_a, phrases_b, mode)
    target_camelot = bed.get("camelot")
    selected = _select_phrase_sequence(bed, phrases_a, phrases_b, mode, vocal_priority)
    events = []
    cursor = 0.0
    bar_sec = 240.0 / max(target_bpm, 1.0)
    warnings: list[str] = []
    for phrase in selected:
        if cursor >= target_duration:
            break
        stretch = target_bpm / max(float(phrase.get("bpm") or target_bpm), 1.0)
        if abs(stretch - 1.0) > max_vocal_stretch - 1.0:
            warnings.append(f"{phrase['id']} vocal stretchRatio {stretch:.3f} exceeds safe limit; phrase skipped.")
            continue
        pitch = choose_groove_pitch_policy(bed, phrase, allow_vocal_pitch_shift=allow_vocal_pitch_shift)
        warnings.extend(pitch["warnings"])
        duration = min(float(phrase["bars"]) * bar_sec, target_duration - cursor)
        event_start = max(0.0, cursor - float(phrase.get("downbeatOffset") or 0.0))
        tail = "echo" if phrase.get("hasTail") else "natural"
        events.append(
            {
                "id": f"vocal_event_{len(events) + 1:03d}",
                "phraseId": phrase["id"],
                "phrase": phrase,
                "source": phrase["source"],
                "trackId": phrase["trackId"],
                "sourceStart": phrase["sourceStart"],
                "sourceEnd": phrase["sourceEnd"],
                "timelineStart": round(event_start, 3),
                "timelineEnd": round(cursor + duration, 3),
                "gainDb": 0.0,
                "pitchShiftSemitones": 0 if not allow_vocal_pitch_shift else pitch.get("vocalPitchShiftSemitones", 0),
                "bedPitchShiftSemitones": pitch.get("bedPitchShiftSemitones", 0),
                "stretchRatio": round(stretch, 4),
                "duckBedDb": -2.5,
                "tailTreatment": tail,
                "handoffToNext": "call_response" if len(events) % 2 == 0 else "overlap_tail",
                "warnings": list(phrase.get("warnings") or []),
            }
        )
        cursor += max(duration, bar_sec * 2)
    score = _arrangement_score(bed, events, warnings)
    return {
        "status": "ok",
        "targetBpm": round(target_bpm, 3),
        "targetCamelot": target_camelot,
        "bed": bed,
        "vocalEvents": events,
        "bedAutomation": [{"type": "sidechain_duck", "amountDb": -2.5}],
        "sectionPlan": _section_plan(events, bar_sec),
        "globalWarnings": _dedupe(warnings),
        "qualityReport": {
            "score": score,
            "summary": f"Selected {bed['id']} as a continuous groove bed and placed {len(events)} vocal phrases as A/B handoffs.",
            "bedReason": bed.get("reason"),
            "selectedVocalPhrases": [event["phraseId"] for event in events],
            "warnings": _dedupe(warnings),
            "strengths": [
                f"Groove bed uses drums={bed['drumsSource']}, bass={bed['bassSource']}, other={bed['otherSource']} and excludes vocals.",
                "Main vocals are arranged as phrase events over one bed, not full-mix crossfades.",
            ],
        },
    }


def choose_groove_target_bpm(bed: dict[str, Any], phrases_a: list[dict[str, Any]], phrases_b: list[dict[str, Any]], mode: str) -> float:
    # The groove bed is the timing authority. If we move target BPM away from it
    # without actually stretching the whole bed, vocal phrases drift against the
    # drums/bass and the result feels like loose crossfades again.
    bed_bpm = float(bed.get("bpm") or 120)
    return bed_bpm


def choose_groove_pitch_policy(bed: dict[str, Any], vocal_phrase: dict[str, Any], *, allow_vocal_pitch_shift: bool = False) -> dict[str, Any]:
    relation = camelot_key_distance(bed.get("camelot"), vocal_phrase.get("camelot")).get("relation")
    if relation in {"same", "adjacent", "relative_major_minor"}:
        return {"bedPitchShiftSemitones": 0, "vocalPitchShiftSemitones": 0, "warnings": [], "relation": relation}
    warnings = ["Bed/vocal Camelot is not clean; prefer shifting/reselecting bed rather than pitching vocal."]
    bed_shift = 2 if relation in {"energy_boost", "diagonal_mix", "mood_shifter", "jaws_mix"} else 3
    if abs(bed_shift) > 2:
        warnings.append("Bed pitch shift would exceed +/-2 semitones; high risk, try another bed.")
    if allow_vocal_pitch_shift:
        warnings.append("Vocal pitch shift is allowed but capped at +/-1 semitone.")
        return {"bedPitchShiftSemitones": 0, "vocalPitchShiftSemitones": 1, "warnings": warnings, "relation": relation}
    return {"bedPitchShiftSemitones": bed_shift, "vocalPitchShiftSemitones": 0, "warnings": warnings, "relation": relation}


def render_groove_vocal_mashup(
    arrangement: dict[str, Any],
    assets: dict[str, dict[str, Any]],
    sr: int = SAMPLE_RATE,
    target_lufs: float = -14.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not arrangement or arrangement.get("status") != "ok":
        raise ValueError("Cannot render groove mashup without a valid groovePlan.")
    bed = arrangement["bed"]
    duration = max([float(event.get("timelineEnd", 0)) + 2.0 for event in arrangement.get("vocalEvents", [])] + [float(bed.get("duration", 16.0))])
    bed_audio, bed_warnings = _render_bed_loop(bed, assets, duration, sr)
    vocal_bus = np.zeros_like(bed_audio)
    warnings = list(arrangement.get("globalWarnings") or []) + bed_warnings
    for event in arrangement.get("vocalEvents", []):
        vocal, event_warnings = _render_vocal_event(event, assets, sr)
        warnings.extend(event_warnings)
        start = max(0, int(float(event.get("timelineStart", 0)) * sr))
        end = min(vocal_bus.shape[1], start + vocal.shape[1])
        if end > start:
            vocal_bus[:, start:end] += vocal[:, : end - start]
            if event.get("tailTreatment") == "echo":
                tail = _delay_tail(vocal[:, : end - start], sr, arrangement.get("targetBpm", 120), wet=0.18)
                tail_start = end
                tail_end = min(vocal_bus.shape[1], tail_start + tail.shape[1])
                if tail_end > tail_start:
                    vocal_bus[:, tail_start:tail_end] += tail[:, : tail_end - tail_start] * _db_to_gain(-10)
    ducked_bed = _duck_bed_with_vocals(bed_audio, vocal_bus, sr, amount_db=-2.5)
    mix = _bus_glue(ducked_bed + vocal_bus)
    mix = normalize_loudness(mix, sr, float(target_lufs))
    mix = _peak_limit(mix, ceiling=0.891)
    metrics = loudness_metrics(mix, sr)
    report = {
        "targetBpm": arrangement.get("targetBpm"),
        "targetCamelot": arrangement.get("targetCamelot"),
        "bed": arrangement.get("bed"),
        "vocalEvents": arrangement.get("vocalEvents", []),
        "globalWarnings": _dedupe(warnings),
        "renderStats": {
            "duration": round(mix.shape[1] / sr, 3),
            "integratedLufs": metrics["lufs"],
            "peakDbfs": metrics["peak_db"],
            "clipped": bool(float(np.max(np.abs(mix))) >= 0.999),
        },
    }
    return mix.astype(np.float32), report


def _make_bed(source: str, drums_source: str, bass_source: str, other_source: str, drums_track: dict, bass_track: dict, other_track: dict, drums_seg: dict, bass_seg: dict, other_seg: dict, stems: dict[str, Path] | None) -> dict[str, Any]:
    start = float(drums_seg.get("start", 0.0))
    end = float(drums_seg.get("end", start + 16.0))
    loopability = _loopability(drums_track, stems, start, end, drums_seg)
    vocal_leakage = min(float(drums_seg.get("vocalDensity", 0.5)), float(bass_seg.get("vocalDensity", 0.5)), float(other_seg.get("vocalDensity", 0.5))) * 0.15
    drum_activity = float(drums_seg.get("drumActivity", 0.5))
    bass_energy = float(bass_seg.get("bassEnergy", 0.5))
    bass_stability = 1.0 - abs(float(bass_seg.get("energyDelta", 0.0)))
    camelot_score = _bed_camelot_score(drums_seg, bass_seg, other_seg)
    bpm_score = _bed_bpm_score(drums_seg, bass_seg, other_seg)
    score = (
        0.25 * drum_activity
        + 0.20 * bass_stability
        + 0.20 * (1.0 - vocal_leakage)
        + 0.15 * loopability
        + 0.10 * camelot_score
        + 0.10 * bpm_score
    ) * 100
    bars = max(1, int(drums_seg.get("barEnd", 0)) - int(drums_seg.get("barStart", 0)))
    return {
        "id": f"bed_{source}_{drums_source}drums_{bass_source}bass_{other_source}other_{uuid.uuid4().hex[:6]}",
        "source": source,
        "drumsSource": drums_source,
        "bassSource": bass_source,
        "otherSource": other_source,
        "drumsTrackId": drums_track.get("id"),
        "bassTrackId": bass_track.get("id"),
        "otherTrackId": other_track.get("id"),
        "sourceStart": round(start, 3),
        "sourceEnd": round(end, 3),
        "drumsStart": float(drums_seg.get("start", start)),
        "drumsEnd": float(drums_seg.get("end", end)),
        "bassStart": float(bass_seg.get("start", start)),
        "bassEnd": float(bass_seg.get("end", end)),
        "otherStart": float(other_seg.get("start", start)),
        "otherEnd": float(other_seg.get("end", end)),
        "bars": bars,
        "bpm": float(drums_seg.get("bpm") or drums_track.get("bpm") or 120),
        "camelot": drums_seg.get("camelot") or drums_track.get("camelot"),
        "energy": float(np.mean([float(drums_seg.get("energy", 0.5)), float(bass_seg.get("energy", 0.5)), float(other_seg.get("energy", 0.5))])),
        "bassEnergy": bass_energy,
        "drumActivity": drum_activity,
        "vocalLeakage": round(vocal_leakage, 4),
        "loopability": round(loopability, 4),
        "score": round(float(np.clip(score, 0, 100)), 1),
        "warnings": [] if vocal_leakage <= 0.18 else ["Bed section may contain vocal leakage."],
        "reason": "Scored for groove strength, stable bass, low vocal leakage, loopability, key and tempo compatibility.",
    }


def _best_bed_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(segments, key=lambda seg: (float(seg.get("drumActivity", 0)) * 0.42 + float(seg.get("bassEnergy", 0)) * 0.24 + (1 - float(seg.get("vocalDensity", 0.5))) * 0.24 + float(seg.get("mixInScore", 0)) * 0.10), reverse=True)


def _render_bed_loop(bed: dict[str, Any], assets: dict[str, dict[str, Any]], duration: float, sr: int) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []
    layers = []
    for stem, track_key, start_key, end_key in (
        ("drums", "drumsTrackId", "drumsStart", "drumsEnd"),
        ("bass", "bassTrackId", "bassStart", "bassEnd"),
        ("other", "otherTrackId", "otherStart", "otherEnd"),
    ):
        track_id = str(bed.get(track_key))
        paths = _cached_stem_paths(track_id)
        if not paths:
            warnings.append(f"Missing stems for groove bed {stem} source {track_id}.")
            continue
        audio = _slice(_load_stereo(paths[stem]), float(bed.get(start_key)), float(bed.get(end_key)), sr)
        if stem == "bass":
            mono = np.mean(audio, axis=0, keepdims=True)
            audio = np.vstack([mono, mono])
        layers.append(audio)
    if not layers:
        raise ValueError("Groove bed has no usable stems.")
    loop = _peak_limit(_sum_layers(layers), 0.85)
    total = max(1, int(math.ceil(duration * sr)))
    out = np.zeros((2, total), dtype=np.float32)
    pos = 0
    fade = int(0.03 * sr)
    while pos < total:
        length = min(loop.shape[1], total - pos)
        chunk = loop[:, :length].copy()
        if pos > 0 and fade > 1 and length > fade:
            curve = np.linspace(0, 1, fade, dtype=np.float32)
            out[:, pos : pos + fade] = out[:, pos : pos + fade] * (1 - curve) + chunk[:, :fade] * curve
            out[:, pos + fade : pos + length] += chunk[:, fade:length]
        else:
            out[:, pos : pos + length] += chunk[:, :length]
        pos += max(1, loop.shape[1] - fade)
    return _peak_limit(_section_energy_motion(out, sr), 0.9), warnings


def _render_vocal_event(event: dict[str, Any], assets: dict[str, dict[str, Any]], sr: int) -> tuple[np.ndarray, list[str]]:
    track_id = str(event.get("trackId"))
    paths = _cached_stem_paths(track_id)
    if not paths:
        return np.zeros((2, 1), dtype=np.float32), [f"Missing vocals stem for {track_id}; vocal phrase skipped."]
    vocal = _slice(_load_stereo(paths["vocals"]), float(event.get("sourceStart", 0)), float(event.get("sourceEnd", 0)), sr)
    ratio = float(event.get("stretchRatio") or 1.0)
    if 0.94 <= ratio <= 1.06 and abs(ratio - 1.0) > 0.015:
        vocal = np.ascontiguousarray(librosa.effects.time_stretch(vocal, rate=ratio), dtype=np.float32)
    vocal = _sos_filter(vocal, "highpass", 95)
    vocal = _compress(vocal, threshold=0.08, ratio=2.2)
    vocal = _normalize_rms(vocal, 0.09)
    vocal = _room(vocal, sr, wet=0.055)
    vocal = _microfade(vocal, sr, 0.01)
    target = max(1, int((float(event.get("timelineEnd", 0)) - float(event.get("timelineStart", 0))) * sr))
    if vocal.shape[1] < target:
        vocal = np.pad(vocal, ((0, 0), (0, target - vocal.shape[1])))
    return _peak_limit(vocal[:, :target], 0.8), []


def _select_phrase_sequence(bed: dict[str, Any], a: list[dict[str, Any]], b: list[dict[str, Any]], mode: str, vocal_priority: str) -> list[dict[str, Any]]:
    a_sorted = sorted(a, key=lambda item: item["score"], reverse=True)
    b_sorted = sorted(b, key=lambda item: item["score"], reverse=True)
    if mode == "a_vocal_on_b_groove":
        order = [*a_sorted[:4], *b_sorted[:2]]
    elif mode == "b_vocal_on_a_groove":
        order = [*b_sorted[:4], *a_sorted[:2]]
    elif mode == "hook_exchange_groove":
        order = _alternate(a_sorted[:4], b_sorted[:4])
    elif bed.get("source") == "A":
        order = _alternate(b_sorted[:4], a_sorted[:3])
    elif bed.get("source") == "B":
        order = _alternate(a_sorted[:4], b_sorted[:3])
    else:
        order = _alternate(a_sorted[:4], b_sorted[:4])
    if vocal_priority == "prefer_a":
        order = _alternate(a_sorted[:5], b_sorted[:2])
    elif vocal_priority == "prefer_b":
        order = _alternate(b_sorted[:5], a_sorted[:2])
    return [item for item in order if item]


def _alternate(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for index in range(max(len(left), len(right))):
        if index < len(left):
            result.append(left[index])
        if index < len(right):
            result.append(right[index])
    return result


def _groove_ui_items(arrangement: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": event["id"],
            "source": event["source"],
            "trackId": event["trackId"],
            "segmentId": event["phraseId"],
            "sourceStart": event["sourceStart"],
            "sourceEnd": event["sourceEnd"],
            "timelineStart": event["timelineStart"],
            "timelineEnd": event["timelineEnd"],
            "layerMode": "vocals",
            "transitionIn": {"type": event.get("handoffToNext", "call_response"), "durationSec": 0.0},
            "transitionOut": {"type": event.get("tailTreatment", "natural"), "durationSec": float(event.get("phrase", {}).get("tailDuration", 0.0))},
            "quality": {"score": event.get("phrase", {}).get("score", 0), "warnings": event.get("warnings", [])},
        }
        for event in arrangement.get("vocalEvents", [])
    ]


def _vocal_density(track: dict[str, Any], start: float, end: float, fallback: float = 0.5) -> float:
    curve = track.get("vocal_density_curve") or (track.get("transition_candidates") or {}).get("vocal_density_curve") or []
    values = [float(item.get("density")) for item in curve if item.get("time") is not None and start <= float(item["time"]) < end and item.get("density") is not None]
    return float(np.mean(values)) if values else float(np.clip(fallback, 0, 1))


def _vocal_energy(track: dict[str, Any], vocal_audio: np.ndarray | None, start: float, end: float) -> float:
    if vocal_audio is not None and end > start:
        clip = _slice(vocal_audio, start, end, SAMPLE_RATE)
        rms = float(np.sqrt(np.mean(np.square(clip)) + 1e-12))
        return float(np.clip(rms * 8.0, 0, 1))
    return _vocal_density(track, start, end, fallback=0.3)


def _estimate_tail(track: dict[str, Any], vocal_audio: np.ndarray | None, start: float, end: float, base: float) -> float:
    if end <= start:
        return 0.0
    after = _vocal_energy(track, vocal_audio, start, end)
    return float(np.clip((after / max(base, 1e-5)) * 0.8, 0.0, 1.5))


def _loopability(track: dict[str, Any], stems: dict[str, Path] | None, start: float, end: float, segment: dict[str, Any]) -> float:
    if stems and all(stem in stems for stem in ("drums", "bass", "other")) and end - start > 1.0:
        parts = [_slice(_load_stereo(stems[name]), start, end, SAMPLE_RATE) for name in ("drums", "bass", "other")]
        audio = _sum_layers(parts)
        one_bar = max(256, int((240 / max(float(track.get("bpm") or 120), 1.0)) * SAMPLE_RATE))
        head = audio[:, : min(one_bar, audio.shape[1])]
        tail = audio[:, -head.shape[1] :]
        rms_diff = abs(_rms(head) - _rms(tail)) / max(_rms(head), _rms(tail), 1e-6)
        cent_diff = abs(_centroid(head) - _centroid(tail)) / 8000.0
        return float(np.clip(1.0 - rms_diff * 0.55 - cent_diff * 0.45, 0, 1))
    return float(np.clip(1.0 - abs(float(segment.get("energyDelta", 0.0))), 0, 1))


def _bed_camelot_score(*segments: dict[str, Any]) -> float:
    values = [seg.get("camelot") for seg in segments if seg.get("camelot")]
    if len(set(values)) <= 1:
        return 1.0
    relations = [camelot_key_distance(values[0], value).get("score", 0) / 100 for value in values[1:]]
    return float(np.mean(relations)) if relations else 0.7


def _bed_bpm_score(*segments: dict[str, Any]) -> float:
    bpms = [float(seg.get("bpm") or 0) for seg in segments if float(seg.get("bpm") or 0) > 0]
    if not bpms:
        return 0.6
    median = float(np.median(bpms))
    return float(np.mean([1.0 - min(1.0, abs(median / bpm - 1.0) / 0.12) for bpm in bpms]))


def _arrangement_score(bed: dict[str, Any], events: list[dict[str, Any]], warnings: list[str]) -> float:
    sources = {event.get("source") for event in events}
    source_bonus = 12 if {"A", "B"}.issubset(sources) else -12
    count_bonus = min(16, len(events) * 3)
    return round(float(np.clip(bed.get("score", 0) * 0.55 + count_bonus + source_bonus - len(warnings) * 3, 0, 100)), 1)


def _section_plan(events: list[dict[str, Any]], bar_sec: float) -> list[dict[str, Any]]:
    sections = []
    section_len = bar_sec * 8
    end = max([event["timelineEnd"] for event in events] + [0])
    cursor = 0.0
    while cursor < end:
        sections.append({"start": round(cursor, 3), "end": round(cursor + section_len, 3), "role": "vocal_handoff"})
        cursor += section_len
    return sections


def _duck_bed_with_vocals(bed: np.ndarray, vocals: np.ndarray, sr: int, amount_db: float) -> np.ndarray:
    length = min(bed.shape[1], vocals.shape[1])
    if length <= 0:
        return bed
    activity = np.mean(np.abs(vocals[:, :length]), axis=0)
    frame = max(128, int(0.04 * sr))
    activity = np.convolve(activity, np.ones(frame, dtype=np.float32) / frame, mode="same")
    threshold = max(float(np.percentile(activity, 64)), 1e-5)
    duck = np.where(activity > threshold, _db_to_gain(amount_db), 1.0).astype(np.float32)
    release = np.ones(max(16, int(0.09 * sr)), dtype=np.float32)
    release /= np.sum(release)
    duck = np.convolve(duck, release, mode="same")
    out = bed.copy()
    out[:, :length] *= duck
    return out


def _section_energy_motion(audio: np.ndarray, sr: int) -> np.ndarray:
    out = audio.copy()
    section = max(1, int(sr * 16))
    for index, start in enumerate(range(0, out.shape[1], section)):
        end = min(out.shape[1], start + section)
        gain = _db_to_gain((index % 3) * 0.35)
        out[:, start:end] *= gain
    return _peak_limit(out, 0.9)


def _bus_glue(audio: np.ndarray) -> np.ndarray:
    return np.tanh(audio * 1.08).astype(np.float32)


def _room(audio: np.ndarray, sr: int, wet: float) -> np.ndarray:
    tail = np.zeros_like(audio)
    for ms, gain in ((43, 0.18), (89, 0.12), (137, 0.08)):
        offset = int(ms / 1000 * sr)
        if offset < audio.shape[1]:
            tail[:, offset:] += audio[:, : audio.shape[1] - offset] * gain
    return _peak_limit(audio + tail * wet, 0.9)


def _delay_tail(audio: np.ndarray, sr: int, bpm: float, wet: float) -> np.ndarray:
    delay = max(1, int((60 / max(bpm, 1.0)) * 0.5 * sr))
    out = np.zeros((2, min(int(2.0 * sr), delay * 4 + audio.shape[1])), dtype=np.float32)
    source = audio[:, -min(audio.shape[1], int(0.8 * sr)) :]
    gain = wet
    offset = 0
    while offset < out.shape[1] and gain > 0.02:
        length = min(source.shape[1], out.shape[1] - offset)
        out[:, offset : offset + length] += source[:, :length] * gain
        gain *= 0.42
        offset += delay
    out *= np.linspace(1, 0, out.shape[1], dtype=np.float32)
    return _sos_filter(out, "highpass", 180)


def _compress(audio: np.ndarray, threshold: float, ratio: float) -> np.ndarray:
    mag = np.abs(audio)
    over = mag > threshold
    out = audio.copy()
    out[over] = np.sign(audio[over]) * (threshold + (mag[over] - threshold) / ratio)
    return out.astype(np.float32)


def _normalize_rms(audio: np.ndarray, target: float) -> np.ndarray:
    rms = _rms(audio)
    return (audio * float(np.clip(target / max(rms, 1e-6), 0.35, 3.0))).astype(np.float32)


def _safe_load_mono(track: dict[str, Any]) -> np.ndarray | None:
    path = track.get("path")
    if not path:
        return None
    try:
        y, _ = librosa.load(Path(path), sr=SAMPLE_RATE, mono=True)
        return np.ascontiguousarray(y, dtype=np.float32)
    except Exception:
        return None


def _cached_stem_paths(track_id: str) -> dict[str, Path] | None:
    root = STEM_DIR / track_id / "demucs_api"
    paths = {name: root / f"{name}.wav" for name in STEM_NAMES}
    return paths if all(path.exists() for path in paths.values()) else None


def _load_stereo(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=False)
    if y.ndim == 1:
        y = np.vstack([y, y])
    if y.shape[0] > 2:
        y = y[:2]
    return np.ascontiguousarray(y, dtype=np.float32)


def _slice(audio: np.ndarray, start: float, end: float, sr: int) -> np.ndarray:
    start_sample = max(0, int(start * sr))
    end_sample = max(start_sample + 1, min(audio.shape[1], int(end * sr)))
    return np.ascontiguousarray(audio[:, start_sample:end_sample], dtype=np.float32)


def _sum_layers(layers: list[np.ndarray]) -> np.ndarray:
    length = max(layer.shape[1] for layer in layers)
    out = np.zeros((2, length), dtype=np.float32)
    for layer in layers:
        out[:, : layer.shape[1]] += layer
    return _peak_limit(out, 0.95)


def _sos_filter(audio: np.ndarray, kind: str, freq: float) -> np.ndarray:
    try:
        sos = signal.butter(2, freq, kind, fs=SAMPLE_RATE, output="sos")
        return signal.sosfilt(sos, audio, axis=1).astype(np.float32)
    except Exception:
        return audio.astype(np.float32)


def _microfade(audio: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    out = audio.copy()
    samples = min(out.shape[1] // 2, max(1, int(seconds * sr)))
    if samples > 1:
        curve = np.linspace(0, 1, samples, dtype=np.float32)
        out[:, :samples] *= curve
        out[:, -samples:] *= curve[::-1]
    return out.astype(np.float32)


def _peak_limit(audio: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    out = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > ceiling:
        out *= ceiling / peak
    return np.clip(out, -ceiling, ceiling).astype(np.float32)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def _centroid(audio: np.ndarray) -> float:
    y = np.mean(audio, axis=0)
    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(y.size, 1 / SAMPLE_RATE)
    return float(np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-9))


def _db_to_gain(db: float) -> float:
    return float(10 ** (db / 20.0))


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result
