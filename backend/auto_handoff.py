from __future__ import annotations

import math
from typing import Any

import numpy as np

from .matching import camelot_key_distance


DEFAULT_SETTINGS = {
    "phraseBars": 8,
    "maxTempoChangePercent": 10,
    "preferStems": True,
    "targetEnergy": "arc",
}


def build_auto_handoff_plan(tracks: list[dict[str, Any]], settings: dict[str, Any] | None = None) -> dict[str, Any]:
    opts = {**DEFAULT_SETTINGS, **(settings or {})}
    ready = [track for track in tracks if track.get("id")]
    if len(ready) < 2:
        raise ValueError("At least two tracks are required for Smart Beat Handoff")

    enriched = [_with_handoff_features(track) for track in ready]
    ordered = _order_tracks(enriched, opts)
    transitions = [_plan_pair(ordered[index - 1], ordered[index], opts) for index in range(1, len(ordered))]
    score = float(np.mean([item["score"] for item in transitions])) if transitions else 0.0
    return {
        "orderedTrackIds": [track["id"] for track in ordered],
        "score": round(score, 1),
        "transitions": transitions,
        "summary": _summary(transitions),
    }


def _with_handoff_features(track: dict[str, Any]) -> dict[str, Any]:
    cues = _cue_candidates(track)
    mix_in_cues = sorted([cue for cue in cues if cue["direction"] == "in"], key=lambda item: _cue_rank_score(item), reverse=True)[:5]
    mix_out_cues = [_with_exit_guard(track, cue) for cue in cues if cue["direction"] == "out"]
    grid = _beat_grid_confidence(track)
    groove = _groove_quality(track)
    vocal = _track_vocal_density(track)
    return {
        **track,
        "_handoff": {
            "mixInCues": mix_in_cues,
            "mixOutCues": sorted(mix_out_cues, key=lambda item: _cue_rank_score(item), reverse=True)[:8],
            "gridConfidence": grid,
            "grooveQuality": groove,
            "vocalDensity": vocal,
        },
    }


def _cue_candidates(track: dict[str, Any]) -> list[dict[str, Any]]:
    duration = _float(track.get("duration"), 0.0)
    candidates = track.get("transition_candidates") or {}
    sections = _sections(track)
    anchors = _anchors(track)
    cue_pool: list[tuple[str, float, str]] = []
    analyzed = _analyzed_cues(candidates)
    if analyzed:
        return analyzed

    if _is_number(candidates.get("intro")):
        cue_pool.append(("mix_in", float(candidates["intro"]), "analysis_intro"))
    if _is_number(candidates.get("outro")):
        cue_pool.append(("mix_out", float(candidates["outro"]), "analysis_outro"))

    for section in sections:
        start = _section_start(section)
        end = _section_end(section)
        kind = str(section.get("type") or "section")
        if 0.5 <= start <= max(0.5, duration * 0.55):
            cue_pool.append(("mix_in", start, f"{kind}_start"))
        if duration * 0.45 <= end <= max(0.5, duration - 0.25):
            cue_pool.append(("mix_out", end, f"{kind}_end"))

    if anchors:
        cue_pool.extend(("mix_in", value, "beat_anchor") for value in anchors if 0.5 <= value <= duration * 0.4)
        cue_pool.extend(("mix_out", value, "beat_anchor") for value in anchors if duration * 0.55 <= value <= duration - 0.25)

    if not cue_pool and duration > 1:
        cue_pool.extend(
            [
                ("mix_in", min(duration * 0.12, 12.0), "fallback"),
                ("mix_out", max(duration * 0.7, duration - 24.0), "fallback"),
            ]
        )

    deduped: dict[tuple[str, int], dict[str, Any]] = {}
    for role, time_value, source in cue_pool:
        time_value = _clamp(time_value, 0.0, max(0.0, duration - 0.25))
        direction = "in" if role == "mix_in" else "out"
        cue = _score_cue(track, time_value, direction, source)
        key = (direction, int(round(time_value * 2)))
        if key not in deduped or cue["score"] > deduped[key]["score"]:
            deduped[key] = cue
    return list(deduped.values())


def _score_cue(track: dict[str, Any], time_value: float, direction: str, source: str) -> dict[str, Any]:
    duration = _float(track.get("duration"), 0.0)
    candidates = track.get("transition_candidates") or {}
    phrase_alignment = _alignment_score(track, time_value)
    vocal_safety = 1.0 - _curve_value(candidates.get("vocal_density_curve") or track.get("vocal_density_curve"), time_value, "density", _boundary_vocal_hint(candidates, direction))
    groove = _local_groove_score(track, time_value)
    energy = _curve_value(candidates.get("energy_curve") or track.get("energy_curve"), time_value, "energy", _boundary_energy_hint(candidates, direction, track))
    structural = 1.0 if source != "beat_anchor" else 0.72
    room = _duration_room_score(duration, time_value, direction)
    energy_fit = 1.0 - abs(energy - (0.45 if direction == "in" else 0.42))
    score = (
        phrase_alignment * 0.28
        + vocal_safety * 0.22
        + groove * 0.18
        + max(0.0, energy_fit) * 0.14
        + structural * 0.10
        + room * 0.08
    )
    return {
        "time": round(float(time_value), 3),
        "role": "mix_in" if direction == "in" else "mix_out",
        "direction": direction,
        "score": round(float(_clamp(score, 0.0, 1.0) * 100), 1),
        "source": source,
        "metrics": {
            "phraseAlignment": round(phrase_alignment, 3),
            "vocalSafety": round(vocal_safety, 3),
            "grooveStability": round(groove, 3),
            "energyFit": round(max(0.0, energy_fit), 3),
            "durationRoom": round(room, 3),
        },
    }


def _analyzed_cues(candidates: dict[str, Any]) -> list[dict[str, Any]]:
    raw = candidates.get("cue_candidates") or []
    if not isinstance(raw, list):
        return []
    cues: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or not _is_number(item.get("time")):
            continue
        role = str(item.get("role") or "")
        direction = _direction_for_role(role)
        if not direction:
            continue
        cues.append(
            {
                "time": round(float(item["time"]), 3),
                "role": role,
                "direction": direction,
                "score": round(_float(item.get("score"), 50.0), 1),
                "source": item.get("source") or "cue_detector_v2",
                "metrics": item.get("components") or {},
                "reasons": item.get("reasons") or [],
            }
        )
    return cues


def _direction_for_role(role: str) -> str | None:
    if role in {"mix_in", "drop", "drum_loop"}:
        return "in"
    if role in {"mix_out", "bridge", "vocal_safe"}:
        return "out"
    return None


def _order_tracks(tracks: list[dict[str, Any]], settings: dict[str, Any]) -> list[dict[str, Any]]:
    remaining = sorted(tracks, key=lambda track: (_track_energy(track), _float(track.get("bpm"), 120.0)))
    ordered = [remaining.pop(0)]
    while remaining:
        current = ordered[-1]
        best_index = max(range(len(remaining)), key=lambda index: _pair_score(current, remaining[index], settings)["score"])
        ordered.append(remaining.pop(best_index))
    return ordered


def _plan_pair(prev: dict[str, Any], next_track: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    pair = _pair_score(prev, next_track, settings)
    out_cue = _best_cue(prev, "out")
    in_cue = _best_cue(next_track, "in")
    out_guard = out_cue.get("lyricGuard") or {}
    bpm = _compatible_bpm(prev, next_track)
    beat_seconds = 60.0 / max(bpm["target"], 1.0)
    bars = _select_bar_count(pair, prev, next_track, settings)
    transition_type = _transition_type(pair, prev, next_track)
    if out_guard.get("risk") == "high":
        transition_type = "effect_tail_handoff"
        bars = min(4, bars)
    duration = bars * 4 * beat_seconds
    rhythm_bed = _rhythm_bed(prev, next_track, transition_type)
    risk = _risk_level(pair)
    warnings = _warnings(pair, prev, next_track, transition_type)
    if out_guard.get("risk") == "high":
        risk = "high"
    elif out_guard.get("risk") == "medium" and risk == "low":
        risk = "medium"
    return {
        "fromTrackId": prev.get("id"),
        "toTrackId": next_track.get("id"),
        "fromName": prev.get("name"),
        "toName": next_track.get("name"),
        "type": transition_type,
        "score": round(pair["score"], 1),
        "barCount": bars,
        "durationSec": round(duration, 3),
        "outgoingCue": _public_cue(out_cue),
        "incomingCue": _public_cue(in_cue),
        "rhythmBed": rhythm_bed,
        "automation": {
            "sharedDrumBed": transition_type in {"drum_bed_handoff", "percussive_loop_bridge", "bass_swap_handoff"},
            "bassSwapAt": 0.55 if pair["bassSafety"] < 0.86 or transition_type == "bass_swap_handoff" else 0.68,
            "vocalDuck": pair["vocalSafety"] < 0.72,
            "tempoStretchRatio": round(bpm["stretch"], 5),
            "targetBpm": round(bpm["target"], 3),
            "outgoingLyricGuard": out_guard,
        },
        "risk": risk,
        "warnings": warnings,
        "components": {key: round(float(value) * 100, 1) for key, value in pair["components"].items()},
        "explanation": _explanation(prev, next_track, transition_type, rhythm_bed, risk, warnings),
    }


def _public_cue(cue: dict[str, Any]) -> dict[str, Any]:
    return {
        "time": cue.get("time"),
        "role": cue.get("role"),
        "score": cue.get("score"),
        "handoffScore": cue.get("handoffScore", cue.get("score")),
        "source": cue.get("source"),
        "reasons": cue.get("reasons") or [],
        "metrics": cue.get("metrics") or {},
        "lyricGuard": cue.get("lyricGuard"),
    }


def _pair_score(prev: dict[str, Any], next_track: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    tempo = _tempo_score(prev, next_track)
    cue = (_cue_rank_score(_best_cue(prev, "out")) + _cue_rank_score(_best_cue(next_track, "in"))) / 200
    rhythm = (_handoff(prev)["grooveQuality"] + _handoff(next_track)["grooveQuality"]) / 2
    vocal = _vocal_safety(prev, next_track)
    bass = _bass_safety(prev, next_track)
    harmonic = _harmonic_score(prev, next_track)
    energy = _energy_flow_score(prev, next_track, settings)
    score = (
        tempo * 0.24
        + cue * 0.18
        + rhythm * 0.16
        + vocal * 0.14
        + bass * 0.12
        + harmonic * 0.10
        + energy * 0.06
    )
    return {
        "score": _clamp(score, 0.0, 1.0) * 100,
        "tempo": tempo,
        "cue": cue,
        "rhythmBedQuality": rhythm,
        "vocalSafety": vocal,
        "bassSafety": bass,
        "harmonic": harmonic,
        "energy": energy,
        "components": {
            "tempo": tempo,
            "cue": cue,
            "rhythmBed": rhythm,
            "vocalSafety": vocal,
            "bassSafety": bass,
            "harmonic": harmonic,
            "energyFlow": energy,
        },
    }


def _transition_type(pair: dict[str, Any], prev: dict[str, Any], next_track: dict[str, Any]) -> str:
    grid = min(_handoff(prev)["gridConfidence"], _handoff(next_track)["gridConfidence"])
    if pair["tempo"] < 0.45 or grid < 0.32:
        return "effect_tail_handoff"
    if pair["vocalSafety"] < 0.55:
        return "vocal_safe_bridge"
    if grid < 0.58:
        return "percussive_loop_bridge"
    if pair["bassSafety"] < 0.72:
        return "bass_swap_handoff"
    return "drum_bed_handoff"


def _select_bar_count(pair: dict[str, Any], prev: dict[str, Any], next_track: dict[str, Any], settings: dict[str, Any]) -> int:
    requested = int(settings.get("phraseBars") or 8)
    requested = 16 if requested >= 16 else 8 if requested >= 8 else 4
    grid = min(_handoff(prev)["gridConfidence"], _handoff(next_track)["gridConfidence"])
    if pair["tempo"] < 0.5 or pair["vocalSafety"] < 0.55 or grid < 0.45:
        return 4
    if pair["score"] >= 82 and grid >= 0.72:
        return max(8, requested)
    return min(8, requested)


def _best_cue(track: dict[str, Any], direction: str) -> dict[str, Any]:
    cues = _handoff(track)["mixInCues" if direction == "in" else "mixOutCues"]
    if cues:
        if direction == "out":
            safe = [cue for cue in cues if (cue.get("lyricGuard") or {}).get("risk") != "high"]
            if safe:
                return safe[0]
        return cues[0]
    duration = _float(track.get("duration"), 0.0)
    time_value = min(8.0, duration * 0.12) if direction == "in" else max(0.0, duration - 16.0)
    cue = _score_cue(track, time_value, direction, "fallback")
    return _with_exit_guard(track, cue) if direction == "out" else cue


def _compatible_bpm(prev: dict[str, Any], next_track: dict[str, Any]) -> dict[str, float]:
    a = _float(prev.get("bpm"), 120.0)
    b = _float(next_track.get("bpm"), a)
    candidates = [b, b * 2, b / 2]
    compatible = min(candidates, key=lambda value: abs(a - value))
    stretch = a / max(compatible, 1e-6)
    return {"target": a, "incoming": compatible, "stretch": stretch}


def _tempo_score(prev: dict[str, Any], next_track: dict[str, Any]) -> float:
    bpm = _compatible_bpm(prev, next_track)
    pct = abs(bpm["incoming"] - bpm["target"]) / max(bpm["target"], 1.0)
    return _clamp(1.0 - pct / 0.12, 0.0, 1.0)


def _rhythm_bed(prev: dict[str, Any], next_track: dict[str, Any], transition_type: str) -> dict[str, Any]:
    prev_quality = _handoff(prev)["grooveQuality"]
    next_quality = _handoff(next_track)["grooveQuality"]
    source = "A" if prev_quality >= next_quality else "B"
    if transition_type == "effect_tail_handoff":
        return {"source": "none", "stem": "full", "reason": "Beat grid is too risky for a shared drum bed."}
    return {
        "source": source,
        "stem": "drums",
        "reason": f"Use {'outgoing' if source == 'A' else 'incoming'} drums as the most stable transition rhythm bed.",
    }


def _warnings(pair: dict[str, Any], prev: dict[str, Any], next_track: dict[str, Any], transition_type: str) -> list[str]:
    warnings: list[str] = []
    out_guard = (_best_cue(prev, "out").get("lyricGuard") or {})
    if out_guard.get("risk") == "high":
        warnings.append("Outgoing lyric cut risk is high; move the exit cue later or render an FX tail after the vocal phrase.")
    elif out_guard.get("risk") == "medium":
        warnings.append("Outgoing vocal phrase is only moderately safe; keep overlap short and preserve the outgoing tail.")
    if pair["tempo"] < 0.55:
        warnings.append("Tempo match is weak; keep the handoff short.")
    if pair["vocalSafety"] < 0.65:
        warnings.append("Vocal overlap risk; duck or avoid one vocal stem.")
    if min(_handoff(prev)["gridConfidence"], _handoff(next_track)["gridConfidence"]) < 0.55:
        warnings.append("Beat grid is locally unstable; prefer bridge or loop-based handoff.")
    if transition_type == "effect_tail_handoff":
        warnings.append("Shared drum bed is unsafe; use FX tail instead of long beatmix.")
    return warnings


def _risk_level(pair: dict[str, Any]) -> str:
    if pair["score"] >= 78:
        return "low"
    if pair["score"] >= 60:
        return "medium"
    return "high"


def _cue_rank_score(cue: dict[str, Any]) -> float:
    return _float(cue.get("handoffScore", cue.get("score")), 0.0)


def _with_exit_guard(track: dict[str, Any], cue: dict[str, Any]) -> dict[str, Any]:
    if cue.get("direction") != "out":
        return cue
    duration = _float(track.get("duration"), 0.0)
    time_value = _float(cue.get("time"), 0.0)
    guard = _outgoing_lyric_guard(track, time_value, str(cue.get("role") or "mix_out"))
    adjusted = _float(cue.get("score"), 50.0) * guard["multiplier"]
    reasons = list(cue.get("reasons") or [])
    if guard["risk"] == "low":
        reasons.append("vocal phrase cleared")
    elif guard["risk"] == "high":
        reasons.append("possible lyric cut")
    position = time_value / max(duration, 1e-6)
    if cue.get("role") == "vocal_safe" and position < 0.65:
        reasons.append("vocal_safe is early; treated as support cue only")
    return {
        **cue,
        "handoffScore": round(_clamp(adjusted, 0.0, 100.0), 1),
        "lyricGuard": guard,
        "reasons": reasons[:6],
    }


def _outgoing_lyric_guard(track: dict[str, Any], time_value: float, role: str) -> dict[str, Any]:
    duration = _float(track.get("duration"), 0.0)
    position = time_value / max(duration, 1e-6)
    stem_guard = _stem_vocal_guard(track, time_value, duration)
    curve = (track.get("transition_candidates") or {}).get("vocal_density_curve") or track.get("vocal_density_curve") or []
    has_curve = isinstance(curve, list) and bool(curve)
    before = _curve_average(curve, max(0.0, time_value - 5.0), 5.0, "density", 0.5) if has_curve else 0.5
    at = _curve_value(curve, time_value, "density", 0.5) if has_curve else 0.5
    after = _curve_average(curve, time_value, 8.0, "density", 0.5) if has_curve else 0.5
    after_peak = _curve_peak(curve, time_value, 8.0, "density", 0.5) if has_curve else 0.5
    reentry = max(0.0, after_peak - min(at, after))
    min_position = 0.45 if duration < 120 else 0.55
    position_score = _clamp((position - min_position) / 0.22, 0.0, 1.0)
    if role == "mix_out":
        role_score = 1.0
    elif role == "bridge":
        role_score = 0.78 if position >= min_position else 0.48
    elif role == "vocal_safe":
        role_score = 0.70 if position >= 0.65 else 0.32
    else:
        role_score = 0.6
    boundary = _section_boundary_score(track, time_value)
    phrase_release = _clamp(1.0 - after * 0.72 - after_peak * 0.42 - reentry * 0.36 + max(0.0, before - at) * 0.24, 0.0, 1.0)
    if stem_guard.get("available"):
        phrase_release = min(phrase_release, _float(stem_guard.get("releaseScore"), phrase_release))
    multiplier = _clamp(0.25 + position_score * 0.24 + phrase_release * 0.30 + role_score * 0.14 + boundary * 0.07, 0.18, 1.08)
    risk = "low"
    if stem_guard.get("risk") == "high" or position < min_position or (after_peak >= 0.72 and after >= 0.50) or (role == "vocal_safe" and position < 0.65):
        risk = "high"
    elif stem_guard.get("risk") == "medium" or phrase_release < 0.48 or after_peak >= 0.62 or position < min_position + 0.08:
        risk = "medium"
    return {
        "risk": risk,
        "source": "demucs_vocals" if stem_guard.get("available") else "vocal_density_curve",
        "multiplier": round(multiplier, 3),
        "position": round(position, 3),
        "minPosition": round(min_position, 3),
        "vocalBefore": round(before, 3),
        "vocalAtCue": round(at, 3),
        "vocalAfterAvg": round(after, 3),
        "vocalAfterPeak": round(after_peak, 3),
        "vocalReentry": round(reentry, 3),
        "phraseRelease": round(phrase_release, 3),
        "sectionBoundary": round(boundary, 3),
        "stemVocal": stem_guard,
    }


def _stem_vocal_guard(track: dict[str, Any], time_value: float, duration: float) -> dict[str, Any]:
    regions = _vocal_regions(track)
    if not regions:
        return {"available": False}
    lookahead = 8.0
    containing = next((region for region in regions if float(region["start"]) - 0.12 <= time_value <= float(region["end"]) + 0.12), None)
    active_after = _region_coverage(regions, time_value, time_value + lookahead)
    next_entry = _next_region_start(regions, time_value)
    last_release = _last_region_end(regions, time_value)
    release_age = time_value - last_release if last_release is not None else None
    inside_tail = max(0.0, float(containing["end"]) - time_value) if containing else 0.0
    next_entry_in = max(0.0, next_entry - time_value) if next_entry is not None else None
    release_score = 1.0
    if containing:
        release_score -= min(0.72, inside_tail / max(lookahead, 1e-6))
    release_score -= min(0.45, active_after * 0.75)
    if next_entry_in is not None and next_entry_in < 4.0:
        release_score -= (4.0 - next_entry_in) * 0.08
    if release_age is not None:
        release_score += min(0.18, release_age / 24.0)
    release_score = _clamp(release_score, 0.0, 1.0)
    risk = "low"
    if inside_tail > 0.9 or active_after >= 0.42:
        risk = "high"
    elif inside_tail > 0.25 or active_after >= 0.20 or (next_entry_in is not None and next_entry_in < 3.0):
        risk = "medium"
    return {
        "available": True,
        "risk": risk,
        "releaseScore": round(release_score, 3),
        "insideVocalTailSec": round(inside_tail, 3),
        "activeAfter": round(active_after, 3),
        "lastReleaseAgoSec": round(release_age, 3) if release_age is not None else None,
        "nextEntryInSec": round(next_entry_in, 3) if next_entry_in is not None else None,
    }


def _explanation(prev: dict[str, Any], next_track: dict[str, Any], kind: str, rhythm_bed: dict[str, Any], risk: str, warnings: list[str]) -> str:
    labels = {
        "drum_bed_handoff": "Use a shared drum bed so both songs ride one groove through the transition.",
        "bass_swap_handoff": "Keep one bass line active at a time and swap low end near the middle of the handoff.",
        "percussive_loop_bridge": "Use a short stable percussion loop because the beat grid is not reliable enough for a long blend.",
        "vocal_safe_bridge": "Protect the vocals by shortening overlap and ducking one side during the handoff.",
        "effect_tail_handoff": "Use an effect tail instead of forcing beatmatch on an unsafe pair.",
    }
    warning_text = f" Warnings: {'; '.join(warnings)}" if warnings else ""
    return f"{labels.get(kind, kind)} Rhythm bed: {rhythm_bed.get('source')} {rhythm_bed.get('stem')}. Risk: {risk}.{warning_text}"


def _summary(transitions: list[dict[str, Any]]) -> str:
    if not transitions:
        return "No transitions planned."
    counts: dict[str, int] = {}
    for item in transitions:
        counts[item["type"]] = counts.get(item["type"], 0) + 1
    mix = ", ".join(f"{key} x{value}" for key, value in sorted(counts.items()))
    return f"Planned {len(transitions)} Smart Beat Handoffs: {mix}."


def _handoff(track: dict[str, Any]) -> dict[str, Any]:
    return track.get("_handoff") or {}


def _beat_grid_confidence(track: dict[str, Any]) -> float:
    bars = [float(value) for value in track.get("bars") or [] if _is_number(value)]
    phrases = [float(value) for value in track.get("phrases") or [] if _is_number(value)]
    candidates = track.get("transition_candidates") or {}
    base = _float(candidates.get("confidence"), 0.35)
    if len(bars) >= 12:
        diffs = np.diff(np.asarray(bars, dtype=float))
        median = float(np.median(diffs)) if diffs.size else 0.0
        jitter = float(np.median(np.abs(diffs - median)) / max(median, 1e-6)) if median else 0.5
        base += _clamp(0.28 - jitter * 1.6, -0.18, 0.28)
    else:
        base -= 0.15
    if len(phrases) >= 3:
        base += 0.08
    return round(_clamp(base, 0.15, 0.95), 3)


def _groove_quality(track: dict[str, Any]) -> float:
    candidates = track.get("transition_candidates") or {}
    energy_curve = candidates.get("energy_curve") or track.get("energy_curve") or []
    values = [_float(item.get("energy"), 0.0) for item in energy_curve if isinstance(item, dict)]
    if not values:
        return _clamp(_track_energy(track), 0.25, 0.75)
    mean = float(np.mean(values))
    stability = 1.0 - min(1.0, float(np.std(values)) * 2.2)
    grid = _beat_grid_confidence(track)
    return round(_clamp(mean * 0.45 + stability * 0.25 + grid * 0.30, 0.1, 0.95), 3)


def _track_vocal_density(track: dict[str, Any]) -> float:
    candidates = track.get("transition_candidates") or {}
    values = []
    for key in ("intro_vocal_density", "outro_vocal_density"):
        if _is_number(candidates.get(key)):
            values.append(float(candidates[key]))
    curve = candidates.get("vocal_density_curve") or track.get("vocal_density_curve") or []
    values.extend(_float(item.get("density"), 0.5) for item in curve if isinstance(item, dict))
    return round(_clamp(float(np.mean(values)) if values else 0.5, 0.0, 1.0), 3)


def _vocal_safety(prev: dict[str, Any], next_track: dict[str, Any]) -> float:
    prev_vocal = _curve_value((prev.get("transition_candidates") or {}).get("vocal_density_curve"), _best_cue(prev, "out")["time"], "density", _handoff(prev)["vocalDensity"])
    next_vocal = _curve_value((next_track.get("transition_candidates") or {}).get("vocal_density_curve"), _best_cue(next_track, "in")["time"], "density", _handoff(next_track)["vocalDensity"])
    return _clamp(1.0 - (prev_vocal * 0.55 + next_vocal * 0.45), 0.0, 1.0)


def _bass_safety(prev: dict[str, Any], next_track: dict[str, Any]) -> float:
    prev_low = _low_frequency(prev)
    next_low = _low_frequency(next_track)
    return _clamp(1.0 - min(prev_low, next_low) * 0.35 - abs(prev_low - next_low) * 0.2, 0.0, 1.0)


def _harmonic_score(prev: dict[str, Any], next_track: dict[str, Any]) -> float:
    result = camelot_key_distance(prev.get("camelot") or prev.get("key"), next_track.get("camelot") or next_track.get("key"))
    return _clamp(_float(result.get("score"), 50.0) / 100, 0.0, 1.0)


def _energy_flow_score(prev: dict[str, Any], next_track: dict[str, Any], settings: dict[str, Any]) -> float:
    prev_energy = _track_energy(prev)
    next_energy = _track_energy(next_track)
    delta = next_energy - prev_energy
    target = settings.get("targetEnergy", "arc")
    if target == "up":
        return _clamp(0.65 + delta, 0.0, 1.0)
    if target == "down":
        return _clamp(0.65 - delta, 0.0, 1.0)
    return _clamp(1.0 - abs(delta) * 1.4, 0.0, 1.0)


def _track_energy(track: dict[str, Any]) -> float:
    profile = track.get("energy_profile") or {}
    return _clamp(_float(profile.get("energy_index"), _float(track.get("energy"), 0.5) * 100) / 100, 0.0, 1.0)


def _low_frequency(track: dict[str, Any]) -> float:
    profile = track.get("energy_profile") or {}
    return _clamp(_float(profile.get("low_frequency_ratio"), 0.45), 0.0, 1.0)


def _alignment_score(track: dict[str, Any], time_value: float) -> float:
    anchors = _anchors(track)
    if not anchors:
        return 0.45
    nearest = min(abs(anchor - time_value) for anchor in anchors)
    bpm = _float(track.get("bpm"), 120.0)
    beat = 60 / max(bpm, 1.0)
    return _clamp(1.0 - nearest / max(beat * 2, 0.25), 0.0, 1.0)


def _local_groove_score(track: dict[str, Any], time_value: float) -> float:
    grid = _beat_grid_confidence(track)
    candidates = track.get("transition_candidates") or {}
    energy = _curve_value(candidates.get("energy_curve") or track.get("energy_curve"), time_value, "energy", _track_energy(track))
    return _clamp(grid * 0.62 + energy * 0.38, 0.0, 1.0)


def _duration_room_score(duration: float, time_value: float, direction: str) -> float:
    if direction == "in":
        return _clamp(time_value / 24.0, 0.0, 1.0)
    if duration <= 0:
        return 0.0
    position = time_value / duration
    late_enough = _clamp((position - 0.48) / 0.26, 0.0, 1.0)
    tail_room = _clamp((duration - time_value) / 24.0, 0.0, 1.0)
    return _clamp(late_enough * 0.72 + tail_room * 0.28, 0.0, 1.0)


def _anchors(track: dict[str, Any]) -> list[float]:
    values = [float(value) for value in (track.get("phrases") or []) + (track.get("bars") or []) if _is_number(value)]
    return sorted(set(round(value, 3) for value in values))


def _sections(track: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = track.get("transition_candidates") or {}
    sections = track.get("sections") or candidates.get("sections") or []
    return [section for section in sections if isinstance(section, dict)]


def _vocal_regions(track: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = track.get("transition_candidates") or {}
    activity = track.get("vocal_activity") or candidates.get("vocal_activity") or {}
    raw = track.get("vocal_regions") or activity.get("regions") or []
    regions = []
    for item in raw:
        if not isinstance(item, dict) or not _is_number(item.get("start")) or not _is_number(item.get("end")):
            continue
        start = float(item["start"])
        end = float(item["end"])
        if end > start:
            regions.append({**item, "start": start, "end": end})
    return sorted(regions, key=lambda item: float(item["start"]))


def _region_coverage(regions: list[dict[str, Any]], start: float, end: float) -> float:
    if end <= start:
        return 0.0
    active = 0.0
    for region in regions:
        active += max(0.0, min(end, float(region["end"])) - max(start, float(region["start"])))
    return _clamp(active / max(end - start, 1e-6), 0.0, 1.0)


def _next_region_start(regions: list[dict[str, Any]], time_value: float) -> float | None:
    future = [float(region["start"]) for region in regions if float(region["start"]) > time_value + 0.12]
    return min(future) if future else None


def _last_region_end(regions: list[dict[str, Any]], time_value: float) -> float | None:
    past = [float(region["end"]) for region in regions if float(region["end"]) <= time_value + 0.12]
    return max(past) if past else None


def _section_start(section: dict[str, Any]) -> float:
    return _float(section.get("startTime", section.get("start", 0.0)), 0.0)


def _section_end(section: dict[str, Any]) -> float:
    return _float(section.get("endTime", section.get("end", _section_start(section))), _section_start(section))


def _curve_value(curve: Any, time_value: float, key: str, fallback: float) -> float:
    if not isinstance(curve, list) or not curve:
        return _clamp(fallback, 0.0, 1.0)
    points = [item for item in curve if isinstance(item, dict) and _is_number(item.get("time")) and _is_number(item.get(key))]
    if not points:
        return _clamp(fallback, 0.0, 1.0)
    nearest = min(points, key=lambda item: abs(float(item["time"]) - time_value))
    return _clamp(_float(nearest.get(key), fallback), 0.0, 1.0)


def _curve_average(curve: Any, start_time: float, duration: float, key: str, fallback: float) -> float:
    if duration <= 0:
        return _curve_value(curve, start_time, key, fallback)
    sample_count = max(2, min(16, int(math.ceil(duration)) + 1))
    values = [_curve_value(curve, start_time + float(offset), key, fallback) for offset in np.linspace(0, duration, sample_count)]
    return _clamp(float(np.mean(values)) if values else fallback, 0.0, 1.0)


def _curve_peak(curve: Any, start_time: float, duration: float, key: str, fallback: float) -> float:
    if duration <= 0:
        return _curve_value(curve, start_time, key, fallback)
    sample_count = max(2, min(18, int(math.ceil(duration * 2)) + 1))
    values = [_curve_value(curve, start_time + float(offset), key, fallback) for offset in np.linspace(0, duration, sample_count)]
    return _clamp(float(np.max(values)) if values else fallback, 0.0, 1.0)


def _section_boundary_score(track: dict[str, Any], time_value: float) -> float:
    boundaries: list[float] = []
    for section in _sections(track):
        boundaries.extend([_section_start(section), _section_end(section)])
    if not boundaries:
        return 0.45
    bpm = _float(track.get("bpm"), 120.0)
    beat = 60.0 / max(bpm, 1.0)
    nearest = min(abs(value - time_value) for value in boundaries)
    return _clamp(1.0 - nearest / max(beat * 8, 1.0), 0.0, 1.0)


def _boundary_vocal_hint(candidates: dict[str, Any], direction: str) -> float:
    key = "intro_vocal_density" if direction == "in" else "outro_vocal_density"
    return _clamp(_float(candidates.get(key), 0.5), 0.0, 1.0)


def _boundary_energy_hint(candidates: dict[str, Any], direction: str, track: dict[str, Any]) -> float:
    key = "intro_energy" if direction == "in" else "outro_energy"
    return _clamp(_float(candidates.get(key), _track_energy(track)), 0.0, 1.0)


def _float(value: Any, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    if not math.isfinite(result):
        return float(fallback)
    return result


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
