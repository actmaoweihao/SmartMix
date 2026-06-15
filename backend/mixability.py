from __future__ import annotations

import math
from typing import Any

import numpy as np

from .transition import plan_transition


PAIR_MATCH_WEIGHTS = {
    "harmonic": 0.40,
    "tempo": 0.25,
    "energy": 0.15,
    "structure": 0.10,
    "style": 0.10,
}

HANDOFF_WEIGHTS = {
    "tempo": 0.24,
    "cue": 0.18,
    "rhythm": 0.16,
    "vocal": 0.14,
    "bass": 0.12,
    "harmonic": 0.10,
    "energy": 0.06,
}

SORT_RECOMMENDED_WEIGHTS = {
    "tempo": 0.32,
    "harmonic": 0.24,
    "energy": 0.14,
    "cue": 0.10,
    "structure": 0.08,
    "vocal": 0.05,
    "bass": 0.04,
    "style": 0.03,
}

SORT_HARMONIC_WEIGHTS = {
    "harmonic": 0.45,
    "tempo": 0.18,
    "energy": 0.10,
    "structure": 0.09,
    "cue": 0.08,
    "vocal": 0.04,
    "bass": 0.03,
    "style": 0.03,
}


def evaluate_mixability(
    prev_track: dict[str, Any],
    next_track: dict[str, Any],
    *,
    profile: str = "pair_match",
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score how safely and musically prev_track can hand off to next_track."""
    settings = settings or {}
    components = {
        "tempo": tempo_compatibility(prev_track, next_track),
        "cue": cue_compatibility(prev_track, next_track),
        "rhythm": rhythm_compatibility(prev_track, next_track),
        "vocal": vocal_safety(prev_track, next_track),
        "bass": bass_safety(prev_track, next_track),
        "harmonic": harmonic_compatibility(prev_track, next_track),
        "energy": energy_continuity(prev_track, next_track, settings),
        "structure": structure_compatibility(prev_track, next_track),
        "style": style_compatibility(prev_track, next_track),
    }
    weights = _weights_for_profile(profile)
    raw_score = sum(components[key]["score"] * weight for key, weight in weights.items())
    adjusted = _adjust_score(raw_score, components)
    return {
        "profile": profile,
        "score": round(float(adjusted), 1),
        "rawScore": round(float(raw_score), 1),
        "level": rank_mixability(adjusted, components),
        "weights": weights,
        "components": components,
        "summary": _summary(adjusted, components),
    }


def order_tracks_by_mixability(
    tracks: list[dict[str, Any]],
    *,
    mode: str = "recommended",
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ready = [track for track in tracks if track.get("id")]
    if len(ready) < 2:
        raise ValueError("At least two tracks are required for mixability ordering.")
    profile = "sort_harmonic" if mode == "harmonic" else "sort_recommended"
    remaining = sorted(ready, key=lambda track: (track_energy(track), _float(track.get("bpm"), 120.0), str(track.get("name") or "")))
    ordered = [remaining.pop(0)]
    transitions: list[dict[str, Any]] = []
    while remaining:
        current = ordered[-1]
        scored = [(candidate, evaluate_mixability(current, candidate, profile=profile, settings=settings)) for candidate in remaining]
        candidate, score = max(scored, key=lambda item: (float(item[1]["score"]), float(item[1]["components"]["cue"].get("score") or 0)))
        remaining.remove(candidate)
        ordered.append(candidate)
        transitions.append(
            {
                "fromTrackId": current.get("id"),
                "toTrackId": candidate.get("id"),
                "fromName": current.get("name"),
                "toName": candidate.get("name"),
                "score": score["score"],
                "level": score["level"],
                "outgoingCue": score["components"]["cue"].get("outgoing"),
                "incomingCue": score["components"]["cue"].get("incoming"),
                "durationSec": score["components"]["structure"].get("overlapSeconds"),
                "phraseBars": score["components"]["structure"].get("phraseBars"),
                "transitionPlan": score["components"]["structure"].get("transition"),
                "components": _compact_components(score["components"]),
                "summary": score["summary"],
            }
        )
    score_values = [float(item["score"]) for item in transitions]
    return {
        "mode": mode,
        "profile": profile,
        "orderedTrackIds": [track["id"] for track in ordered],
        "score": round(float(np.mean(score_values)) if score_values else 0.0, 1),
        "transitions": transitions,
    }


def tempo_compatibility(prev_track: dict[str, Any], next_track: dict[str, Any]) -> dict[str, Any]:
    a = _float(prev_track.get("bpm"), 0.0)
    b = _float(next_track.get("bpm"), 0.0)
    if a <= 0 or b <= 0:
        return {"score": 50.0, "delta": None, "reason": "At least one track has unknown BPM."}
    candidates = [b, b * 2, b / 2]
    compatible = min(candidates, key=lambda value: abs(a - value))
    delta = abs(a - compatible)
    pct = delta / max(a, 1.0)
    score = _clamp(100.0 - pct * 420.0, 0.0, 100.0)
    stretch = a / max(compatible, 1e-6)
    return {
        "score": round(score, 1),
        "delta": round(delta, 2),
        "normalized_bpm_b": round(compatible, 2),
        "targetBpm": round(a, 3),
        "compatibleIncomingBpm": round(compatible, 3),
        "stretchRatio": round(stretch, 5),
        "reason": "Compares BPM difference with half/double tempo normalization.",
    }


def _weights_for_profile(profile: str) -> dict[str, float]:
    if profile == "handoff":
        return HANDOFF_WEIGHTS
    if profile == "sort_recommended":
        return SORT_RECOMMENDED_WEIGHTS
    if profile == "sort_harmonic":
        return SORT_HARMONIC_WEIGHTS
    return PAIR_MATCH_WEIGHTS


def _compact_components(components: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {key: round(float(value.get("score") or 0.0), 1) for key, value in components.items()}


def cue_compatibility(prev_track: dict[str, Any], next_track: dict[str, Any]) -> dict[str, Any]:
    out_cues = cues_for_direction(prev_track, "out")
    in_cues = cues_for_direction(next_track, "in")
    out_best = out_cues[0] if out_cues else fallback_cue(prev_track, "out")
    in_best = in_cues[0] if in_cues else fallback_cue(next_track, "in")
    score = (_float(out_best.get("score"), 50.0) + _float(in_best.get("score"), 50.0)) / 2
    return {
        "score": round(_clamp(score, 0.0, 100.0), 1),
        "outgoing": public_cue(out_best),
        "incoming": public_cue(in_best),
        "reason": "Averages the best outgoing and incoming cue candidates.",
    }


def rhythm_compatibility(prev_track: dict[str, Any], next_track: dict[str, Any]) -> dict[str, Any]:
    prev_grid = beat_grid_confidence(prev_track)
    next_grid = beat_grid_confidence(next_track)
    prev_groove = groove_quality(prev_track)
    next_groove = groove_quality(next_track)
    score = (prev_grid * 0.35 + next_grid * 0.35 + prev_groove * 0.15 + next_groove * 0.15) * 100
    return {
        "score": round(_clamp(score, 0.0, 100.0), 1),
        "gridConfidence": round(min(prev_grid, next_grid), 3),
        "prevGrid": round(prev_grid, 3),
        "nextGrid": round(next_grid, 3),
        "prevGroove": round(prev_groove, 3),
        "nextGroove": round(next_groove, 3),
        "reason": "Combines beat-grid confidence and local groove stability.",
    }


def vocal_safety(prev_track: dict[str, Any], next_track: dict[str, Any]) -> dict[str, Any]:
    out_time = cue_compatibility(prev_track, next_track)["outgoing"]["time"]
    in_time = cue_compatibility(prev_track, next_track)["incoming"]["time"]
    prev_vocal = curve_value(_vocal_curve(prev_track), out_time, "density", track_vocal_density(prev_track))
    next_vocal = curve_value(_vocal_curve(next_track), in_time, "density", track_vocal_density(next_track))
    stem_penalty = max(_stem_vocal_risk(prev_track, out_time), _stem_vocal_risk(next_track, in_time)) * 22
    score = (1.0 - (prev_vocal * 0.55 + next_vocal * 0.45)) * 100 - stem_penalty
    return {
        "score": round(_clamp(score, 0.0, 100.0), 1),
        "outgoingDensity": round(prev_vocal, 3),
        "incomingDensity": round(next_vocal, 3),
        "stemRiskPenalty": round(stem_penalty, 1),
        "reason": "Penalizes overlapping vocal density at the chosen outgoing/incoming cues.",
    }


def bass_safety(prev_track: dict[str, Any], next_track: dict[str, Any]) -> dict[str, Any]:
    prev_low = low_frequency_ratio(prev_track)
    next_low = low_frequency_ratio(next_track)
    score = (1.0 - min(prev_low, next_low) * 0.35 - abs(prev_low - next_low) * 0.2) * 100
    return {
        "score": round(_clamp(score, 0.0, 100.0), 1),
        "outgoingLowFrequency": round(prev_low, 4),
        "incomingLowFrequency": round(next_low, 4),
        "reason": "Penalizes dense low-frequency overlap and large low-end mismatch.",
    }


def harmonic_compatibility(prev_track: dict[str, Any], next_track: dict[str, Any]) -> dict[str, Any]:
    from .matching import camelot_key_distance

    prev_key = track_camelot(prev_track)
    next_key = track_camelot(next_track)
    result = camelot_key_distance(prev_key, next_key)
    return {
        **result,
        "score": round(float(result.get("score", 40.0)), 1),
        "from": prev_key,
        "to": next_key,
    }


def energy_continuity(prev_track: dict[str, Any], next_track: dict[str, Any], settings: dict[str, Any] | None = None) -> dict[str, Any]:
    from .matching import energy_match_score

    settings = settings or {}
    base = energy_match_score(prev_track, next_track)
    prev_energy = track_energy(prev_track)
    next_energy = track_energy(next_track)
    flow_target = settings.get("targetEnergy", "keep")
    delta = next_energy - prev_energy
    if flow_target == "up":
        flow_score = _clamp(65.0 + delta * 100.0, 0.0, 100.0)
    elif flow_target == "down":
        flow_score = _clamp(65.0 - delta * 100.0, 0.0, 100.0)
    elif flow_target == "arc":
        flow_score = _clamp(100.0 - abs(delta) * 140.0, 0.0, 100.0)
    else:
        flow_score = _clamp(100.0 - abs(delta) * 120.0, 0.0, 100.0)
    score = float(base.get("score", 55.0)) * 0.72 + flow_score * 0.28
    return {
        **base,
        "score": round(score, 1),
        "flowScore": round(flow_score, 1),
        "energyDelta": round(delta, 3),
        "reason": f"{base.get('reason', 'Compares energy profiles')} Also accounts for target energy flow.",
    }


def structure_compatibility(prev_track: dict[str, Any], next_track: dict[str, Any]) -> dict[str, Any]:
    transition = plan_transition(
        prev_track,
        next_track,
        {
            "crossfade": 64,
            "autoTransition": False,
            "aiPrecision": True,
            "phraseBars": 16,
        },
    ).to_dict()
    seconds = float(transition["seconds"])
    if seconds >= 24:
        score = 96.0
    elif seconds >= 16:
        score = 88.0
    elif seconds >= 8:
        score = 76.0
    elif seconds >= 4:
        score = 58.0
    else:
        score = 35.0
    confidence = float(transition.get("confidence") or 0.0)
    score = score * 0.82 + confidence * 100 * 0.18
    return {
        "score": round(_clamp(score, 0.0, 100.0), 1),
        "overlap_seconds": seconds,
        "phrase_bars": transition.get("phrase_bars", 0),
        "overlapSeconds": seconds,
        "phraseBars": transition.get("phrase_bars", 0),
        "planConfidence": round(confidence, 2),
        "transition": transition,
        "reason": "Scores whether the outro/intro can carry a phrase-length overlap.",
    }


def style_compatibility(prev_track: dict[str, Any], next_track: dict[str, Any]) -> dict[str, Any]:
    from .matching import style_match_score

    result = style_match_score(prev_track, next_track)
    return {**result, "score": round(float(result.get("score", 62.0)), 1)}


def cues_for_direction(track: dict[str, Any], direction: str) -> list[dict[str, Any]]:
    candidates = track.get("transition_candidates") or {}
    raw = candidates.get("cue_candidates") or []
    cues = []
    for item in raw:
        if not isinstance(item, dict) or not _is_number(item.get("time")):
            continue
        role = str(item.get("role") or "")
        cue_direction = direction_for_role(role)
        if cue_direction != direction:
            continue
        cues.append(
            {
                "time": round(float(item["time"]), 3),
                "role": role,
                "direction": cue_direction,
                "score": round(_float(item.get("score"), 50.0), 1),
                "source": item.get("source") or "cue_detector_v2",
                "metrics": item.get("components") or item.get("metrics") or {},
                "reasons": item.get("reasons") or [],
            }
        )
    if not cues:
        candidates_key = "outro" if direction == "out" else "intro"
        if _is_number(candidates.get(candidates_key)):
            cues.append(score_simple_cue(track, float(candidates[candidates_key]), direction, f"analysis_{candidates_key}"))
    return sorted(cues, key=lambda item: _float(item.get("score"), 0.0), reverse=True)


def score_simple_cue(track: dict[str, Any], time_value: float, direction: str, source: str) -> dict[str, Any]:
    phrase = alignment_score(track, time_value)
    vocal = 1.0 - curve_value(_vocal_curve(track), time_value, "density", track_vocal_density(track))
    groove = local_groove_score(track, time_value)
    room = duration_room_score(_float(track.get("duration"), 0.0), time_value, direction)
    score = (phrase * 0.30 + vocal * 0.26 + groove * 0.22 + room * 0.22) * 100
    return {
        "time": round(float(time_value), 3),
        "role": "mix_out" if direction == "out" else "mix_in",
        "direction": direction,
        "score": round(_clamp(score, 0.0, 100.0), 1),
        "source": source,
        "metrics": {
            "phraseAlignment": round(phrase, 3),
            "vocalSafety": round(vocal, 3),
            "grooveStability": round(groove, 3),
            "durationRoom": round(room, 3),
        },
    }


def fallback_cue(track: dict[str, Any], direction: str) -> dict[str, Any]:
    duration = _float(track.get("duration"), 0.0)
    time_value = min(duration * 0.12, 12.0) if direction == "in" else max(0.0, duration - 24.0)
    return score_simple_cue(track, time_value, direction, "fallback")


def public_cue(cue: dict[str, Any]) -> dict[str, Any]:
    return {
        "time": cue.get("time"),
        "role": cue.get("role"),
        "score": cue.get("score"),
        "source": cue.get("source"),
        "reasons": cue.get("reasons") or [],
        "metrics": cue.get("metrics") or {},
    }


def rank_mixability(score: float, components: dict[str, dict[str, Any]] | None = None) -> str:
    components = components or {}
    harmonic = components.get("harmonic") or {}
    if harmonic.get("relation") in {"clash", "jaws_mix"}:
        return "usable" if score >= 60 else "avoid"
    if score >= 90:
        return "perfect"
    if score >= 75:
        return "recommended"
    if score >= 60:
        return "usable"
    return "avoid"


def track_camelot(track: dict[str, Any]) -> str | None:
    from .matching import key_label_to_camelot, parse_camelot

    camelot = track.get("camelot")
    if camelot is not None:
        parsed = parse_camelot(camelot)
        return f"{parsed[0]}{parsed[1]}" if parsed else None
    return key_label_to_camelot(track.get("key"), track.get("mode"))


def beat_grid_confidence(track: dict[str, Any]) -> float:
    bars = [float(value) for value in track.get("bars") or [] if _is_number(value)]
    phrases = [float(value) for value in track.get("phrases") or [] if _is_number(value)]
    candidates = track.get("transition_candidates") or {}
    base = _float(candidates.get("confidence"), _float(track.get("beat_confidence"), 0.35))
    if len(bars) >= 12:
        diffs = np.diff(np.asarray(bars, dtype=float))
        median = float(np.median(diffs)) if diffs.size else 0.0
        jitter = float(np.median(np.abs(diffs - median)) / max(median, 1e-6)) if median else 0.5
        base += _clamp(0.28 - jitter * 1.6, -0.18, 0.28)
    else:
        base -= 0.15
    if len(phrases) >= 3:
        base += 0.08
    return _clamp(base, 0.15, 0.95)


def groove_quality(track: dict[str, Any]) -> float:
    curve = _energy_curve(track)
    values = [_float(item.get("energy"), 0.0) for item in curve if isinstance(item, dict)]
    if not values:
        return _clamp(track_energy(track), 0.25, 0.75)
    mean = float(np.mean(values))
    stability = 1.0 - min(1.0, float(np.std(values)) * 2.2)
    grid = beat_grid_confidence(track)
    return _clamp(mean * 0.45 + stability * 0.25 + grid * 0.30, 0.1, 0.95)


def track_vocal_density(track: dict[str, Any]) -> float:
    candidates = track.get("transition_candidates") or {}
    values = []
    for key in ("intro_vocal_density", "outro_vocal_density"):
        if _is_number(candidates.get(key)):
            values.append(float(candidates[key]))
    values.extend(_float(item.get("density"), 0.5) for item in _vocal_curve(track) if isinstance(item, dict))
    return _clamp(float(np.mean(values)) if values else 0.5, 0.0, 1.0)


def track_energy(track: dict[str, Any]) -> float:
    profile = track.get("energy_profile") or {}
    return _clamp(_float(profile.get("energy_index"), _float(track.get("energy"), 0.5) * 100) / 100, 0.0, 1.0)


def low_frequency_ratio(track: dict[str, Any]) -> float:
    profile = track.get("energy_profile") or {}
    return _clamp(_float(profile.get("low_frequency_ratio"), 0.32), 0.0, 1.0)


def alignment_score(track: dict[str, Any], time_value: float) -> float:
    anchors = _anchors(track)
    if not anchors:
        return 0.45
    nearest = min(abs(anchor - time_value) for anchor in anchors)
    bpm = _float(track.get("bpm"), 120.0)
    beat = 60 / max(bpm, 1.0)
    return _clamp(1.0 - nearest / max(beat * 2, 0.25), 0.0, 1.0)


def local_groove_score(track: dict[str, Any], time_value: float) -> float:
    grid = beat_grid_confidence(track)
    energy = curve_value(_energy_curve(track), time_value, "energy", track_energy(track))
    return _clamp(grid * 0.62 + energy * 0.38, 0.0, 1.0)


def duration_room_score(duration: float, time_value: float, direction: str) -> float:
    if direction == "in":
        return _clamp(time_value / 24.0, 0.0, 1.0)
    if duration <= 0:
        return 0.0
    position = time_value / duration
    late_enough = _clamp((position - 0.48) / 0.26, 0.0, 1.0)
    tail_room = _clamp((duration - time_value) / 24.0, 0.0, 1.0)
    return _clamp(late_enough * 0.72 + tail_room * 0.28, 0.0, 1.0)


def curve_value(curve: Any, time_value: float | None, key: str, fallback: float) -> float:
    if not isinstance(curve, list) or not curve or time_value is None:
        return _clamp(fallback, 0.0, 1.0)
    points = [item for item in curve if isinstance(item, dict) and _is_number(item.get("time")) and _is_number(item.get(key))]
    if not points:
        return _clamp(fallback, 0.0, 1.0)
    nearest = min(points, key=lambda item: abs(float(item["time"]) - float(time_value)))
    return _clamp(_float(nearest.get(key), fallback), 0.0, 1.0)


def direction_for_role(role: str) -> str | None:
    if role in {"mix_in", "drop", "drum_loop"}:
        return "in"
    if role in {"mix_out", "bridge", "vocal_safe"}:
        return "out"
    return None


def _adjust_score(raw_score: float, components: dict[str, dict[str, Any]]) -> float:
    score = float(raw_score)
    harmonic = components["harmonic"]
    if harmonic.get("relation") == "clash":
        score = min(score, 79.0)
    if harmonic.get("relation") == "unknown":
        score = min(score, 84.0)
    if harmonic.get("relation") == "energy_boost":
        score = min(score, 89.0)
    if harmonic.get("relation") in {"diagonal_mix", "mood_shifter"}:
        score = min(score, 84.0)
    if harmonic.get("relation") == "jaws_mix":
        score = min(score, 79.0)
    for key, ceiling in (("tempo", 84.0), ("structure", 84.0), ("vocal", 86.0)):
        if float(components[key].get("score") or 0) < 60:
            score = min(score, ceiling)
    if float(components["energy"].get("score") or 0) < 60:
        score = min(score, 89.0)
    if components["style"].get("relation") == "contrast":
        score = min(score, 86.0)
    return _clamp(score, 0.0, 100.0)


def _summary(score: float, components: dict[str, dict[str, Any]]) -> str:
    weak = [
        key
        for key in ("tempo", "cue", "rhythm", "vocal", "bass", "harmonic", "energy", "structure", "style")
        if float(components[key].get("score") or 0) < 65
    ]
    if not weak:
        return f"Strong handoff candidate ({score:.1f}) with no major weak component."
    return f"Handoff score {score:.1f}; watch {', '.join(weak[:3])}."


def _stem_vocal_risk(track: dict[str, Any], time_value: float | None) -> float:
    if time_value is None:
        return 0.0
    activity = track.get("vocal_activity") or (track.get("transition_candidates") or {}).get("vocal_activity") or {}
    regions = activity.get("regions") or track.get("vocal_regions") or []
    if not isinstance(regions, list) or not regions:
        return 0.0
    lookahead = 6.0
    active = 0.0
    for region in regions:
        if not isinstance(region, dict) or not _is_number(region.get("start")) or not _is_number(region.get("end")):
            continue
        start = float(region["start"])
        end = float(region["end"])
        active += max(0.0, min(time_value + lookahead, end) - max(time_value, start))
    return _clamp(active / lookahead, 0.0, 1.0)


def _vocal_curve(track: dict[str, Any]) -> list[Any]:
    candidates = track.get("transition_candidates") or {}
    return candidates.get("vocal_density_curve") or track.get("vocal_density_curve") or []


def _energy_curve(track: dict[str, Any]) -> list[Any]:
    candidates = track.get("transition_candidates") or {}
    return candidates.get("energy_curve") or track.get("energy_curve") or []


def _anchors(track: dict[str, Any]) -> list[float]:
    values = []
    for key in ("phrases", "bars", "beats"):
        values.extend(float(value) for value in track.get(key) or [] if _is_number(value))
    return sorted(set(round(value, 3) for value in values))


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
