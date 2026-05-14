from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TransitionPlan:
    seconds: float
    phrase_bars: int
    available_seconds: float
    prev_outro: float
    next_intro: float
    prev_overlap_start: float
    next_overlap_start: float
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("seconds", "available_seconds", "prev_outro", "next_intro", "prev_overlap_start", "next_overlap_start", "confidence"):
            payload[key] = round(float(payload[key]), 3 if key != "confidence" else 2)
        return payload


def plan_transition(prev_track: dict[str, Any], next_track: dict[str, Any], settings: dict[str, Any] | None = None) -> TransitionPlan:
    settings = settings or {}
    requested = float(settings.get("crossfade", 8))
    auto = bool(settings.get("autoTransition", True))
    ai_precision = bool(settings.get("aiPrecision", False))
    phrase_bars = int(settings.get("phraseBars", 8))

    prev_duration = float(prev_track.get("duration") or 0)
    next_duration = float(next_track.get("duration") or 0)
    max_by_length = max(0.5, min(prev_duration, next_duration) * 0.35)

    prev_out = _candidate_outro(prev_track, prefer_analysis=ai_precision)
    next_in = _candidate_intro(next_track, prefer_analysis=ai_precision)
    tail = max(0.0, prev_duration - prev_out)
    head = max(0.0, next_in)
    available = max(0.0, min(tail, head, max_by_length))

    phrase_seconds = _phrase_transition_seconds(prev_track, next_track, phrase_bars) if ai_precision else None
    if phrase_seconds:
        requested = phrase_seconds

    if available > 0:
        requested = min(requested, available)
    if auto:
        structural = float(prev_track.get("outro_low") or 0) + float(next_track.get("intro_low") or 0) + 2
        requested = max(2.0, min(requested, structural))

    seconds = min(requested, max_by_length)
    prev_overlap_start = max(0.0, prev_out)
    next_overlap_start = max(0.0, next_in - seconds)
    phrase_fit = _largest_phrase_fit(prev_track, next_track, seconds)
    confidence = _plan_confidence(prev_track, next_track, available, seconds, phrase_fit)
    reason = "phrase-grid" if phrase_fit else "duration"
    if _low_vocal_overlap(prev_track, next_track):
        reason += "+low-vocal"
    if auto:
        reason += "+energy"
    return TransitionPlan(
        seconds=round(float(seconds), 3),
        phrase_bars=phrase_fit,
        available_seconds=round(float(available), 3),
        prev_outro=round(float(prev_out), 3),
        next_intro=round(float(next_in), 3),
        prev_overlap_start=round(float(prev_overlap_start), 3),
        next_overlap_start=round(float(next_overlap_start), 3),
        confidence=round(float(confidence), 2),
        reason=reason,
    )


def _candidate_intro(track: dict[str, Any], prefer_analysis: bool) -> float:
    candidates = track.get("transition_candidates") or {}
    value = candidates.get("intro") if prefer_analysis else None
    if value is None:
        value = track.get("introPoint", track.get("intro_low", 0))
    return float(value or 0)


def _candidate_outro(track: dict[str, Any], prefer_analysis: bool) -> float:
    candidates = track.get("transition_candidates") or {}
    duration = float(track.get("duration") or 0)
    value = candidates.get("outro") if prefer_analysis else None
    if value is None:
        value = track.get("outroPoint")
    if value is None:
        value = duration - float(track.get("outro_low") or 0)
    return float(value or 0)


def _phrase_transition_seconds(prev_track: dict[str, Any], next_track: dict[str, Any], phrase_bars: int) -> float | None:
    bpm = _avg_bpm(prev_track, next_track)
    if not bpm:
        return None
    return max(2.0, phrase_bars * 4 * (60 / bpm))


def _largest_phrase_fit(prev_track: dict[str, Any], next_track: dict[str, Any], seconds: float) -> int:
    bpm = _avg_bpm(prev_track, next_track)
    if not bpm:
        return 0
    for bars in (16, 8, 4):
        phrase_seconds = bars * 4 * (60 / bpm)
        if phrase_seconds <= seconds + 0.25:
            return bars
    return 0


def _avg_bpm(prev_track: dict[str, Any], next_track: dict[str, Any]) -> float | None:
    bpms = [float(track.get("bpm") or 0) for track in (prev_track, next_track)]
    bpms = [bpm for bpm in bpms if bpm > 0]
    if not bpms:
        return None
    return float(np.mean(bpms))


def _plan_confidence(
    prev_track: dict[str, Any],
    next_track: dict[str, Any],
    available: float,
    seconds: float,
    phrase_bars: int,
) -> float:
    score = 0.35
    if phrase_bars:
        score += 0.25
    if available >= seconds and seconds >= 4:
        score += 0.15
    prev_candidates = prev_track.get("transition_candidates") or {}
    next_candidates = next_track.get("transition_candidates") or {}
    score += min(0.2, float(prev_candidates.get("confidence") or 0) * 0.1 + float(next_candidates.get("confidence") or 0) * 0.1)
    score += _vocal_density_bonus(prev_candidates.get("outro_vocal_density"))
    score += _vocal_density_bonus(next_candidates.get("intro_vocal_density"))
    return min(0.95, score)


def _vocal_density_bonus(value: Any) -> float:
    if not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, min(0.06, (1.0 - float(value)) * 0.06))


def _low_vocal_overlap(prev_track: dict[str, Any], next_track: dict[str, Any]) -> bool:
    prev_candidates = prev_track.get("transition_candidates") or {}
    next_candidates = next_track.get("transition_candidates") or {}
    prev_density = prev_candidates.get("outro_vocal_density")
    next_density = next_candidates.get("intro_vocal_density")
    if not isinstance(prev_density, (int, float)) or not isinstance(next_density, (int, float)):
        return False
    return float(prev_density) <= 0.45 and float(next_density) <= 0.45
