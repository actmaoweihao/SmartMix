from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from .arrangement import (
    build_music_segments,
    choose_transition_type,
    score_segment_transition,
)
from .loudness import loudness_metrics, normalize_loudness
from .mixing import SAMPLE_RATE, _convert_to_mp3
from .storage import EXPORT_DIR, STEM_DIR


STEM_NAMES = ("vocals", "drums", "bass", "other")
LAYER_MODES = {"full_mix", "vocals", "drums", "bass", "other", "drums_bass_other", "instrumental", "vocal_over_instrumental"}
SEQUENTIAL_MODES = {"smooth_join", "hook_swap", "energy_build"}
LAYERED_MODES = {"a_vocal_b_instrumental", "b_vocal_a_instrumental"}


def analyze_mashup_tracks(track_a: dict[str, Any], track_b: dict[str, Any], bars_per_segment: int = 16, use_stems: bool = True) -> dict[str, Any]:
    audio_a = _load_mono(Path(track_a["path"]))
    audio_b = _load_mono(Path(track_b["path"]))
    return {
        "trackA": {"segments": build_music_segments(track_a, "A", bars_per_segment, audio=audio_a, sr=SAMPLE_RATE)},
        "trackB": {"segments": build_music_segments(track_b, "B", bars_per_segment, audio=audio_b, sr=SAMPLE_RATE)},
        "useStems": bool(use_stems),
    }


def build_mashup_plan(
    track_a: dict[str, Any],
    track_b: dict[str, Any],
    *,
    mode: str = "auto",
    target_duration_sec: float = 180.0,
    bars_per_segment: int = 16,
    use_stems: bool = True,
    transition_strictness: str = "balanced",
    stem_usage: str = "auto",
    vocal_priority: str = "auto",
    energy_curve: str = "smooth",
    return_alternatives: bool = True,
) -> dict[str, Any]:
    analysis = analyze_mashup_tracks(track_a, track_b, bars_per_segment, use_stems)
    segments_a = analysis["trackA"]["segments"]
    segments_b = analysis["trackB"]["segments"]
    if not segments_a or not segments_b:
        return {"plan": [], "score": 0, "warnings": ["Not enough segment data to build a mashup plan."], "analysis": analysis}

    context = {
        "useStems": use_stems and stem_usage != "force_full_mix",
        "transitionStrictness": transition_strictness,
        "stemUsage": stem_usage,
        "vocalPriority": vocal_priority,
        "energyCurve": energy_curve,
    }
    candidate_modes = ["smooth_join", "hook_swap", "a_vocal_b_instrumental", "b_vocal_a_instrumental", "energy_build"] if mode == "auto" else [mode]
    candidates = []
    for candidate_mode in candidate_modes:
        for sequence in _candidate_sequences(candidate_mode, segments_a, segments_b, target_duration_sec, context):
            plan, transition_reports = _sequence_to_plan(sequence, candidate_mode, target_duration_sec, context)
            quality = _quality_report(plan, transition_reports, candidate_mode, track_a, track_b, context)
            candidates.append(
                {
                    "plan": plan,
                    "score": quality["score"],
                    "warnings": quality["warnings"],
                    "mode": candidate_mode,
                    "qualityReport": quality,
                    "analysis": analysis,
                }
            )

    if not candidates:
        return {"plan": [], "score": 0, "warnings": ["No viable mashup arrangement found."], "analysis": analysis}
    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = candidates[0]
    alternatives = [
        {key: value for key, value in item.items() if key != "analysis"}
        for item in candidates[1:4]
    ] if return_alternatives else []
    selected["alternativePlans"] = alternatives
    return selected


def generate_mashup_plan_v2(*args, **kwargs) -> dict[str, Any]:
    return build_mashup_plan(*args, **kwargs)


def render_mashup_plan(
    plan: list[dict[str, Any]],
    tracks_by_id: dict[str, dict[str, Any]],
    *,
    fmt: str = "wav",
    target_lufs: float = -14.0,
    use_stems: bool = True,
) -> dict[str, Any]:
    return render_mashup_plan_v2(plan, tracks_by_id, fmt=fmt, target_lufs=target_lufs, use_stems=use_stems)


def render_mashup_plan_v2(
    plan: list[dict[str, Any]],
    tracks_by_id: dict[str, dict[str, Any]],
    *,
    fmt: str = "wav",
    target_lufs: float = -14.0,
    use_stems: bool = True,
) -> dict[str, Any]:
    if not plan:
        raise ValueError("Mashup plan is empty")
    rendered, render_warnings = _render_plan_audio(plan, tracks_by_id, use_stems=use_stems)
    rendered = normalize_loudness(rendered, SAMPLE_RATE, float(target_lufs))
    rendered = _peak_limit(rendered, ceiling=0.891)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = EXPORT_DIR / f"mashup_{uuid.uuid4().hex}.wav"
    sf.write(wav_path, rendered.T, SAMPLE_RATE, subtype="PCM_16")
    output_path = _convert_to_mp3(wav_path) if fmt == "mp3" else wav_path
    metrics = loudness_metrics(rendered, SAMPLE_RATE)
    report = {
        "filename": output_path.name,
        "items": len(plan),
        "duration": round(rendered.shape[1] / SAMPLE_RATE, 3),
        "format": fmt,
        "targetLufs": target_lufs,
        "loudness": metrics,
        "finalLufs": metrics["lufs"],
        "peak": round(float(np.max(np.abs(rendered)) + 1e-12), 6),
        "peakDb": metrics["peak_db"],
        "usedStems": bool(use_stems),
        "warnings": _dedupe(render_warnings),
        "containsInvalid": bool(not np.all(np.isfinite(rendered))),
    }
    return {"ok": True, "downloadUrl": f"/api/exports/{output_path.name}", "report": report}


def _candidate_sequences(mode: str, a: list[dict[str, Any]], b: list[dict[str, Any]], target_duration: float, context: dict[str, Any]) -> list[list[dict[str, Any]]]:
    if mode == "smooth_join":
        starts = _rank(a, {"intro_like", "verse_like", "breakdown_like"})[:3]
        middles = _rank(a, {"chorus_like", "drop_like", "pre_chorus_like", "verse_like"})[:3]
        endings = _rank(b, {"chorus_like", "drop_like", "outro_like"})[:4]
        return [[s, m, e] for s in starts for m in middles for e in endings if _unique([s, m, e])]
    if mode == "hook_swap":
        verses = _rank(a, {"verse_like", "intro_like", "pre_chorus_like"})[:3]
        hooks = _rank(b, {"chorus_like", "drop_like"})[:4]
        breaks = _rank(a, {"breakdown_like", "bridge_like", "verse_like"})[:3]
        outros = _rank(b, {"chorus_like", "drop_like", "outro_like"})[:3]
        return [[v, h, br, out] for v in verses for h in hooks for br in breaks for out in outros if _unique([v, h, br, out])]
    if mode == "energy_build":
        pool = sorted(a + b, key=lambda item: (float(item.get("energy", 0)), float(item.get("energyEnd", 0))))
        sequences = []
        for offset in range(min(3, len(pool))):
            seq = _ensure_both_sources(pool[offset : offset + 6], a, b)
            if len(seq) >= 2:
                sequences.append(seq)
        return sequences
    if mode == "a_vocal_b_instrumental":
        return _layered_sequences(a, b, "A", context)
    if mode == "b_vocal_a_instrumental":
        return _layered_sequences(b, a, "B", context)
    return _candidate_sequences("smooth_join", a, b, target_duration, context)


def _layered_sequences(vocal_segments: list[dict[str, Any]], bed_segments: list[dict[str, Any]], vocal_source: str, context: dict[str, Any]) -> list[list[dict[str, Any]]]:
    vocals = sorted(vocal_segments, key=lambda item: (float(item.get("vocalDensity", 0)), float(item.get("energy", 0))), reverse=True)[:4]
    beds = sorted(bed_segments, key=lambda item: (1.0 - float(item.get("vocalDensity", 0)), float(item.get("mixInScore", 0))), reverse=True)[:4]
    sequences = []
    for vocal in vocals:
        for bed in beds:
            vocal_item = {**vocal, "_preferredLayer": "vocals", "_layerRole": "vocal", "_vocalSource": vocal_source}
            bed_item = {**bed, "_preferredLayer": "instrumental", "_layerRole": "bed"}
            sequences.append([bed_item, vocal_item])
    return sequences


def _sequence_to_plan(sequence: list[dict[str, Any]], mode: str, target_duration: float, context: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if mode in LAYERED_MODES:
        return _layer_sequence_to_plan(sequence, mode, target_duration, context)
    plan = []
    reports = []
    cursor = 0.0
    previous = None
    for index, segment in enumerate(sequence):
        duration = min(_segment_duration(segment), max(8.0, target_duration - cursor + 8.0))
        transition_in = _none_transition()
        if previous:
            compatibility = score_segment_transition(previous, segment, mode, context)
            transition_in = compatibility["transitionSpec"]
            reports.append(_transition_report(previous, segment, compatibility))
            cursor = max(0.0, cursor - float(transition_in.get("durationSec", 0.0)))
        transition_out = _none_transition()
        item = _plan_item(segment, cursor, cursor + duration, "full_mix", transition_in, transition_out, compatibility if previous else None)
        if plan:
            plan[-1]["transitionOut"] = transition_in
        plan.append(item)
        cursor += duration
        previous = segment
        if cursor >= target_duration:
            break
    return plan, reports


def _layer_sequence_to_plan(sequence: list[dict[str, Any]], mode: str, target_duration: float, context: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(sequence) < 2:
        return [], []
    bed, vocal = sequence[0], sequence[1]
    duration = max(4.0, min(_segment_duration(bed), _segment_duration(vocal), target_duration))
    compatibility = score_segment_transition(vocal, bed, mode, context)
    transition = choose_transition_type(vocal, bed, compatibility, mode, bool(context.get("useStems", True)))
    compatibility["recommendedTransition"] = transition["type"]
    compatibility["transitionSpec"] = transition
    bed_item = _plan_item(bed, 0.0, duration, "instrumental", _none_transition(), transition, compatibility)
    vocal_item = _plan_item(vocal, 0.0, duration, "vocals", _none_transition(), transition, compatibility, gain_db=1.0)
    bed_item["stemPolicy"] = {"vocals": "mute", "drums": bed["source"], "bass": bed["source"], "other": bed["source"]}
    vocal_item["stemPolicy"] = {"vocals": vocal["source"], "drums": "mute", "bass": "mute", "other": "mute"}
    return [bed_item, vocal_item], [_transition_report(vocal, bed, compatibility)]


def _plan_item(
    segment: dict[str, Any],
    start: float,
    end: float,
    layer_mode: str,
    transition_in: dict[str, Any],
    transition_out: dict[str, Any],
    compatibility: dict[str, Any] | None,
    gain_db: float = 0.0,
) -> dict[str, Any]:
    bpm = float(segment.get("bpm") or 0)
    target_bpm = bpm
    stretch = 1.0
    if compatibility:
        stretch = float(compatibility.get("stretchRatio") or 1.0)
        target_bpm = bpm / stretch if stretch else bpm
    return {
        "id": f"item_{uuid.uuid4().hex[:8]}",
        "source": segment["source"],
        "trackId": segment["trackId"],
        "segmentId": segment["id"],
        "segmentLabel": segment.get("label"),
        "sourceStart": float(segment["start"]),
        "sourceEnd": float(segment["end"]),
        "timelineStart": round(float(start), 3),
        "timelineEnd": round(float(end), 3),
        "layerMode": layer_mode,
        "stemPolicy": {"vocals": segment["source"], "drums": segment["source"], "bass": segment["source"], "other": segment["source"]},
        "gainDb": gain_db,
        "targetBpm": round(float(target_bpm or 0), 3),
        "stretchRatio": round(float(stretch), 4),
        "pitchShiftSemitones": 0,
        "transitionIn": transition_in,
        "transitionOut": transition_out,
        "quality": {
            "score": compatibility["score"] if compatibility else 100,
            "warnings": compatibility["warnings"] if compatibility else [],
            "fixes": compatibility["fixes"] if compatibility else [],
        },
    }


def _quality_report(
    plan: list[dict[str, Any]],
    transition_reports: list[dict[str, Any]],
    mode: str,
    track_a: dict[str, Any],
    track_b: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    transition_score = float(np.mean([item["score"] for item in transition_reports])) if transition_reports else 72.0
    shape = _arrangement_shape_score(plan, mode)
    hook = 10.0 if any(item.get("segmentLabel") in {"chorus_like", "drop_like"} for item in plan) else -5.0
    both = 10.0 if {"A", "B"}.issubset({item.get("source") for item in plan}) else -25.0
    risks = sum(len(item.get("quality", {}).get("warnings", [])) for item in plan) * 2.5
    score = float(np.clip(transition_score * 0.58 + shape * 0.22 + hook + both - risks, 0, 100))
    warnings = _dedupe([warning for item in plan for warning in item.get("quality", {}).get("warnings", [])])
    strengths = []
    bpm_a = float(track_a.get("bpm") or 0)
    bpm_b = float(track_b.get("bpm") or 0)
    if bpm_a > 0 and bpm_b > 0:
        strengths.append(f"BPM delta {abs(bpm_a - bpm_b) / max(bpm_a, bpm_b) * 100:.1f}% considered in stretch plan.")
    strengths.append(f"Mode {mode} scored against phrase, energy, vocal, bass, and timbre constraints.")
    if any(item.get("layerMode") in {"vocals", "instrumental"} for item in plan):
        strengths.append("Stem-aware vocal/instrumental layering is planned.")
    summary = _summary_for_mode(mode)
    if transition_reports:
        summary += f" Main handoff uses {transition_reports[0]['type']}."
    return {
        "score": round(score, 1),
        "summary": summary,
        "strengths": strengths,
        "warnings": warnings,
        "transitionReports": transition_reports,
        "arrangementShapeScore": round(shape, 1),
        "riskPenalty": round(risks, 1),
    }


def score_plan(plan: list[dict[str, Any]], warnings: list[str] | None = None) -> float:
    if not plan:
        return 0.0
    transition_scores = [float(item.get("quality", {}).get("score", 72)) for item in plan[1:]]
    base = float(np.mean(transition_scores)) if transition_scores else 72.0
    source_bonus = 8.0 if {"A", "B"}.issubset({item.get("source") for item in plan}) else -20.0
    layer_bonus = 6.0 if any(item.get("layerMode") != "full_mix" for item in plan) else 0.0
    warning_penalty = min(18.0, 4.0 * len(warnings or []))
    return float(np.clip(base + source_bonus + layer_bonus - warning_penalty, 0, 100))


def _render_plan_audio(plan: list[dict[str, Any]], tracks_by_id: dict[str, dict[str, Any]], *, use_stems: bool) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []
    end_time = max(float(item.get("timelineEnd", 0)) for item in plan) + 3.0
    output = np.zeros((2, max(1, int(np.ceil(end_time * SAMPLE_RATE)))), dtype=np.float32)
    for item in sorted(plan, key=lambda value: float(value.get("timelineStart", 0))):
        track = tracks_by_id.get(str(item.get("trackId")))
        if not track:
            raise ValueError(f"Unknown track in mashup plan: {item.get('trackId')}")
        clip, clip_warnings = _load_plan_clip(item, track, use_stems=use_stems)
        warnings.extend(clip_warnings)
        clip = _tempo_align(clip, item)
        clip = _apply_transition_processing(clip, item)
        clip = _fit_to_timeline(clip, item)
        clip = _apply_envelope(clip, item)
        clip *= _db_to_gain(float(item.get("gainDb", 0.0)))
        if item.get("layerMode") == "vocals":
            clip = _polish_vocal(clip)
        if item.get("layerMode") == "instrumental":
            clip *= 0.92
        start_sample = max(0, int(round(float(item.get("timelineStart", 0)) * SAMPLE_RATE)))
        end_sample = min(output.shape[1], start_sample + clip.shape[1])
        if end_sample > start_sample:
            output[:, start_sample:end_sample] += clip[:, : end_sample - start_sample]
    return _peak_limit(output, ceiling=0.98), warnings


def _load_plan_clip(item: dict[str, Any], track: dict[str, Any], *, use_stems: bool) -> tuple[np.ndarray, list[str]]:
    mode = item.get("layerMode") if item.get("layerMode") in LAYER_MODES else "full_mix"
    track_id = str(item.get("trackId"))
    start = float(item.get("sourceStart", 0))
    end = float(item.get("sourceEnd", start))
    warnings: list[str] = []
    if mode != "full_mix" and use_stems:
        paths = _cached_stem_paths(track_id)
        if paths:
            if mode in STEM_NAMES:
                return _slice_audio(_load_stereo(paths[mode]), start, end), warnings
            if mode in {"instrumental", "drums_bass_other", "vocal_over_instrumental"}:
                parts = [_load_stereo(paths[name]) for name in ("drums", "bass", "other")]
                return _slice_audio(_sum_layers(parts), start, end), warnings
        warnings.append(f"Missing stems for {track.get('name') or track_id}; {mode} fell back to full_mix.")
    elif mode != "full_mix" and not use_stems:
        warnings.append(f"Stems disabled; {mode} fell back to full_mix.")
    return _slice_audio(_load_stereo(Path(track["path"])), start, end), warnings


def _tempo_align(clip: np.ndarray, item: dict[str, Any]) -> np.ndarray:
    ratio = float(item.get("stretchRatio") or 1.0)
    if 0.94 <= ratio <= 1.06 and abs(ratio - 1.0) >= 0.015 and clip.shape[1] > 64:
        return np.ascontiguousarray(librosa.effects.time_stretch(clip, rate=ratio), dtype=np.float32)
    return clip


def _fit_to_timeline(clip: np.ndarray, item: dict[str, Any]) -> np.ndarray:
    target = max(1, int(round((float(item.get("timelineEnd", 0)) - float(item.get("timelineStart", 0))) * SAMPLE_RATE)))
    if clip.shape[1] >= target:
        return clip[:, :target]
    return np.concatenate([clip, np.zeros((2, target - clip.shape[1]), dtype=np.float32)], axis=1)


def _apply_envelope(clip: np.ndarray, item: dict[str, Any]) -> np.ndarray:
    out = clip.copy()
    transition_in = _transition_dict(item.get("transitionIn"))
    transition_out = _transition_dict(item.get("transitionOut"))
    _apply_fade_in(out, transition_in)
    _apply_fade_out(out, transition_out)
    _micro_fade(out)
    return out


def _apply_transition_processing(clip: np.ndarray, item: dict[str, Any]) -> np.ndarray:
    out = clip.copy()
    transition_in = _transition_dict(item.get("transitionIn"))
    transition_out = _transition_dict(item.get("transitionOut"))
    for transition in (transition_in, transition_out):
        kind = transition.get("type")
        if kind == "filter_sweep":
            out = _filter_sweep(out, incoming=transition is transition_in)
        elif kind == "bass_swap":
            out = _bass_swap_filter(out, incoming=transition is transition_in)
        elif kind == "echo_out" and transition is transition_out:
            out = _append_echo_tail(out)
        elif kind == "reverb_tail" and transition is transition_out:
            out = _append_reverb_tail(out)
        elif kind == "breakdown_bridge" and transition is transition_out:
            out = _append_reverb_tail(_filter_sweep(out, incoming=False), wet=0.14)
    return out


def _apply_fade_in(buffer: np.ndarray, transition: dict[str, Any]) -> None:
    kind = transition.get("type", "none")
    if kind in {"none", "hard_cut"}:
        return
    samples = min(buffer.shape[1], int(float(transition.get("durationSec") or 0) * SAMPLE_RATE))
    if samples > 1:
        _, fade = _equal_power(samples)
        buffer[:, :samples] *= fade


def _apply_fade_out(buffer: np.ndarray, transition: dict[str, Any]) -> None:
    kind = transition.get("type", "none")
    if kind in {"none", "hard_cut"}:
        return
    samples = min(buffer.shape[1], int(float(transition.get("durationSec") or 0) * SAMPLE_RATE))
    if samples > 1:
        fade, _ = _equal_power(samples)
        buffer[:, -samples:] *= fade


def _filter_sweep(buffer: np.ndarray, incoming: bool) -> np.ndarray:
    if buffer.shape[1] < 128:
        return buffer
    chunks = max(1, int(math.ceil(buffer.shape[1] / (SAMPLE_RATE * 0.25))))
    rendered = np.zeros_like(buffer)
    for index in range(chunks):
        start = int(index * buffer.shape[1] / chunks)
        end = int((index + 1) * buffer.shape[1] / chunks)
        t = index / max(1, chunks - 1)
        if incoming:
            cutoff = 120 + 720 * t
            kind = "highpass"
        else:
            cutoff = 7000 - 5600 * t
            kind = "lowpass"
        rendered[:, start:end] = _sos_filter(buffer[:, start:end], kind, float(np.clip(cutoff, 80, 8000)))
    return rendered.astype(np.float32)


def _bass_swap_filter(buffer: np.ndarray, incoming: bool) -> np.ndarray:
    low = _sos_filter(buffer, "lowpass", 180)
    high = buffer - low
    x = np.linspace(0, 1, buffer.shape[1], dtype=np.float32)
    low_gain = np.sqrt(x) if incoming else np.power(1 - x, 1.8)
    return np.clip(high + low * low_gain, -1, 1).astype(np.float32)


def _append_echo_tail(buffer: np.ndarray, wet: float = 0.20) -> np.ndarray:
    tail_samples = min(buffer.shape[1], SAMPLE_RATE * 2)
    source = buffer[:, -tail_samples:]
    tail = np.zeros((2, SAMPLE_RATE * 2), dtype=np.float32)
    delays = [(0.22, 0.46), (0.44, 0.28), (0.66, 0.16), (0.88, 0.10)]
    for delay, gain in delays:
        start = int(delay * SAMPLE_RATE)
        length = min(source.shape[1], tail.shape[1] - start)
        if length > 0:
            tail[:, start : start + length] += source[:, :length] * gain * wet
    fade = np.linspace(1, 0, tail.shape[1], dtype=np.float32)
    tail *= fade
    return np.concatenate([buffer, tail], axis=1).astype(np.float32)


def _append_reverb_tail(buffer: np.ndarray, wet: float = 0.16) -> np.ndarray:
    tail_samples = min(buffer.shape[1], SAMPLE_RATE * 2)
    source = buffer[:, -tail_samples:]
    tail = np.zeros((2, SAMPLE_RATE * 2), dtype=np.float32)
    for delay_ms, gain in [(37, 0.24), (71, 0.18), (113, 0.14), (179, 0.10), (293, 0.08)]:
        start = int(delay_ms / 1000 * SAMPLE_RATE)
        length = min(source.shape[1], tail.shape[1] - start)
        if length > 0:
            tail[:, start : start + length] += source[:, :length] * gain * wet
    tail *= np.linspace(1, 0, tail.shape[1], dtype=np.float32)
    return np.concatenate([buffer, tail], axis=1).astype(np.float32)


def _polish_vocal(buffer: np.ndarray) -> np.ndarray:
    out = _sos_filter(buffer, "highpass", 95)
    return np.tanh(out * 1.08).astype(np.float32)


def _micro_fade(buffer: np.ndarray) -> None:
    samples = min(buffer.shape[1] // 2, int(0.012 * SAMPLE_RATE))
    if samples <= 1:
        return
    fade = np.linspace(0, 1, samples, dtype=np.float32)
    buffer[:, :samples] *= fade
    buffer[:, -samples:] *= fade[::-1]


def _transition_report(left: dict[str, Any], right: dict[str, Any], compatibility: dict[str, Any]) -> dict[str, Any]:
    return {
        "from": left.get("id"),
        "to": right.get("id"),
        "score": compatibility["score"],
        "type": compatibility["recommendedTransition"],
        "reason": _transition_reason(compatibility),
        "components": compatibility["components"],
        "warnings": compatibility["warnings"],
        "fixes": compatibility["fixes"],
    }


def _transition_reason(compatibility: dict[str, Any]) -> str:
    kind = compatibility.get("recommendedTransition")
    if kind == "bass_swap":
        return "BPM/key are usable, but both segments carry strong low end, so the low frequencies are exchanged."
    if kind == "echo_out":
        return "Vocal or hook tail is risky, so an echo tail masks the handoff."
    if kind == "filter_sweep":
        return "Timbre changes noticeably, so filtering masks the color shift."
    if kind == "hard_cut":
        return "Clean phrase/downbeat alignment allows a tight cut."
    if kind == "vocal_over_instrumental":
        return "Vocal density and instrumental bed support a layered mashup."
    return "Balanced phrase, energy, and vocal scores support a smooth crossfade."


def _arrangement_shape_score(plan: list[dict[str, Any]], mode: str) -> float:
    labels = [item.get("segmentLabel") for item in plan]
    if mode == "smooth_join":
        return 88.0 if labels and labels[-1] in {"outro_like", "chorus_like", "drop_like"} else 72.0
    if mode == "hook_swap":
        hooks = sum(1 for label in labels if label in {"chorus_like", "drop_like"})
        return 70.0 + min(20.0, hooks * 10.0)
    if mode in LAYERED_MODES:
        return 86.0 if any(item.get("layerMode") == "vocals" for item in plan) and any(item.get("layerMode") == "instrumental" for item in plan) else 60.0
    if mode == "energy_build":
        energies = [float(item.get("quality", {}).get("score", 0)) for item in plan]
        return 82.0 if len(energies) >= 2 else 62.0
    return 75.0


def _summary_for_mode(mode: str) -> str:
    return {
        "smooth_join": "This plan searches for a natural A-to-B song handoff.",
        "hook_swap": "This plan swaps hook-like chorus/drop sections between the two songs.",
        "a_vocal_b_instrumental": "This plan places A vocals over B instrumental material.",
        "b_vocal_a_instrumental": "This plan places B vocals over A instrumental material.",
        "energy_build": "This plan orders segments toward a stronger energy build.",
    }.get(mode, "This plan balances compatibility and arrangement shape.")


def _rank(segments: list[dict[str, Any]], labels: set[str]) -> list[dict[str, Any]]:
    pool = [item for item in segments if item.get("label") in labels] or segments
    return sorted(pool, key=lambda item: (float(item.get("mixInScore", 0)) + float(item.get("mixOutScore", 0)), float(item.get("energy", 0))), reverse=True)


def _ensure_both_sources(selected: list[dict[str, Any]], a: list[dict[str, Any]], b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = list(selected)
    sources = {item.get("source") for item in result}
    if "A" not in sources and a:
        result.insert(0, min(a, key=lambda item: float(item.get("energy", 0))))
    if "B" not in sources and b:
        result.append(max(b, key=lambda item: float(item.get("energy", 0))))
    return result


def _unique(items: list[dict[str, Any]]) -> bool:
    keys = [(item.get("source"), item.get("id")) for item in items]
    return len(keys) == len(set(keys))


def _segment_duration(segment: dict[str, Any]) -> float:
    return max(0.5, float(segment.get("end", 0)) - float(segment.get("start", 0)))


def _none_transition() -> dict[str, Any]:
    return {"type": "none", "durationSec": 0.0, "bars": 0, "automation": {}, "warnings": [], "fixes": []}


def _transition_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return {"type": value, "durationSec": 4.0 if value != "none" else 0.0, "automation": {}}
    return _none_transition()


def _equal_power(samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 1, samples, dtype=np.float32)
    return np.cos(x * np.pi / 2).astype(np.float32), np.sin(x * np.pi / 2).astype(np.float32)


def _sos_filter(buffer: np.ndarray, kind: str, freq: float) -> np.ndarray:
    try:
        sos = signal.butter(2, freq, kind, fs=SAMPLE_RATE, output="sos")
        return signal.sosfilt(sos, buffer, axis=1).astype(np.float32)
    except Exception:
        return buffer.astype(np.float32)


def _load_stereo(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=False)
    if y.ndim == 1:
        y = np.vstack([y, y])
    if y.shape[0] > 2:
        y = y[:2]
    return np.ascontiguousarray(y, dtype=np.float32)


def _load_mono(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return np.ascontiguousarray(y, dtype=np.float32)


def _slice_audio(audio: np.ndarray, start: float, end: float) -> np.ndarray:
    start_sample = max(0, int(start * SAMPLE_RATE))
    end_sample = max(start_sample + 1, min(audio.shape[1], int(end * SAMPLE_RATE)))
    return np.ascontiguousarray(audio[:, start_sample:end_sample], dtype=np.float32)


def _sum_layers(layers: list[np.ndarray]) -> np.ndarray:
    max_len = max(layer.shape[1] for layer in layers)
    mix = np.zeros((2, max_len), dtype=np.float32)
    for layer in layers:
        mix[:, : layer.shape[1]] += layer
    return _peak_limit(mix)


def _cached_stem_paths(track_id: str) -> dict[str, Path] | None:
    root = STEM_DIR / track_id / "demucs_api"
    stems = {name: root / f"{name}.wav" for name in STEM_NAMES}
    return stems if all(path.exists() for path in stems.values()) else None


def _peak_limit(buffer: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    out = np.nan_to_num(buffer, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > ceiling:
        out *= ceiling / peak
    return np.clip(out, -ceiling, ceiling).astype(np.float32)


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
