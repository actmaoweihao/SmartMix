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
from .matching import camelot_key_distance
from .mixing import SAMPLE_RATE, _convert_to_mp3
from .storage import EXPORT_DIR, STEM_DIR


STEM_NAMES = ("vocals", "drums", "bass", "other")
LAYER_MODES = {"full_mix", "vocals", "drums", "bass", "other", "drums_bass_other", "instrumental", "vocal_over_instrumental"}
SEQUENTIAL_MODES = {"smooth_join", "hook_swap", "energy_build"}
LAYERED_MODES = {"a_vocal_b_instrumental", "b_vocal_a_instrumental"}
GROOVE_MODES = {"groove_vocal_handoff", "a_vocal_on_b_groove", "b_vocal_on_a_groove", "call_response_groove", "hook_exchange_groove"}

RENDER_REVIEW_FINDINGS = [
    "Before the layered renderer, Mashup render mostly consumed old segment items and could behave like full_mix append plus crossfade.",
    "Stem policies were present in plan items but not consistently transformed into independent vocals/drums/bass/other layer events.",
    "bass_swap and vocal duck fixes could be reported without enforcing per-stem bass/vocal gain envelopes in the timeline.",
    "transition automation lived partly in JSON fields and partly in item fades; there was no unified layer automation engine.",
    "time-stretch and pitch policy warnings were not attached to every rendered layer, making risky artifacts easy to miss.",
]


def analyze_mashup_tracks(track_a: dict[str, Any], track_b: dict[str, Any], bars_per_segment: int = 16, use_stems: bool = True) -> dict[str, Any]:
    audio_a = _load_mono(Path(track_a["path"]))
    audio_b = _load_mono(Path(track_b["path"]))
    segmentation_report = {}
    try:
        from .segmentation import analyze_track_segmentation

        stems_a = _cached_stem_paths(str(track_a.get("id"))) if use_stems else None
        stems_b = _cached_stem_paths(str(track_b.get("id"))) if use_stems else None
        segmentation_report = {
            "trackA": _compact_segmentation_report(analyze_track_segmentation({**track_a, "source": "A"}, "A", audio=audio_a, sr=SAMPLE_RATE, stems=stems_a)),
            "trackB": _compact_segmentation_report(analyze_track_segmentation({**track_b, "source": "B"}, "B", audio=audio_b, sr=SAMPLE_RATE, stems=stems_b)),
        }
    except Exception as exc:
        segmentation_report = {"warnings": [f"Segmentation report failed: {exc}"]}
    return {
        "trackA": {"segments": build_music_segments(track_a, "A", bars_per_segment, audio=audio_a, sr=SAMPLE_RATE)},
        "trackB": {"segments": build_music_segments(track_b, "B", bars_per_segment, audio=audio_b, sr=SAMPLE_RATE)},
        "segmentationReport": segmentation_report,
        "useStems": bool(use_stems),
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
    bed_preference: str = "auto",
    allow_hybrid_bed: bool = True,
    allow_vocal_pitch_shift: bool = False,
    max_vocal_stretch: float = 1.06,
    return_alternatives: bool = True,
) -> dict[str, Any]:
    if mode in GROOVE_MODES:
        from .groove_mashup import build_groove_mashup_plan

        return build_groove_mashup_plan(
            track_a,
            track_b,
            mode=mode,
            target_duration_sec=target_duration_sec,
            use_stems=use_stems,
            vocal_priority=vocal_priority,
            bed_preference=bed_preference,
            allow_hybrid_bed=allow_hybrid_bed,
            allow_vocal_pitch_shift=allow_vocal_pitch_shift,
            max_vocal_stretch=max_vocal_stretch,
            return_alternatives=return_alternatives,
        )
    groove_attempt = None
    if mode == "auto" and use_stems and stem_usage != "force_full_mix":
        from .groove_mashup import build_groove_mashup_plan

        groove_attempt = build_groove_mashup_plan(
            track_a,
            track_b,
            mode="groove_vocal_handoff",
            target_duration_sec=target_duration_sec,
            use_stems=use_stems,
            vocal_priority=vocal_priority,
            bed_preference=bed_preference,
            allow_hybrid_bed=allow_hybrid_bed,
            allow_vocal_pitch_shift=allow_vocal_pitch_shift,
            max_vocal_stretch=max_vocal_stretch,
            return_alternatives=return_alternatives,
        )
        if groove_attempt.get("groovePlan", {}).get("status") == "ok" and groove_attempt.get("plan"):
            groove_attempt["mode"] = "groove_vocal_handoff"
            groove_attempt["warnings"] = _dedupe(["Auto selected groove_vocal_handoff to avoid legacy full-mix crossfade.", *groove_attempt.get("warnings", [])])
            return groove_attempt
        if groove_attempt.get("groovePlan", {}).get("status") == "stems_required":
            groove_attempt["warnings"] = _dedupe([*groove_attempt.get("warnings", []), "Auto did not fall back to legacy crossfade because Groove 人声接力 needs stems for the requested sound."])
            groove_attempt["groovePlan"]["globalWarnings"] = groove_attempt["warnings"]
            return groove_attempt
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
            layered = build_mashup_render_plan(plan, candidate_mode, context, transition_reports)
            quality = _quality_report(plan, layered["transitions"], candidate_mode, track_a, track_b, context)
            layered["qualityReport"] = quality
            candidates.append(
                {
                    "plan": plan,
                    "items": plan,
                    "layers": layered["layers"],
                    "transitions": layered["transitions"],
                    "targetBpm": layered["targetBpm"],
                    "targetCamelot": layered["targetCamelot"],
                    "score": quality["score"],
                    "warnings": quality["warnings"],
                    "mode": candidate_mode,
                    "qualityReport": quality,
                    "analysis": analysis,
                }
            )

    if not candidates:
        warnings = ["No viable mashup arrangement found."]
        if groove_attempt:
            warnings.extend(groove_attempt.get("warnings", []))
        return {"plan": [], "score": 0, "warnings": _dedupe(warnings), "analysis": analysis}
    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = candidates[0]
    if groove_attempt and groove_attempt.get("warnings"):
        selected["warnings"] = _dedupe([*selected.get("warnings", []), *groove_attempt.get("warnings", [])])
        selected["qualityReport"]["warnings"] = _dedupe([*selected["qualityReport"].get("warnings", []), *groove_attempt.get("warnings", [])])
    alternatives = [
        {key: value for key, value in item.items() if key != "analysis"}
        for item in candidates[1:4]
    ] if return_alternatives else []
    selected["alternativePlans"] = alternatives
    return selected


def generate_mashup_plan_v2(*args, **kwargs) -> dict[str, Any]:
    return build_mashup_plan(*args, **kwargs)


def render_mashup_plan(
    plan: list[dict[str, Any]] | dict[str, Any],
    tracks_by_id: dict[str, dict[str, Any]],
    *,
    fmt: str = "wav",
    target_lufs: float = -14.0,
    use_stems: bool = True,
) -> dict[str, Any]:
    return render_mashup_plan_v2(plan, tracks_by_id, fmt=fmt, target_lufs=target_lufs, use_stems=use_stems)


def render_mashup_plan_v2(
    plan: list[dict[str, Any]] | dict[str, Any],
    tracks_by_id: dict[str, dict[str, Any]],
    *,
    fmt: str = "wav",
    target_lufs: float = -14.0,
    use_stems: bool = True,
) -> dict[str, Any]:
    if not plan:
        raise ValueError("Mashup plan is empty")
    render_plan = plan if isinstance(plan, dict) else {"items": plan, "plan": plan}
    if isinstance(render_plan, dict) and render_plan.get("groovePlan"):
        from .groove_mashup import render_groove_vocal_mashup

        rendered, groove_report = render_groove_vocal_mashup(render_plan["groovePlan"], tracks_by_id, SAMPLE_RATE, target_lufs=target_lufs)
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        wav_path = EXPORT_DIR / f"groove_mashup_{uuid.uuid4().hex}.wav"
        sf.write(wav_path, rendered.T, SAMPLE_RATE, subtype="PCM_16")
        output_path = _convert_to_mp3(wav_path) if fmt == "mp3" else wav_path
        metrics = loudness_metrics(rendered, SAMPLE_RATE)
        return {
            "ok": True,
            "downloadUrl": f"/api/exports/{output_path.name}",
            "report": {
                "filename": output_path.name,
                "format": fmt,
                "targetLufs": target_lufs,
                "finalLufs": metrics["lufs"],
                "peak": round(float(np.max(np.abs(rendered)) + 1e-12), 6),
                "peakDb": metrics["peak_db"],
                "usedStems": True,
                "warnings": groove_report.get("globalWarnings", []),
                "globalWarnings": groove_report.get("globalWarnings", []),
                "groovePlan": render_plan["groovePlan"],
                "renderStats": groove_report.get("renderStats", {}),
                "containsInvalid": bool(not np.all(np.isfinite(rendered))),
            },
        }
    if not render_plan.get("layers"):
        render_plan.update(build_mashup_render_plan(render_plan.get("items") or render_plan.get("plan") or [], "auto", {"useStems": use_stems}, []))
    rendered, render_warnings = render_layered_mashup(render_plan, tracks_by_id, sr=SAMPLE_RATE, use_stems=use_stems)
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
        "targetBpm": render_plan.get("targetBpm"),
        "targetCamelot": render_plan.get("targetCamelot"),
        "transitions": render_plan.get("transitions", []),
        "layers": _layer_report(render_plan.get("layers", [])),
        "globalWarnings": _dedupe([*render_plan.get("globalWarnings", []), *render_warnings]),
        "reviewFindings": RENDER_REVIEW_FINDINGS,
        "warnings": _dedupe(render_warnings),
        "containsInvalid": bool(not np.all(np.isfinite(rendered))),
        "renderStats": {
            "duration": round(rendered.shape[1] / SAMPLE_RATE, 3),
            "integratedLufs": metrics["lufs"],
            "peakDbfs": metrics["peak_db"],
            "clipped": bool(float(np.max(np.abs(rendered)) + 1e-12) > 0.999),
        },
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
            decision = choose_transition_type_v2(previous, segment, compatibility, mode, bool(context.get("useStems", True)))
            compatibility["recommendedTransition"] = decision["type"]
            compatibility["transitionSpec"] = _decision_to_transition_spec(decision, previous, segment)
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
    decision = choose_transition_type_v2(vocal, bed, compatibility, mode, bool(context.get("useStems", True)))
    transition = _decision_to_transition_spec(decision, vocal, bed)
    compatibility["recommendedTransition"] = decision["type"]
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
        "energy": float(segment.get("energy", 0.0)),
        "energyStart": float(segment.get("energyStart", segment.get("energy", 0.0))),
        "energyEnd": float(segment.get("energyEnd", segment.get("energy", 0.0))),
        "vocalDensity": float(segment.get("vocalDensity", 0.0)),
        "vocalStart": float(segment.get("vocalStart", segment.get("vocalDensity", 0.0))),
        "vocalEnd": float(segment.get("vocalEnd", segment.get("vocalDensity", 0.0))),
        "bassEnergy": float(segment.get("bassEnergy", 0.0)),
        "drumActivity": float(segment.get("drumActivity", 0.0)),
        "brightness": float(segment.get("brightness", 0.0)),
        "bpm": bpm,
        "camelot": segment.get("camelot"),
        "key": segment.get("key"),
        "isCleanEntry": bool(segment.get("isCleanEntry", True)),
        "isCleanExit": bool(segment.get("isCleanExit", True)),
        "barStart": int(segment.get("barStart", 0)),
        "barEnd": int(segment.get("barEnd", 0)),
        "downbeatTime": float(segment.get("downbeatTime", segment.get("start", 0.0))),
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
    stretch_penalty, stretch_warnings = _stretch_risk_penalty(plan, mode)
    risks = sum(len(item.get("quality", {}).get("warnings", [])) for item in plan) * 2.5 + stretch_penalty
    score = float(np.clip(transition_score * 0.58 + shape * 0.22 + hook + both - risks, 0, 100))
    if stretch_penalty >= 24 and mode in SEQUENTIAL_MODES:
        score = min(score, 42.0)
    warnings = _dedupe([warning for item in plan for warning in item.get("quality", {}).get("warnings", [])] + stretch_warnings)
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


def _stretch_risk_penalty(plan: list[dict[str, Any]], mode: str) -> tuple[float, list[str]]:
    penalty = 0.0
    warnings: list[str] = []
    unsafe = []
    audible = []
    vocal_risk = []
    for item in plan:
        ratio = float(item.get("stretchRatio") or 1.0)
        label = item.get("segmentId") or item.get("id") or item.get("source")
        if ratio < 0.88 or ratio > 1.12:
            unsafe.append(f"{label} stretchRatio {ratio:.3f}")
            penalty += 30.0
        elif ratio < 0.94 or ratio > 1.06:
            audible.append(f"{label} stretchRatio {ratio:.3f}")
            penalty += 12.0
        elif item.get("layerMode") == "vocals" and (ratio < 0.97 or ratio > 1.03):
            vocal_risk.append(f"{label} vocal stretchRatio {ratio:.3f}")
            penalty += 6.0
    if unsafe:
        warnings.append("Unsafe tempo mismatch for legacy segment handoff: " + "; ".join(unsafe[:3]) + ". Use Groove 人声接力 or pick closer-BPM tracks.")
    if audible:
        warnings.append("Audible tempo stretch in plan: " + "; ".join(audible[:3]) + ".")
    if vocal_risk:
        warnings.append("Vocal stretch is outside transparent range: " + "; ".join(vocal_risk[:3]) + ".")
    if mode in SEQUENTIAL_MODES and (unsafe or audible):
        warnings.append("Legacy smooth/hook/energy modes still behave like segment handoffs; Groove modes are recommended for bootleg-style vocal edits.")
    return penalty, warnings


def score_plan(plan: list[dict[str, Any]], warnings: list[str] | None = None) -> float:
    if not plan:
        return 0.0
    transition_scores = [float(item.get("quality", {}).get("score", 72)) for item in plan[1:]]
    base = float(np.mean(transition_scores)) if transition_scores else 72.0
    source_bonus = 8.0 if {"A", "B"}.issubset({item.get("source") for item in plan}) else -20.0
    layer_bonus = 6.0 if any(item.get("layerMode") != "full_mix" for item in plan) else 0.0
    warning_penalty = min(18.0, 4.0 * len(warnings or []))
    return float(np.clip(base + source_bonus + layer_bonus - warning_penalty, 0, 100))


def choose_transition_type_v2(
    seg_a: dict[str, Any],
    seg_b: dict[str, Any],
    compatibility: dict[str, Any],
    mode: str,
    use_stems: bool,
) -> dict[str, Any]:
    components = compatibility.get("components", {})
    bpm_ok = float(components.get("bpm", 0)) >= 82
    camelot_ok = float(components.get("camelot", 0)) >= 78
    phrase_ok = float(components.get("phrase", 0)) >= 78
    vocal_tail = float(seg_a.get("vocalEnd", seg_a.get("vocalDensity", 0))) >= 0.58
    incoming_vocal = float(seg_b.get("vocalStart", seg_b.get("vocalDensity", 0))) >= 0.52
    bass_conflict = float(seg_a.get("bassEnergy", 0)) >= 0.58 and float(seg_b.get("bassEnergy", 0)) >= 0.58
    timbre_gap = abs(float(seg_a.get("brightness", 0)) - float(seg_b.get("brightness", 0))) + abs(float(seg_a.get("bassEnergy", 0)) - float(seg_b.get("bassEnergy", 0)))

    if mode in LAYERED_MODES:
        if use_stems:
            return {
                "type": "vocal_over_instrumental",
                "confidence": 0.94,
                "reason": "Layered vocal/instrumental mode requires separate vocals and instrumental stems.",
                "requiredStems": True,
                "fallbackType": "equal_power_crossfade",
                "warnings": [],
            }
        return {
            "type": "equal_power_crossfade",
            "confidence": 0.46,
            "reason": "Cannot create clean vocal_over_instrumental because stems are missing. Falling back to full_mix crossfade.",
            "requiredStems": True,
            "fallbackType": "equal_power_crossfade",
            "warnings": ["stems required for clean vocal_over_instrumental"],
        }

    if seg_b.get("label") in {"drop_like", "chorus_like"} and seg_a.get("isCleanExit") and seg_b.get("isCleanEntry") and bpm_ok and camelot_ok and phrase_ok:
        if vocal_tail:
            return {"type": "vocal_drop", "confidence": 0.86, "reason": "Incoming hook/drop is aligned, but outgoing vocal must exit before impact.", "requiredStems": bool(use_stems), "fallbackType": "hard_cut", "warnings": []}
        return {"type": "hard_cut", "confidence": 0.9, "reason": "Clean downbeat and harmonic match supports a short hard cut into the hook/drop.", "requiredStems": False, "fallbackType": "equal_power_crossfade", "warnings": []}
    if vocal_tail and not incoming_vocal:
        return {"type": "echo_out", "confidence": 0.78, "reason": "Outgoing vocal tail would sound chopped; echo tail masks the exit.", "requiredStems": False, "fallbackType": "reverb_tail", "warnings": []}
    if bass_conflict:
        return {"type": "bass_swap", "confidence": 0.88, "reason": "Both segments have high bass energy; crossfade would cause low-end masking.", "requiredStems": bool(use_stems), "fallbackType": "filter_sweep", "warnings": []}
    if timbre_gap >= 0.55 or float(components.get("timbre", 100)) < 58:
        if compatibility.get("score", 100) < 58:
            return {"type": "breakdown_bridge", "confidence": 0.68, "reason": "Low compatibility plus large color change needs a short bridge.", "requiredStems": bool(use_stems), "fallbackType": "filter_sweep", "warnings": []}
        return {"type": "filter_sweep", "confidence": 0.76, "reason": "Large timbre gap; filter sweep masks the color shift.", "requiredStems": False, "fallbackType": "equal_power_crossfade", "warnings": []}
    if (seg_a.get("label") == "outro_like" or seg_b.get("label") == "intro_like") and not (vocal_tail and incoming_vocal):
        return {"type": "equal_power_crossfade", "confidence": 0.8, "reason": "Outro/intro handoff with low vocal conflict supports equal-power blending.", "requiredStems": False, "fallbackType": "equal_power_crossfade", "warnings": []}
    if compatibility.get("score", 100) < 52:
        return {"type": "breakdown_bridge", "confidence": 0.58, "reason": "Low direct transition score; bridge reduces energy/timbre shock.", "requiredStems": bool(use_stems), "fallbackType": "filter_sweep", "warnings": []}
    return {"type": "equal_power_crossfade", "confidence": 0.62, "reason": "No stronger transition rule matched; using equal-power crossfade as fallback.", "requiredStems": False, "fallbackType": "equal_power_crossfade", "warnings": []}


def _decision_to_transition_spec(decision: dict[str, Any], seg_a: dict[str, Any], seg_b: dict[str, Any]) -> dict[str, Any]:
    kind = decision["type"]
    bpm = float(seg_b.get("bpm") or seg_a.get("bpm") or 120)
    bar = 240.0 / max(bpm, 1.0)
    durations = {
        "hard_cut": 0.08,
        "vocal_drop": min(bar, 2.0),
        "echo_out": min(bar * 2, 4.0),
        "reverb_tail": min(bar * 2, 4.0),
        "bass_swap": min(bar * 4, 8.0),
        "filter_sweep": min(bar * 4, 8.0),
        "breakdown_bridge": min(bar * 4, 8.0),
        "vocal_over_instrumental": max(4.0, min(bar * 4, 8.0)),
        "equal_power_crossfade": min(bar * 2, 6.0),
    }
    spec = _transition_spec(kind, durations.get(kind, min(bar * 2, 6.0)), warnings=decision.get("warnings") or [], fixes=[])
    spec["reason"] = decision.get("reason", "")
    spec["confidence"] = decision.get("confidence", 0.0)
    spec["requiredStems"] = decision.get("requiredStems", False)
    spec["fallbackType"] = decision.get("fallbackType")
    return spec


def choose_target_bpm(segments: list[dict[str, Any]], mode: str = "auto") -> float:
    bpms = [float(item.get("bpm") or item.get("targetBpm") or 0) for item in segments if float(item.get("bpm") or item.get("targetBpm") or 0) > 0]
    if not bpms:
        return 120.0
    if mode == "a_vocal_b_instrumental":
        for item in segments:
            if item.get("source") == "A" or item.get("layerMode") == "vocals":
                return float(item.get("bpm") or np.median(bpms))
    if mode == "b_vocal_a_instrumental":
        for item in segments:
            if item.get("source") == "B" or item.get("layerMode") == "vocals":
                return float(item.get("bpm") or np.median(bpms))
    if mode == "energy_build":
        high = sorted(segments, key=lambda item: float(item.get("energy", 0)), reverse=True)[:2]
        high_bpms = [float(item.get("bpm") or 0) for item in high if float(item.get("bpm") or 0) > 0]
        if high_bpms:
            return float(np.median(high_bpms))
    median = float(np.median(bpms))
    safe = [bpm for bpm in bpms if 0.94 <= median / bpm <= 1.06]
    return float(np.median(safe or bpms))


def compute_stretch_policy(segment: dict[str, Any], target_bpm: float, stem: str) -> dict[str, Any]:
    source_bpm = float(segment.get("bpm") or segment.get("targetBpm") or target_bpm or 0)
    ratio = float(target_bpm / source_bpm) if source_bpm > 0 else 1.0
    warnings: list[str] = []
    high_risk = False
    if stem == "vocals":
        if not 0.94 <= ratio <= 1.06:
            warnings.append(f"Vocal layer {segment.get('segmentId')} stretchRatio {ratio:.3f} exceeds vocal-safe range.")
            high_risk = True
        elif not 0.97 <= ratio <= 1.03:
            warnings.append(f"Vocal layer {segment.get('segmentId')} stretchRatio {ratio:.3f} may be audible.")
    elif stem in {"drums", "bass", "other"}:
        if not 0.88 <= ratio <= 1.12:
            warnings.append(f"{stem} layer {segment.get('segmentId')} stretchRatio {ratio:.3f} exceeds render limits.")
            high_risk = True
        elif not 0.94 <= ratio <= 1.06:
            warnings.append(f"{stem} layer {segment.get('segmentId')} stretchRatio {ratio:.3f} may be audible.")
    elif not 0.94 <= ratio <= 1.06:
        warnings.append(f"full layer {segment.get('segmentId')} stretchRatio {ratio:.3f} may sound unnatural.")
        high_risk = not 0.88 <= ratio <= 1.12
    return {"stretchRatio": round(ratio, 4), "warnings": warnings, "highRisk": high_risk, "apply": bool(0.94 <= ratio <= 1.06)}


def compute_pitch_policy(vocal_segment: dict[str, Any], instrumental_segment: dict[str, Any], mode: str = "auto") -> dict[str, Any]:
    relation = camelot_key_distance(vocal_segment.get("camelot"), instrumental_segment.get("camelot")).get("relation")
    warnings: list[str] = []
    target = "none"
    semitones = 0
    high_risk = False
    if relation in {"same", "adjacent", "relative_major_minor"}:
        return {"target": target, "pitchShiftSemitones": 0, "warnings": [], "highRisk": False, "relation": relation}
    if mode in LAYERED_MODES:
        target = "instrumental"
        semitones = 2 if relation in {"energy_boost", "diagonal_mix", "mood_shifter", "jaws_mix"} else 3
        if abs(semitones) > 2:
            warnings.append("Instrumental pitch shift exceeds +/-2 semitones; high risk, not applied automatically.")
            high_risk = True
        else:
            warnings.append("Instrumental pitch shift suggested; vocal pitch is protected.")
    else:
        warnings.append("Camelot clash detected; full-mix pitch shift is not forced, prefer shorter/effected transition.")
    return {"target": target, "pitchShiftSemitones": semitones, "warnings": warnings, "highRisk": high_risk, "relation": relation}


def build_mashup_render_plan(
    items: list[dict[str, Any]],
    mode: str,
    context: dict[str, Any] | None = None,
    transition_reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    context = context or {}
    use_stems = bool(context.get("useStems", True))
    target_bpm = choose_target_bpm(items, mode)
    target_camelot = _choose_target_camelot(items)
    layers: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    global_warnings: list[str] = []
    for index, item in enumerate(items):
        layers.extend(_item_to_layers(item, target_bpm, mode, use_stems))
        if index > 0:
            previous = items[index - 1]
            transition = build_layered_transition(previous, item, mode, _transition_name(item.get("transitionIn")), use_stems, {"targetBpm": target_bpm})
            transition["score"] = float(item.get("quality", {}).get("score", 70))
            transitions.append(transition)
            global_warnings.extend(transition.get("warnings", []))
            layers.extend([layer for layer in transition.get("layerEvents", []) if layer.get("role") == "tail"])
    if mode in LAYERED_MODES and len(items) >= 2:
        pitch = compute_pitch_policy(next((item for item in items if item.get("layerMode") == "vocals"), items[-1]), next((item for item in items if item.get("layerMode") == "instrumental"), items[0]), mode)
        global_warnings.extend(pitch["warnings"])
    for layer in layers:
        global_warnings.extend(layer.get("warnings", []))
    return {
        "targetBpm": round(float(target_bpm), 3),
        "targetCamelot": target_camelot,
        "items": items,
        "plan": items,
        "transitions": transitions,
        "layers": layers,
        "globalWarnings": _dedupe(global_warnings),
        "qualityReport": {},
    }


def _choose_target_camelot(items: list[dict[str, Any]]) -> str | None:
    for item in sorted(items, key=lambda value: float(value.get("energy", 0)), reverse=True):
        if item.get("camelot"):
            return str(item.get("camelot"))
    return None


def _transition_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("type") or "none")
    if isinstance(value, str):
        return "equal_power_crossfade" if value == "crossfade" else value
    return "none"


def _item_to_layers(item: dict[str, Any], target_bpm: float, mode: str, use_stems: bool) -> list[dict[str, Any]]:
    layer_mode = item.get("layerMode") or "full_mix"
    if layer_mode == "vocals":
        return [_layer_event(item, "vocals", item["timelineStart"], item["timelineEnd"], "main_vocal", target_bpm, _item_automation(item, "vocals"))]
    if layer_mode in {"instrumental", "drums_bass_other", "vocal_over_instrumental"}:
        return [
            _layer_event(item, stem, item["timelineStart"], item["timelineEnd"], "instrumental", target_bpm, _item_automation(item, stem, duck=True))
            for stem in ("drums", "bass", "other")
        ]
    if use_stems and _item_needs_stem_layers(item):
        return [_layer_event(item, stem, item["timelineStart"], item["timelineEnd"], _stem_role(stem, item), target_bpm, _item_automation(item, stem)) for stem in STEM_NAMES]
    return [_layer_event(item, "full", item["timelineStart"], item["timelineEnd"], "bed", target_bpm, _item_automation(item, "full"))]


def _item_needs_stem_layers(item: dict[str, Any]) -> bool:
    transitions = [_transition_dict(item.get("transitionIn")), _transition_dict(item.get("transitionOut"))]
    return any(transition.get("type") in {"bass_swap", "vocal_drop"} or transition.get("automation", {}).get("vocal_duck") for transition in transitions)


def _stem_role(stem: str, item: dict[str, Any]) -> str:
    if stem == "vocals":
        return "main_vocal" if item.get("vocalDensity", 0) > 0.5 else "bed"
    if stem == "bass":
        return "bass_in" if _transition_name(item.get("transitionIn")) == "bass_swap" else "bass_out" if _transition_name(item.get("transitionOut")) == "bass_swap" else "bed"
    if stem == "drums":
        return "drums_bridge" if _transition_name(item.get("transitionIn")) == "drum_fill_bridge" else "bed"
    return "bed"


def _item_automation(item: dict[str, Any], stem: str, *, duck: bool = False) -> dict[str, Any]:
    duration = max(0.01, float(item.get("timelineEnd", 0)) - float(item.get("timelineStart", 0)))
    transition_in = _transition_dict(item.get("transitionIn"))
    transition_out = _transition_dict(item.get("transitionOut"))
    gain = [[0.0, 0.0], [1.0, 0.0]]
    low_gain = []
    highpass = []
    lowpass = []
    in_dur = min(1.0, float(transition_in.get("durationSec") or 0.0) / duration)
    out_dur = min(1.0, float(transition_out.get("durationSec") or 0.0) / duration)
    if transition_in.get("type") in {"equal_power_crossfade", "crossfade", "filter_sweep", "bass_swap", "echo_out", "reverb_tail", "breakdown_bridge"}:
        gain = [[0.0, -60.0], [in_dur, 0.0], [1.0, 0.0]]
    if transition_out.get("type") in {"equal_power_crossfade", "crossfade", "filter_sweep", "bass_swap", "echo_out", "reverb_tail", "breakdown_bridge"}:
        start = max(0.0, 1.0 - out_dur)
        gain = _merge_env(gain, [[start, 0.0], [1.0, -60.0]])
    if transition_in.get("type") == "hard_cut" or transition_out.get("type") == "hard_cut":
        gain = _merge_env(gain, [[0.0, 0.0], [1.0, 0.0]])
    if stem == "bass":
        if transition_in.get("type") == "bass_swap":
            gain = _merge_env(gain, [[0.0, -60.0], [min(1.0, in_dur * 0.45), -24.0], [max(0.001, in_dur * 0.60), 0.0]])
        if transition_out.get("type") == "bass_swap":
            start = max(0.0, 1.0 - out_dur)
            gain = _merge_env(gain, [[start, 0.0], [min(1.0, start + out_dur * 0.45), 0.0], [min(1.0, start + out_dur * 0.60), -24.0], [1.0, -60.0]])
    if stem == "vocals" and (transition_in.get("automation", {}).get("vocal_duck") or transition_out.get("automation", {}).get("vocal_duck")):
        gain = _merge_env(gain, [[0.0, -12.0], [1.0, -12.0]])
    if transition_in.get("type") == "filter_sweep":
        highpass = [[0.0, 1200.0], [max(in_dur, 0.01), 40.0]]
    if transition_out.get("type") in {"filter_sweep", "breakdown_bridge"}:
        lowpass = [[max(0.0, 1.0 - out_dur), 12000.0], [1.0, 850.0]]
    if transition_in.get("type") == "bass_swap" or transition_out.get("type") == "bass_swap":
        low_gain = [[0.0, 0.0], [0.5, -6.0], [1.0, 0.0]] if stem != "bass" else low_gain
    automation = {"gain": gain}
    if low_gain:
        automation["lowGainDb"] = low_gain
    if highpass:
        automation["highpassHz"] = highpass
    if lowpass:
        automation["lowpassHz"] = lowpass
    if duck:
        automation["duckDb"] = [[0.0, -2.5], [1.0, -2.5]]
    if stem == "vocals":
        automation["highpassHz"] = automation.get("highpassHz") or [[0.0, 95.0], [1.0, 95.0]]
    return automation


def _merge_env(*envs: list[list[float]]) -> list[list[float]]:
    points: dict[float, float] = {}
    for env in envs:
        for x, y in env:
            points[float(x)] = float(y)
    return [[x, points[x]] for x in sorted(points)]


def _layer_event(
    item: dict[str, Any],
    stem: str,
    timeline_start: Any,
    timeline_end: Any,
    role: str,
    target_bpm: float,
    automation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = compute_stretch_policy(item, target_bpm, "full" if stem == "full" else stem)
    pitch = float(item.get("pitchShiftSemitones") or 0.0)
    warnings = list(policy.get("warnings", []))
    if stem == "vocals" and abs(pitch) > 1:
        warnings.append("Vocals pitch shift exceeds +/-1 semitone; not applied automatically.")
    if abs(pitch) > 2:
        warnings.append("Pitch shift exceeds +/-2 semitones; high risk.")
    return {
        "id": f"layer_{uuid.uuid4().hex[:8]}",
        "trackId": item.get("trackId"),
        "source": item.get("source"),
        "segmentId": item.get("segmentId") or item.get("id"),
        "stem": stem,
        "sourceStart": float(item.get("sourceStart", item.get("start", 0.0))),
        "sourceEnd": float(item.get("sourceEnd", item.get("end", item.get("sourceStart", 0.0)))),
        "timelineStart": round(float(timeline_start), 3),
        "timelineEnd": round(float(timeline_end), 3),
        "gainDb": float(item.get("gainDb", 0.0)),
        "pan": float(item.get("pan", 0.0)),
        "stretchRatio": policy["stretchRatio"],
        "pitchShiftSemitones": pitch,
        "automation": automation or {"gain": [[0.0, 0.0], [1.0, 0.0]]},
        "role": role,
        "warnings": warnings,
    }


def _tail_layer(item: dict[str, Any], tail_type: str, start: float, duration: float, target_bpm: float) -> dict[str, Any]:
    source_end = float(item.get("sourceEnd", item.get("end", 0.0)))
    source_start = max(float(item.get("sourceStart", item.get("start", 0.0))), source_end - min(2.0, max(0.5, duration)))
    layer = _layer_event(
        {**item, "sourceStart": source_start, "sourceEnd": source_end},
        "full",
        start,
        start + min(3.0, duration + 1.5),
        "tail",
        target_bpm,
        {"gain": [[0.0, -9.0], [1.0, -60.0]], "highpassHz": [[0.0, 160.0], [1.0, 220.0]], "tailType": tail_type},
    )
    layer["tailType"] = tail_type
    return layer


def render_layered_mashup(
    plan: dict[str, Any],
    assets: dict[str, dict[str, Any]],
    sr: int = SAMPLE_RATE,
    *,
    use_stems: bool = True,
) -> tuple[np.ndarray, list[str]]:
    layers = list(plan.get("layers") or [])
    if not layers:
        return _render_plan_audio(plan.get("items") or plan.get("plan") or [], assets, use_stems=use_stems)
    warnings: list[str] = list(plan.get("globalWarnings") or [])
    end_time = max(float(layer.get("timelineEnd", 0)) for layer in layers) + 1.0
    output = np.zeros((2, max(1, int(np.ceil(end_time * sr)))), dtype=np.float32)
    for layer in sorted(layers, key=lambda item: float(item.get("timelineStart", 0))):
        audio, layer_warnings = _load_layer_audio(layer, assets, use_stems=use_stems)
        warnings.extend(layer_warnings)
        audio = _prepare_layer_audio(audio, layer, sr)
        audio = _fit_layer_audio(audio, layer, sr)
        if layer.get("tailType") == "echo":
            audio = apply_simple_delay_tail(audio, sr, bpm=float(plan.get("targetBpm") or 120), feedback=0.34, beats=0.5)
        elif layer.get("tailType") == "reverb":
            audio = apply_simple_reverb_tail(audio, sr, decay_sec=1.8)
        audio = apply_automation(audio, sr, layer.get("automation") or {}, layer.get("role", "bed"))
        audio = apply_microfade(audio, sr, fade_ms=10 if layer.get("role") != "tail" else 4)
        audio *= _db_to_gain(float(layer.get("gainDb", 0.0)))
        start_sample = max(0, int(round(float(layer.get("timelineStart", 0)) * sr)))
        end_sample = start_sample + audio.shape[1]
        if end_sample > output.shape[1]:
            output = np.pad(output, ((0, 0), (0, end_sample - output.shape[1])))
        output[:, start_sample:end_sample] += audio
    output = _dc_block(_peak_limit(output, ceiling=0.98))
    return output, _dedupe(warnings)


def _load_layer_audio(layer: dict[str, Any], assets: dict[str, dict[str, Any]], *, use_stems: bool) -> tuple[np.ndarray, list[str]]:
    track_id = str(layer.get("trackId"))
    track = assets.get(track_id)
    if not track:
        raise ValueError(f"Unknown track in mashup layer: {track_id}")
    stem = str(layer.get("stem") or "full")
    start = float(layer.get("sourceStart", 0.0))
    end = float(layer.get("sourceEnd", start))
    warnings = list(layer.get("warnings") or [])
    if stem != "full":
        paths = _cached_stem_paths(track_id) if use_stems else None
        if paths and stem in paths:
            return _slice_audio(_load_stereo(paths[stem]), start, end), warnings
        warnings.append(f"Missing {stem} stem for {track.get('name') or track_id}; layer fell back to full mix.")
        if layer.get("role") in {"main_vocal", "instrumental"}:
            warnings.append("Cannot create clean vocal_over_instrumental because stems are missing. Falling back to full_mix crossfade.")
    return _slice_audio(_load_stereo(Path(track["path"])), start, end), warnings


def _prepare_layer_audio(audio: np.ndarray, layer: dict[str, Any], sr: int) -> np.ndarray:
    out = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    ratio = float(layer.get("stretchRatio") or 1.0)
    stem = layer.get("stem")
    can_stretch = 0.94 <= ratio <= 1.06 and abs(ratio - 1.0) >= 0.015 and not (stem == "vocals" and not 0.97 <= ratio <= 1.03)
    if can_stretch and out.shape[1] > 64:
        out = np.ascontiguousarray(librosa.effects.time_stretch(out, rate=ratio), dtype=np.float32)
    pitch = float(layer.get("pitchShiftSemitones") or 0.0)
    can_pitch = abs(pitch) <= (1.0 if stem == "vocals" else 2.0) and abs(pitch) >= 0.01
    if can_pitch and out.shape[1] > 64:
        out = np.ascontiguousarray(librosa.effects.pitch_shift(out, sr=sr, n_steps=pitch), dtype=np.float32)
    if stem == "vocals":
        out = _polish_vocal(out)
    return _dc_block(out)


def _fit_layer_audio(audio: np.ndarray, layer: dict[str, Any], sr: int) -> np.ndarray:
    target = max(1, int(round((float(layer.get("timelineEnd", 0)) - float(layer.get("timelineStart", 0))) * sr)))
    if layer.get("role") == "tail":
        target = max(target, min(audio.shape[1] + int(2.0 * sr), int(3.0 * sr)))
    if audio.shape[1] >= target:
        return audio[:, :target]
    return np.pad(audio, ((0, 0), (0, target - audio.shape[1]))).astype(np.float32)


def apply_automation(audio: np.ndarray, sr: int, automation: dict[str, Any], role: str = "bed") -> np.ndarray:
    out = audio.copy().astype(np.float32)
    samples = out.shape[1]
    if samples <= 1:
        return out
    gain_curve = automation.get("gainCurve")
    if gain_curve == "equal_power_out":
        out *= make_equal_power_fade_out(samples)
    elif gain_curve == "equal_power_in":
        out *= make_equal_power_fade_in(samples)
    elif automation.get("gain"):
        out = apply_gain_envelope(out, _env_to_db_curve(automation["gain"], samples))
    if automation.get("duckDb"):
        out = apply_gain_envelope(out, _env_to_db_curve(automation["duckDb"], samples))
    if automation.get("lowGainDb"):
        out = _apply_low_gain(out, _env_to_db_curve(automation["lowGainDb"], samples))
    highpass_env = automation.get("highpassHz")
    lowpass_env = automation.get("lowpassHz")
    if highpass_env or lowpass_env:
        out = apply_simple_filter_sweep(out, sr, highpass_env=highpass_env, lowpass_env=lowpass_env)
    return _peak_limit(out, ceiling=0.98)


def make_equal_power_fade_out(n: int) -> np.ndarray:
    return np.cos(np.linspace(0, np.pi / 2, max(1, n), dtype=np.float32))[None, :]


def make_equal_power_fade_in(n: int) -> np.ndarray:
    return np.sin(np.linspace(0, np.pi / 2, max(1, n), dtype=np.float32))[None, :]


def apply_microfade(audio: np.ndarray, sr: int, fade_ms: float = 10) -> np.ndarray:
    out = audio.copy()
    samples = min(out.shape[1] // 2, max(1, int(sr * fade_ms / 1000)))
    if samples > 1:
        fade = np.linspace(0, 1, samples, dtype=np.float32)
        out[:, :samples] *= fade
        out[:, -samples:] *= fade[::-1]
    return out.astype(np.float32)


def apply_gain_envelope(audio: np.ndarray, envelope_db: np.ndarray) -> np.ndarray:
    return (audio * np.power(10.0, envelope_db[None, :] / 20.0)).astype(np.float32)


def apply_simple_filter_sweep(audio: np.ndarray, sr: int, highpass_env: list | None = None, lowpass_env: list | None = None) -> np.ndarray:
    if audio.shape[1] < 64:
        return audio.astype(np.float32)
    block = max(256, int(sr * 0.075))
    out = np.zeros_like(audio, dtype=np.float32)
    hp = _env_to_value_curve(highpass_env, audio.shape[1], default=0.0) if highpass_env else None
    lp = _env_to_value_curve(lowpass_env, audio.shape[1], default=sr / 2 - 100) if lowpass_env else None
    previous_tail = np.zeros((2, 0), dtype=np.float32)
    for start in range(0, audio.shape[1], block):
        end = min(audio.shape[1], start + block)
        chunk = audio[:, start:end]
        rendered = chunk
        if hp is not None:
            cutoff = float(np.clip(np.mean(hp[start:end]), 20, sr / 2 - 100))
            rendered = _sos_filter(rendered, "highpass", cutoff)
        if lp is not None:
            cutoff = float(np.clip(np.mean(lp[start:end]), 40, sr / 2 - 100))
            rendered = _sos_filter(rendered, "lowpass", cutoff)
        if previous_tail.shape[1]:
            fade_len = min(previous_tail.shape[1], rendered.shape[1], int(0.01 * sr))
            if fade_len > 1:
                fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
                fade_out = 1 - fade_in
                rendered[:, :fade_len] = previous_tail[:, -fade_len:] * fade_out + rendered[:, :fade_len] * fade_in
        out[:, start:end] = rendered[:, : end - start]
        previous_tail = rendered
    if lp is not None:
        out = _sos_filter(out, "lowpass", float(np.clip(np.min(lp), 40, sr / 2 - 100)))
    return out.astype(np.float32)


def apply_simple_delay_tail(audio_tail: np.ndarray, sr: int, bpm: float, feedback: float, beats: float) -> np.ndarray:
    source_len = min(audio_tail.shape[1], int(sr * 2.0))
    source = audio_tail[:, -source_len:]
    delay = max(1, int((60.0 / max(bpm, 1.0)) * beats * sr))
    tail_len = min(int(sr * 3.0), source_len + delay * 4)
    out = np.zeros((2, tail_len), dtype=np.float32)
    out[:, :source_len] += source * 0.45
    gain = float(np.clip(feedback, 0.1, 0.7))
    offset = delay
    while offset < tail_len and gain > 0.03:
        length = min(source_len, tail_len - offset)
        out[:, offset : offset + length] += source[:, :length] * gain
        gain *= feedback
        offset += delay
    out = _sos_filter(out, "highpass", 160)
    out = _sos_filter(out, "lowpass", 5500)
    out *= np.linspace(1, 0, tail_len, dtype=np.float32)
    return out.astype(np.float32)


def apply_simple_reverb_tail(audio_tail: np.ndarray, sr: int, decay_sec: float) -> np.ndarray:
    source_len = min(audio_tail.shape[1], int(sr * 1.5))
    source = audio_tail[:, -source_len:]
    tail_len = max(source_len, int(sr * decay_sec))
    out = np.zeros((2, tail_len), dtype=np.float32)
    for ms, gain in ((31, 0.26), (67, 0.2), (109, 0.16), (181, 0.11), (277, 0.08), (421, 0.05)):
        offset = int(ms / 1000 * sr)
        length = min(source_len, tail_len - offset)
        if length > 0:
            out[:, offset : offset + length] += source[:, :length] * gain
    out = _sos_filter(out, "highpass", 180)
    out *= np.linspace(1, 0, tail_len, dtype=np.float32) * _db_to_gain(-8)
    return out.astype(np.float32)


def sidechain_like_duck(instrumental: np.ndarray, vocal: np.ndarray, sr: int, amount_db: float = -2.5) -> np.ndarray:
    length = min(instrumental.shape[1], vocal.shape[1])
    if length <= 0:
        return instrumental
    out = instrumental.copy()
    frame = max(128, int(sr * 0.04))
    vocal_mono = np.mean(np.abs(vocal[:, :length]), axis=0)
    activity = np.convolve(vocal_mono, np.ones(frame, dtype=np.float32) / frame, mode="same")
    threshold = max(float(np.percentile(activity, 65)), 1e-5)
    duck = np.where(activity > threshold, _db_to_gain(amount_db), 1.0).astype(np.float32)
    smoothing = np.ones(max(8, int(sr * 0.03)), dtype=np.float32)
    smoothing /= np.sum(smoothing)
    duck = np.convolve(duck, smoothing, mode="same")
    out[:, :length] *= duck
    return out.astype(np.float32)


def _env_to_db_curve(points: list, samples: int) -> np.ndarray:
    return _env_to_value_curve(points, samples, default=0.0)


def _env_to_value_curve(points: list | None, samples: int, default: float = 0.0) -> np.ndarray:
    if not points:
        return np.full(samples, default, dtype=np.float32)
    xs = []
    ys = []
    for point in points:
        if len(point) < 2:
            continue
        x = float(point[0])
        xs.append(x if 0 <= x <= 1 else x / max(samples / SAMPLE_RATE, 1e-9))
        ys.append(float(point[1]))
    if not xs:
        return np.full(samples, default, dtype=np.float32)
    order = np.argsort(xs)
    xs = np.clip(np.asarray(xs, dtype=np.float32)[order], 0, 1)
    ys = np.asarray(ys, dtype=np.float32)[order]
    grid = np.linspace(0, 1, samples, dtype=np.float32)
    return np.interp(grid, xs, ys).astype(np.float32)


def _apply_low_gain(audio: np.ndarray, envelope_db: np.ndarray) -> np.ndarray:
    low = _sos_filter(audio, "lowpass", 180)
    high = audio - low
    return (high + low * np.power(10.0, envelope_db[None, :] / 20.0)).astype(np.float32)


def _dc_block(audio: np.ndarray) -> np.ndarray:
    return (audio - np.mean(audio, axis=1, keepdims=True)).astype(np.float32)


def _layer_report(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "stem": layer.get("stem"),
            "source": layer.get("source"),
            "segmentId": layer.get("segmentId"),
            "stretchRatio": layer.get("stretchRatio"),
            "pitchShiftSemitones": layer.get("pitchShiftSemitones"),
            "role": layer.get("role"),
            "warnings": layer.get("warnings", []),
        }
        for layer in layers
    ]


def build_layered_transition(
    seg_a: dict[str, Any],
    seg_b: dict[str, Any],
    mode: str,
    transition_type: str,
    use_stems: bool,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = context or {}
    start = float(seg_b.get("timelineStart", seg_a.get("timelineEnd", 0)))
    duration = max(0.02, min(float(seg_a.get("timelineEnd", start)) - start, float(seg_b.get("timelineEnd", start + 4)) - start))
    if transition_type in {"none", "crossfade"}:
        transition_type = "equal_power_crossfade"
    if transition_type == "hard_cut":
        duration = min(duration, 0.12)
    layer_events: list[dict[str, Any]] = []
    warnings: list[str] = []
    fixes: list[str] = []
    reason = ""
    target_bpm = float(context.get("targetBpm") or choose_target_bpm([seg_a, seg_b], mode))
    if transition_type == "vocal_over_instrumental":
        if use_stems:
            vocal = seg_a if seg_a.get("layerMode") == "vocals" else seg_b
            bed = seg_b if vocal is seg_a else seg_a
            layer_events.append(_layer_event(vocal, "vocals", vocal.get("timelineStart", 0), vocal.get("timelineEnd", duration), "main_vocal", target_bpm, {"gain": [[0, 0], [1, 0]], "highpassHz": [[0, 100], [1, 100]], "reverbSend": [[0, -22], [1, -22]]}))
            for stem in ("drums", "bass", "other"):
                layer_events.append(_layer_event(bed, stem, bed.get("timelineStart", 0), bed.get("timelineEnd", duration), "instrumental", target_bpm, {"gain": [[0, 0], [1, 0]], "duckDb": [[0, -2.5], [1, -2.5]]}))
            fixes.append(f"{vocal.get('source')} vocals layered over {bed.get('source')} drums/bass/other; instrumental vocals muted.")
            reason = "Clean vocal_over_instrumental uses vocals stem plus opposite instrumental stems."
        else:
            warnings.append("Cannot create clean vocal_over_instrumental because stems are missing. Falling back to full_mix crossfade.")
            transition_type = "equal_power_crossfade"
    if transition_type == "equal_power_crossfade":
        layer_events.append(_layer_event(seg_a, "full", start, start + duration, "bed", target_bpm, {"gain": [[0, 0], [1, -60]], "gainCurve": "equal_power_out"}))
        layer_events.append(_layer_event(seg_b, "full", start, start + duration, "bed", target_bpm, {"gain": [[0, -60], [1, 0]], "gainCurve": "equal_power_in"}))
        reason = reason or "Equal-power crossfade avoids the center dip and avoids linear fade loudness bumps."
    elif transition_type == "hard_cut":
        layer_events.append(_layer_event(seg_a, "full", max(0.0, start - 0.02), start, "bed", target_bpm, {"gain": [[0, 0], [1, -60]], "microfade": True}))
        layer_events.append(_layer_event(seg_b, "full", start, start + 0.02, "bed", target_bpm, {"gain": [[0, -60], [1, 0]], "microfade": True}))
        if not (seg_a.get("isCleanExit") and seg_b.get("isCleanEntry")):
            warnings.append("Hard cut is not fully clean at bar/downbeat boundary.")
        reason = "Downbeat hard cut with 5-20ms microfade, no long blend."
    elif transition_type == "vocal_drop":
        layer_events.append(_layer_event(seg_a, "vocals" if use_stems else "full", max(0, start - duration), start, "main_vocal", target_bpm, {"gain": [[0, 0], [0.72, 0], [1, -60]], "microfade": True}))
        layer_events.append(_layer_event(seg_b, "full", start, start + max(duration, 0.5), "bed", target_bpm, {"gain": [[0, -60], [0.08, 0], [1, 0]], "microfade": True}))
        fixes.append("vocal dropped before incoming drop")
        reason = "Outgoing dry vocal exits before incoming drop impact."
    elif transition_type == "bass_swap":
        if use_stems:
            for stem in ("drums", "other", "vocals"):
                layer_events.append(_layer_event(seg_a, stem, start, start + duration, "bed", target_bpm, {"gain": [[0, 0], [1, -8 if stem == "drums" else -60]], "gainCurve": "equal_power_out"}))
                layer_events.append(_layer_event(seg_b, stem, start, start + duration, "bed", target_bpm, {"gain": [[0, -8 if stem == "drums" else -60], [1, 0]], "gainCurve": "equal_power_in"}))
            layer_events.append(_layer_event(seg_a, "bass", start, start + duration, "bass_out", target_bpm, {"gain": [[0, 0], [0.45, 0], [0.6, -24], [1, -60]]}))
            layer_events.append(_layer_event(seg_b, "bass", start, start + duration, "bass_in", target_bpm, {"gain": [[0, -60], [0.45, -24], [0.6, 0], [1, 0]]}))
            fixes.append("bass conflict fixed by bass_swap")
            reason = "Both segments have high bass energy; outgoing bass exits before incoming bass reaches full level."
        else:
            layer_events.append(_layer_event(seg_a, "full", start, start + duration, "bed", target_bpm, {"gain": [[0, 0], [1, -60]], "lowGainDb": [[0, 0], [0.5, -12], [1, -24]], "gainCurve": "equal_power_out"}))
            layer_events.append(_layer_event(seg_b, "full", start, start + duration, "bed", target_bpm, {"gain": [[0, -60], [1, 0]], "lowGainDb": [[0, -24], [0.5, -12], [1, 0]], "gainCurve": "equal_power_in"}))
            warnings.append("No stems for bass_swap; using full-mix EQ simulation.")
            reason = "Bass conflict fixed with low-band EQ simulation."
    elif transition_type == "filter_sweep":
        layer_events.append(_layer_event(seg_a, "full", start, start + duration, "bed", target_bpm, {"gain": [[0, 0], [1, -60]], "lowpassHz": [[0, 12000], [1, 850]], "gainCurve": "equal_power_out"}))
        layer_events.append(_layer_event(seg_b, "full", start, start + duration, "bed", target_bpm, {"gain": [[0, -60], [1, 0]], "highpassHz": [[0, 1200], [1, 40]], "gainCurve": "equal_power_in"}))
        reason = "Filter automation masks large timbre differences."
    elif transition_type == "echo_out":
        layer_events.append(_layer_event(seg_a, "full", start, start + duration, "bed", target_bpm, {"gain": [[0, 0], [1, -60]], "gainCurve": "equal_power_out"}))
        layer_events.append(_tail_layer(seg_a, "echo", start, duration, target_bpm))
        layer_events.append(_layer_event(seg_b, "full", start, start + duration, "bed", target_bpm, {"gain": [[0, -60], [1, 0]], "gainCurve": "equal_power_in"}))
        fixes.append(f"echo tail duration {duration:.2f}s")
        reason = "Echo tail carries outgoing vocal/hook decay under the incoming entry."
    elif transition_type == "reverb_tail":
        layer_events.append(_layer_event(seg_a, "full", start, start + duration, "bed", target_bpm, {"gain": [[0, 0], [1, -60]], "gainCurve": "equal_power_out"}))
        layer_events.append(_tail_layer(seg_a, "reverb", start, duration, target_bpm))
        layer_events.append(_layer_event(seg_b, "full", start, start + duration, "bed", target_bpm, {"gain": [[0, -60], [1, 0]], "gainCurve": "equal_power_in"}))
        reason = "Low-level reverb tail softens the dry cut."
    elif transition_type in {"drum_fill_bridge", "breakdown_bridge"}:
        bridge_stem = "drums" if transition_type == "drum_fill_bridge" and use_stems else "full"
        if transition_type == "drum_fill_bridge" and not use_stems:
            warnings.append("drum_fill_bridge needs drums stem; using full-mix bridge fallback.")
        layer_events.append(_layer_event(seg_a, bridge_stem, start, start + duration, "drums_bridge", target_bpm, {"gain": [[0, 0], [0.82, -3], [1, -60]], "highpassHz": [[0, 80], [1, 250]], "microfade": True}))
        layer_events.append(_tail_layer(seg_a, "reverb", start, duration, target_bpm))
        layer_events.append(_layer_event(seg_b, "full", start + duration * 0.5, start + duration, "bed", target_bpm, {"gain": [[0, -60], [1, 0]], "highpassHz": [[0, 1000], [1, 40]]}))
        fixes.append("bridge uses filter automation, bass reduction, and tail/bed layer")
        reason = "Bridge reduces tonal and energy shock before the next section."
    return {
        "id": f"transition_{uuid.uuid4().hex[:8]}",
        "type": transition_type,
        "fromSegmentId": seg_a.get("segmentId") or seg_a.get("id"),
        "toSegmentId": seg_b.get("segmentId") or seg_b.get("id"),
        "startTime": round(start, 3),
        "durationSec": round(duration, 3),
        "bars": int(max(0, round(duration / max(240.0 / max(float(seg_b.get('bpm') or 120), 1.0), 1e-6)))),
        "layerEvents": layer_events,
        "warnings": warnings,
        "fixes": fixes,
        "reason": reason,
        "score": 70,
    }


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
        warnings.extend(_time_pitch_warnings(item))
        clip = _tempo_align(clip, item)
        clip = _fit_to_timeline(clip, item)
        clip = _basic_gain_match(clip, item)
        clip = _apply_transition_processing(clip, item)
        clip = _fit_to_timeline(clip, item, allow_tail=True)
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
    if use_stems:
        paths = _cached_stem_paths(track_id)
        if paths:
            if mode == "full_mix" and _needs_stem_aware_full_mix(item):
                return _stem_aware_full_mix(paths, item, start, end), warnings
            if mode in STEM_NAMES:
                return _slice_audio(_load_stereo(paths[mode]), start, end), warnings
            if mode in {"instrumental", "drums_bass_other", "vocal_over_instrumental"}:
                parts = [_load_stereo(paths[name]) for name in ("drums", "bass", "other")]
                return _slice_audio(_sum_layers(parts), start, end), warnings
        if mode != "full_mix" or _needs_stem_aware_full_mix(item):
            warnings.append(f"Missing stems for {track.get('name') or track_id}; {mode} fell back to full_mix.")
    elif mode != "full_mix" and not use_stems:
        warnings.append(f"Stems disabled; {mode} fell back to full_mix.")
    return _slice_audio(_load_stereo(Path(track["path"])), start, end), warnings


def _time_pitch_warnings(item: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    ratio = float(item.get("stretchRatio") or 1.0)
    if ratio < 0.88 or ratio > 1.12:
        warnings.append(f"{item.get('segmentId')} stretchRatio {ratio:.3f} exceeds safe render limits; left at source tempo.")
    elif ratio < 0.94 or ratio > 1.06:
        warnings.append(f"{item.get('segmentId')} stretchRatio {ratio:.3f} is audible; renderer avoids heavy time-stretch.")
    pitch = float(item.get("pitchShiftSemitones") or 0.0)
    if abs(pitch) > 2:
        warnings.append(f"{item.get('segmentId')} pitch shift {pitch:.1f} semitones is high risk and is not applied automatically.")
    elif abs(pitch) > 0.01:
        warnings.append(f"{item.get('segmentId')} pitch shift suggestion is reported but not rendered in this prototype.")
    return warnings


def _needs_stem_aware_full_mix(item: dict[str, Any]) -> bool:
    transitions = [_transition_dict(item.get("transitionIn")), _transition_dict(item.get("transitionOut"))]
    for transition in transitions:
        if transition.get("type") == "bass_swap":
            return True
        if transition.get("automation", {}).get("vocal_duck"):
            return True
    return False


def _stem_aware_full_mix(paths: dict[str, Path], item: dict[str, Any], start: float, end: float) -> np.ndarray:
    layers = {name: _slice_audio(_load_stereo(paths[name]), start, end) for name in STEM_NAMES}
    max_len = max(layer.shape[1] for layer in layers.values())
    mix = np.zeros((2, max_len), dtype=np.float32)
    for name, layer in layers.items():
        envelope = _stem_envelope(name, max_len, item)
        mix[:, : layer.shape[1]] += layer * envelope[: layer.shape[1]]
    return _peak_limit(mix)


def _stem_envelope(stem_name: str, samples: int, item: dict[str, Any]) -> np.ndarray:
    envelope = np.ones(samples, dtype=np.float32)
    transitions = (
        (_transition_dict(item.get("transitionIn")), True),
        (_transition_dict(item.get("transitionOut")), False),
    )
    for transition, incoming in transitions:
        duration = int(float(transition.get("durationSec") or 0.0) * SAMPLE_RATE)
        if duration <= 1:
            continue
        duration = min(duration, samples)
        if stem_name == "vocals" and transition.get("automation", {}).get("vocal_duck"):
            duck = np.full(duration, _db_to_gain(-12.0), dtype=np.float32)
            ramp = np.linspace(1.0, duck[0], duration, dtype=np.float32) if not incoming else np.linspace(duck[0], 1.0, duration, dtype=np.float32)
            if incoming:
                envelope[:duration] *= ramp
            else:
                envelope[-duration:] *= ramp
        if stem_name == "bass" and transition.get("type") == "bass_swap":
            if incoming:
                ramp = np.sin(np.linspace(0, np.pi / 2, duration, dtype=np.float32))
                envelope[:duration] *= ramp
            else:
                ramp = np.cos(np.linspace(0, np.pi / 2, duration, dtype=np.float32))
                envelope[-duration:] *= ramp
    return envelope


def _tempo_align(clip: np.ndarray, item: dict[str, Any]) -> np.ndarray:
    ratio = float(item.get("stretchRatio") or 1.0)
    if 0.94 <= ratio <= 1.06 and abs(ratio - 1.0) >= 0.015 and clip.shape[1] > 64:
        return np.ascontiguousarray(librosa.effects.time_stretch(clip, rate=ratio), dtype=np.float32)
    return clip


def _fit_to_timeline(clip: np.ndarray, item: dict[str, Any], *, allow_tail: bool = False) -> np.ndarray:
    target = max(1, int(round((float(item.get("timelineEnd", 0)) - float(item.get("timelineStart", 0))) * SAMPLE_RATE)))
    if allow_tail and _transition_dict(item.get("transitionOut")).get("type") in {"echo_out", "reverb_tail", "breakdown_bridge"}:
        target += int(2.0 * SAMPLE_RATE)
    if clip.shape[1] >= target:
        return clip[:, :target]
    return np.concatenate([clip, np.zeros((2, target - clip.shape[1]), dtype=np.float32)], axis=1)


def _basic_gain_match(clip: np.ndarray, item: dict[str, Any]) -> np.ndarray:
    if clip.shape[1] < 32:
        return clip
    rms = float(np.sqrt(np.mean(np.square(clip)) + 1e-12))
    target = 0.075 if item.get("layerMode") == "vocals" else 0.105
    gain = float(np.clip(target / max(rms, 1e-6), 0.5, 2.0))
    return (clip * gain).astype(np.float32)


def _apply_envelope(clip: np.ndarray, item: dict[str, Any]) -> np.ndarray:
    out = clip.copy()
    transition_in = _transition_dict(item.get("transitionIn"))
    transition_out = _transition_dict(item.get("transitionOut"))
    body_samples = max(1, int(round((float(item.get("timelineEnd", 0)) - float(item.get("timelineStart", 0))) * SAMPLE_RATE)))
    _apply_fade_in(out, transition_in)
    _apply_fade_out(out, transition_out, end_sample=min(body_samples, out.shape[1]))
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


def _apply_fade_out(buffer: np.ndarray, transition: dict[str, Any], *, end_sample: int | None = None) -> None:
    kind = transition.get("type", "none")
    if kind in {"none", "hard_cut"}:
        return
    boundary = int(np.clip(end_sample if end_sample is not None else buffer.shape[1], 0, buffer.shape[1]))
    samples = min(boundary, int(float(transition.get("durationSec") or 0) * SAMPLE_RATE))
    if samples > 1:
        fade, _ = _equal_power(samples)
        buffer[:, boundary - samples : boundary] *= fade


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


def _transition_spec(kind: str, duration: float, *, bars: int | None = None, warnings: list[str] | None = None, fixes: list[str] | None = None) -> dict[str, Any]:
    duration = max(0.02, float(duration))
    bars = int(bars if bars is not None else max(0, round(duration / 2.0)))
    automation = {"gain": [[0.0, 0.0], [1.0, 0.0]]}
    if kind in {"equal_power_crossfade", "crossfade"}:
        automation = {"a_gain": [[0, 0], [1, -60]], "b_gain": [[0, -60], [1, 0]], "curve": "equal_power"}
    elif kind == "bass_swap":
        automation = {
            "a_gain": [[0, 0], [1, -6]],
            "b_gain": [[0, -8], [1, 0]],
            "a_bass_gain": [[0, 0], [0.45, 0], [0.6, -24], [1, -60]],
            "b_bass_gain": [[0, -60], [0.45, -24], [0.6, 0], [1, 0]],
        }
    elif kind == "filter_sweep":
        automation = {"a_lowpassHz": [[0, 12000], [1, 850]], "b_highpassHz": [[0, 1200], [1, 40]]}
    elif kind in {"echo_out", "reverb_tail", "breakdown_bridge"}:
        automation = {"tail": kind, "tailWet": 0.22 if kind == "echo_out" else 0.16}
    elif kind in {"hard_cut", "vocal_drop"}:
        automation = {"microfade": True}
    return {"type": kind, "durationSec": round(duration, 3), "bars": bars, "automation": automation, "warnings": warnings or [], "fixes": fixes or []}


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
