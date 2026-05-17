from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from .loudness import loudness_metrics, normalize_loudness
from .storage import EXPORT_DIR
from .tuning import _demucs_available, _resolve_torch_device, _rubberband_command, _separate_stems_with_demucs_api


SAMPLE_RATE = 44100


DEFAULT_OPTIONS = {
    "targetMode": "quality",
    "useStemSeparation": False,
    "stemEngine": "demucs",
    "timeStretchEngine": "rubberband",
    "preserveFormants": True,
    "maxTempoChangePercent": 10,
    "maxPitchShiftSemitones": 2,
    "previewDurationBeforeTransition": 8,
    "previewDurationAfterTransition": 8,
    "exportFormat": "wav",
    "beginnerSafeMode": True,
    "device": "auto",
}


def generate_seamless_transition(
    outgoing_audio_path: Path,
    incoming_audio_path: Path,
    outgoing_analysis: dict[str, Any],
    incoming_analysis: dict[str, Any],
    recommendation: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    warnings: list[str] = []
    rec = recommendation or _fallback_recommendation(outgoing_analysis, incoming_analysis)
    alignment = align_transition_cues(outgoing_analysis, incoming_analysis, rec)
    alignment = _refine_alignment_for_smoothness(outgoing_analysis, incoming_analysis, alignment, rec)
    if not alignment["phraseAligned"]:
        warnings.append("Cue points could not both snap to 8/16-bar phrase boundaries; preview may feel less seamless.")

    tempo = compute_tempo_adjustment(float(outgoing_analysis.get("bpm") or 0), float(incoming_analysis.get("bpm") or 0), opts)
    if tempo["risk"] == "high":
        warnings.append("BPM difference is too large for transparent time-stretch; using short transition strategy instead of forcing beatmix.")
    pitch = compute_pitch_shift_for_harmonic_mixing(outgoing_analysis.get("camelot") or outgoing_analysis.get("key"), incoming_analysis.get("camelot") or incoming_analysis.get("key"), opts)
    if pitch["risk"] == "high" and pitch["semitones"] == 0:
        warnings.append("Key clash would require more than the allowed pitch shift; using transition masking instead of harmonic blend.")

    before = float(opts["previewDurationBeforeTransition"])
    after = float(opts["previewDurationAfterTransition"])
    overlap = _effective_overlap(float(alignment["overlapDuration"]), rec.get("method"))
    alignment["overlapDuration"] = overlap
    out_start = max(0.0, alignment["outgoingExitTime"] - before)
    in_start = max(0.0, alignment["incomingEntryTime"])
    outgoing_segment = _load_segment(outgoing_audio_path, out_start, before + overlap)
    incoming_segment = _load_segment(incoming_audio_path, in_start, overlap + after)

    with tempfile.TemporaryDirectory(prefix="smartmix_transition_") as temp:
        workspace = Path(temp)
        if tempo["shouldStretch"]:
            incoming_segment = _time_stretch_buffer(incoming_segment, tempo["stretchRatio"], workspace, opts, warnings)
        if pitch["shouldPitchShift"]:
            incoming_segment = _pitch_shift_buffer(incoming_segment, int(round(pitch["semitones"])), workspace, opts, warnings)

        vocal_conflict_before = _vocal_conflict(outgoing_analysis, incoming_analysis, alignment["outgoingExitTime"], alignment["incomingEntryTime"], overlap)
        stem_report = _maybe_separate_transition_stems(outgoing_segment, incoming_segment, workspace, opts, warnings)
        rendered = _render_transition_audio(outgoing_segment, incoming_segment, rec, overlap, vocal_conflict_before, stem_report)
        render_overlap = float(stem_report.get("renderOverlapDuration") or overlap)
        loudness = _match_loudness_report(outgoing_segment[:, : _seconds_to_samples(before)], incoming_segment[:, _seconds_to_samples(render_overlap) :])
        rendered = normalize_loudness(rendered, SAMPLE_RATE, -16)
        rendered = _limit_peak(rendered)

        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        wav_path = EXPORT_DIR / f"{uuid.uuid4().hex}_transition.wav"
        sf.write(wav_path, rendered.T, SAMPLE_RATE, subtype="PCM_16")
        output_path = wav_path
        if opts.get("exportFormat") == "mp3":
            output_path = _convert_to_mp3(wav_path)

    render_overlap = float(stem_report.get("renderOverlapDuration") or overlap)
    vocal_conflict_after = _estimate_after_conflict(vocal_conflict_before, {"method": stem_report.get("renderMethod") or rec.get("method")})
    report = {
        "bpmBefore": {"outgoing": outgoing_analysis.get("bpm"), "incoming": incoming_analysis.get("bpm")},
        "bpmAfter": {"outgoing": outgoing_analysis.get("bpm"), "incoming": tempo["targetBpm"] if tempo["shouldStretch"] else incoming_analysis.get("bpm")},
        "keyBefore": {"outgoing": outgoing_analysis.get("camelot") or outgoing_analysis.get("key"), "incoming": incoming_analysis.get("camelot") or incoming_analysis.get("key")},
        "keyAfter": {"outgoing": outgoing_analysis.get("camelot") or outgoing_analysis.get("key"), "incoming": pitch["targetKey"] if pitch["shouldPitchShift"] else incoming_analysis.get("camelot") or incoming_analysis.get("key")},
        "tempoChangePercent": tempo["tempoChangePercent"],
        "pitchShiftSemitones": pitch["semitones"] if pitch["shouldPitchShift"] else 0,
        "usedStemSeparation": stem_report["used"],
        "usedFormantPreservation": bool(opts.get("preserveFormants") and pitch["shouldPitchShift"]),
        "crossfadeCurve": _curve_for_method(stem_report.get("renderMethod") or rec.get("method")),
        "overlapDuration": render_overlap,
        "requestedOverlapDuration": overlap,
        "renderOverlapDuration": render_overlap,
        "vocalConflictBefore": vocal_conflict_before,
        "vocalConflictAfter": vocal_conflict_after,
        "loudnessMatchDb": loudness["loudnessDifferenceDb"],
        "transientShiftMs": round((stem_report.get("transientShiftSamples") or 0) / SAMPLE_RATE * 1000, 2),
        "incomingVocalDelayMs": round(((stem_report.get("vocalHandoff") or {}).get("incomingStartFraction") or 0) * render_overlap * 1000, 2),
        "incomingEnergyTrimDb": round(float((stem_report.get("energyHandoff") or {}).get("incomingTrimDb") or 0), 2),
        "incomingBandTrimDb": stem_report.get("frequencyHandoff") or {},
        "glueWet": round(float(stem_report.get("glueWet") or 0), 3),
        "rhythmBridgeWet": round(float(stem_report.get("rhythmBridgeWet") or 0), 3),
        "outgoingVocalGuarded": bool((stem_report.get("vocalHandoff") or {}).get("outgoingGuarded")),
        "outgoingNextVocalStartMs": round(((stem_report.get("vocalHandoff") or {}).get("outgoingNextVocalStartFraction") or 0) * render_overlap * 1000, 2),
        "outgoingVocalCutMs": round(((stem_report.get("vocalHandoff") or {}).get("outgoingEndFraction") or 0) * render_overlap * 1000, 2),
        "requestedMethod": rec.get("method", "beatmix"),
        "renderMethod": stem_report.get("renderMethod") or rec.get("method", "beatmix"),
        "strategyAdapted": bool(stem_report.get("renderMethod") and stem_report.get("renderMethod") != rec.get("method", "beatmix")),
        "cueRefinement": alignment.get("cueRefinement") or {},
        "riskScore": _risk_score(tempo, pitch, vocal_conflict_before, stem_report["used"]),
        "explanation": _processing_explanation(
            rec,
            tempo,
            pitch,
            stem_report["used"],
            vocal_conflict_before,
            loudness,
            stem_report.get("transientShiftSamples") or 0,
            stem_report.get("vocalHandoff") or {},
            stem_report.get("energyHandoff") or {},
            stem_report.get("frequencyHandoff") or {},
            stem_report.get("renderMethod") or rec.get("method", "beatmix"),
            render_overlap,
        ),
    }
    outgoing_cue = {**(rec.get("outgoingCue") or {})}
    incoming_cue = {**(rec.get("incomingCue") or {})}
    outgoing_cue["time"] = alignment["outgoingExitTime"]
    incoming_cue["time"] = alignment["incomingEntryTime"]
    outgoing_cue["adjusted"] = bool((alignment.get("cueRefinement") or {}).get("enabled"))
    incoming_cue["adjusted"] = bool((alignment.get("cueRefinement") or {}).get("enabled"))
    return {
        "audioPath": str(output_path),
        "url": f"/api/exports/{output_path.name}",
        "previewStartTime": out_start,
        "previewEndTime": out_start + rendered.shape[1] / SAMPLE_RATE,
        "outgoingCue": outgoing_cue,
        "incomingCue": incoming_cue,
        "requestedOutgoingCue": rec.get("outgoingCue"),
        "requestedIncomingCue": rec.get("incomingCue"),
        "alignment": alignment,
        "method": stem_report.get("renderMethod") or rec.get("method", "beatmix"),
        "processingReport": report,
        "warnings": warnings,
    }


def align_transition_cues(outgoing: dict[str, Any], incoming: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any]:
    outgoing_cue_time = float((rec.get("outgoingCue") or {}).get("time") or _candidate_time(outgoing, "outro"))
    incoming_cue_time = float((rec.get("incomingCue") or {}).get("time") or _candidate_time(incoming, "intro"))
    outgoing_snap = _snap_to_phrase(outgoing, outgoing_cue_time)
    incoming_snap = _snap_to_phrase(incoming, incoming_cue_time)
    phrase_aligned = outgoing_snap["phrase"] and incoming_snap["phrase"]
    drift = abs(outgoing_snap["time"] - outgoing_cue_time) + abs(incoming_snap["time"] - incoming_cue_time)
    return {
        "outgoingExitTime": round(outgoing_snap["time"], 3),
        "incomingEntryTime": round(incoming_snap["time"], 3),
        "outgoingDownbeatTime": round(outgoing_snap["downbeat"], 3),
        "incomingDownbeatTime": round(incoming_snap["downbeat"], 3),
        "overlapDuration": float(rec.get("overlapDuration") or 8),
        "phraseAligned": phrase_aligned,
        "alignmentConfidence": round(max(0.25, (0.9 if phrase_aligned else 0.62) - min(0.3, drift / 16)), 2),
    }


def _refine_alignment_for_smoothness(outgoing: dict[str, Any], incoming: dict[str, Any], alignment: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any]:
    method = rec.get("method", "beatmix")
    overlap = float(alignment.get("overlapDuration") or rec.get("overlapDuration") or 8)
    outgoing_refined = _best_local_cue(outgoing, float(alignment["outgoingExitTime"]), "outgoing", method, overlap)
    incoming_refined = _best_local_cue(incoming, float(alignment["incomingEntryTime"]), "incoming", method, overlap)
    before_conflict = _vocal_conflict(outgoing, incoming, float(alignment["outgoingExitTime"]), float(alignment["incomingEntryTime"]), overlap)
    after_conflict = _vocal_conflict(outgoing, incoming, outgoing_refined["time"], incoming_refined["time"], overlap)
    if after_conflict <= before_conflict + 0.04 or outgoing_refined["improved"] or incoming_refined["improved"]:
        refined = {**alignment}
        refined["outgoingExitTime"] = round(outgoing_refined["time"], 3)
        refined["incomingEntryTime"] = round(incoming_refined["time"], 3)
        refined["outgoingDownbeatTime"] = round(outgoing_refined["time"], 3)
        refined["incomingDownbeatTime"] = round(incoming_refined["time"], 3)
        refined["cueRefinement"] = {
            "enabled": bool(outgoing_refined["improved"] or incoming_refined["improved"]),
            "outgoingShiftSec": round(outgoing_refined["time"] - float(alignment["outgoingExitTime"]), 3),
            "incomingShiftSec": round(incoming_refined["time"] - float(alignment["incomingEntryTime"]), 3),
            "vocalConflictBefore": before_conflict,
            "vocalConflictAfter": after_conflict,
        }
        if refined["cueRefinement"]["enabled"]:
            refined["alignmentConfidence"] = round(min(0.95, float(refined["alignmentConfidence"]) + 0.05), 2)
        return refined
    return {**alignment, "cueRefinement": {"enabled": False, "vocalConflictBefore": before_conflict, "vocalConflictAfter": before_conflict}}


def _best_local_cue(track: dict[str, Any], cue_time: float, role: str, method: str, overlap: float) -> dict[str, Any]:
    bars = [float(value) for value in track.get("bars") or []]
    phrases = [float(value) for value in track.get("phrases") or []]
    anchors = sorted(set(bars + phrases))
    if not anchors:
        return {"time": cue_time, "improved": False}
    window_before = 24 if role == "outgoing" else 8
    window_after = 16 if role == "outgoing" else 24
    candidates = [value for value in anchors if cue_time - window_before <= value <= cue_time + window_after]
    if not candidates:
        return {"time": cue_time, "improved": False}
    original_score = _cue_smoothness_score(track, cue_time, cue_time, role, method, overlap)
    best = min(candidates, key=lambda value: _cue_smoothness_score(track, value, cue_time, role, method, overlap))
    best_score = _cue_smoothness_score(track, best, cue_time, role, method, overlap)
    return {"time": best, "improved": best_score + 0.04 < original_score and abs(best - cue_time) >= 0.25}


def _cue_smoothness_score(track: dict[str, Any], candidate: float, original: float, role: str, method: str, overlap: float) -> float:
    vocal_curve = track.get("vocal_density_curve") or []
    energy_curve = track.get("energy_curve") or []
    vocal = _curve_value(vocal_curve, candidate, "density")
    vocal_window = _curve_average(vocal_curve, candidate, max(2.0, overlap * (0.85 if role == "outgoing" else 0.55)), "density")
    vocal_peak = _curve_peak(vocal_curve, candidate, max(2.0, overlap * (0.9 if role == "outgoing" else 0.55)), "density")
    early_vocal = _curve_average(vocal_curve, candidate + 1.0, max(1.0, min(4.0, overlap * 0.35)), "density")
    energy = _curve_value(energy_curve, candidate, "energy")
    energy_window = _curve_average(energy_curve, candidate, max(2.0, overlap * 0.65), "energy")
    local_after = _curve_value(energy_curve, candidate + 4, "energy")
    distance_penalty = abs(candidate - original) / 30
    phrase_bonus = 0 if candidate in [float(v) for v in track.get("phrases") or []] else 0.05
    if role == "outgoing":
        energy_shape = max(0.0, local_after - energy) * 0.2
        next_phrase_penalty = max(0.0, vocal_peak - vocal) * 0.5
        return vocal * 0.2 + vocal_window * 0.28 + vocal_peak * 0.38 + energy_window * 0.08 + energy_shape + next_phrase_penalty + distance_penalty + phrase_bonus
    intro_penalty = 0.0 if method in {"beatmix", "bass_swap", "echo_out"} else 0.04
    return vocal * 0.22 + early_vocal * 0.42 + max(0.0, energy_window - 0.75) * 0.12 + distance_penalty + phrase_bonus + intro_penalty


def compute_tempo_adjustment(outgoing_bpm: float, incoming_bpm: float, options: dict[str, Any] | None = None) -> dict[str, Any]:
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    target = outgoing_bpm or incoming_bpm or 120
    compatible_incoming = min([incoming_bpm, incoming_bpm / 2, incoming_bpm * 2], key=lambda value: abs(value - target)) if incoming_bpm else target
    percent = abs(compatible_incoming - target) / max(target, 1) * 100
    max_percent = float(opts.get("maxTempoChangePercent", 10))
    if percent <= 6 and percent <= max_percent:
        return {"shouldStretch": percent > 0.25, "targetBpm": target, "stretchRatio": round(target / compatible_incoming, 5), "tempoChangePercent": round(percent, 3), "risk": "low"}
    if percent <= 10 and percent <= max_percent and opts.get("targetMode") == "quality":
        return {"shouldStretch": True, "targetBpm": target, "stretchRatio": round(target / compatible_incoming, 5), "tempoChangePercent": round(percent, 3), "risk": "medium"}
    return {"shouldStretch": False, "targetBpm": target, "stretchRatio": 1.0, "tempoChangePercent": round(percent, 3), "risk": "high"}


def compute_pitch_shift_for_harmonic_mixing(outgoing_key: str | None, incoming_key: str | None, options: dict[str, Any] | None = None) -> dict[str, Any]:
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    max_shift = int(opts.get("maxPitchShiftSemitones", 2))
    if not outgoing_key or not incoming_key or outgoing_key == incoming_key:
        return {"shouldPitchShift": False, "targetKey": incoming_key, "semitones": 0, "expectedCamelotRelation": "same" if outgoing_key == incoming_key else "unknown", "risk": "low"}
    semitones = _small_camelot_shift(outgoing_key, incoming_key)
    if semitones is None or abs(semitones) > max_shift:
        return {"shouldPitchShift": False, "targetKey": None, "semitones": 0, "expectedCamelotRelation": "clash", "risk": "high"}
    return {"shouldPitchShift": semitones != 0, "targetKey": outgoing_key, "semitones": semitones, "expectedCamelotRelation": "same", "risk": "low" if abs(semitones) <= 1 else "medium"}


def _render_transition_audio(outgoing: np.ndarray, incoming: np.ndarray, rec: dict[str, Any], overlap: float, vocal_conflict: float, stem_report: dict[str, Any]) -> np.ndarray:
    before_samples = max(0, outgoing.shape[1] - _seconds_to_samples(overlap))
    overlap_samples = min(_seconds_to_samples(overlap), outgoing.shape[1] - before_samples, incoming.shape[1])
    out_overlap = outgoing[:, before_samples : before_samples + overlap_samples]
    in_overlap = incoming[:, :overlap_samples]
    method = rec.get("method", "beatmix")

    if method == "echo_out":
        out_overlap = _apply_echo_tail(out_overlap, feedback=0.36, wet=0.42)
    if stem_report.get("used") and stem_report.get("paths"):
        overlap_mix, shift, vocal_handoff, energy_handoff, frequency_handoff, render_method, active_overlap_samples = _stem_overlap_from_paths(stem_report["paths"], before_samples, overlap_samples, method, vocal_conflict)
    else:
        overlap_mix, shift, vocal_handoff, energy_handoff, frequency_handoff, render_method, active_overlap_samples = _stem_like_overlap(out_overlap, in_overlap, method, vocal_conflict, False)
    active_overlap_samples = min(active_overlap_samples, overlap_samples, incoming.shape[1])
    outgoing_pre = outgoing[:, : max(0, outgoing.shape[1] - active_overlap_samples)]
    after = incoming[:, active_overlap_samples:]
    stem_report["transientShiftSamples"] = int(shift)
    stem_report["vocalHandoff"] = vocal_handoff
    stem_report["energyHandoff"] = energy_handoff
    stem_report["frequencyHandoff"] = frequency_handoff
    stem_report["renderMethod"] = render_method
    stem_report["renderOverlapDuration"] = active_overlap_samples / SAMPLE_RATE
    stem_report["glueWet"] = _glue_wet_for_method(render_method, vocal_conflict)
    stem_report["rhythmBridgeWet"] = _rhythm_bridge_wet_for_method(render_method, vocal_conflict)
    outgoing_pre, overlap_mix, after = _smooth_transition_edges(outgoing_pre, overlap_mix, after)
    return np.concatenate([outgoing_pre, overlap_mix, after], axis=1)


def _stem_like_overlap(outgoing: np.ndarray, incoming: np.ndarray, method: str, vocal_conflict: float, stems_used: bool) -> tuple[np.ndarray, int, dict[str, float], dict[str, float], dict[str, float], str, int]:
    samples = min(outgoing.shape[1], incoming.shape[1])
    outgoing = outgoing[:, :samples]
    incoming = incoming[:, :samples]
    out_low = _sos_filter(outgoing, "lowpass", 160)
    in_low = _sos_filter(incoming, "lowpass", 160)
    out_harmonic, out_percussive = _hpss(outgoing - out_low)
    in_harmonic, in_percussive = _hpss(incoming - in_low)
    shift = _estimate_transient_shift_samples(out_percussive, in_percussive)
    in_low = _shift_audio(in_low, shift)
    in_percussive = _shift_audio(in_percussive, shift)
    vocal_handoff = _vocal_handoff_timing(out_harmonic, in_harmonic, vocal_conflict, method)
    energy_handoff = _energy_handoff_profile(out_low + out_percussive + out_harmonic * 0.35, in_low + in_percussive + in_harmonic * 0.35, method)
    frequency_handoff = _frequency_handoff_profile(outgoing, incoming, method)
    render_method = _adapt_render_method(method, vocal_conflict, vocal_handoff, energy_handoff, frequency_handoff)
    active_samples = _active_overlap_samples(samples, render_method, method)
    if active_samples < samples:
        outgoing = outgoing[:, -active_samples:]
        incoming = incoming[:, :active_samples]
        samples = active_samples
        out_low = _sos_filter(outgoing, "lowpass", 160)
        in_low = _sos_filter(incoming, "lowpass", 160)
        out_harmonic, out_percussive = _hpss(outgoing - out_low)
        in_harmonic, in_percussive = _hpss(incoming - in_low)
        in_low = _shift_audio(in_low, shift)
        in_percussive = _shift_audio(in_percussive, shift)
    if render_method != method:
        vocal_handoff = _vocal_handoff_timing(out_harmonic, in_harmonic, vocal_conflict, render_method)
        energy_handoff = _energy_handoff_profile(out_low + out_percussive + out_harmonic * 0.35, in_low + in_percussive + in_harmonic * 0.35, render_method)
        frequency_handoff = _frequency_handoff_profile(outgoing, incoming, render_method)
    if render_method == "echo_out":
        out_harmonic = _apply_echo_tail(out_harmonic, feedback=0.28, wet=0.34)
    curves = _automation_curves(samples, render_method, vocal_conflict, vocal_handoff, energy_handoff, frequency_handoff)
    mix = (
        out_low * curves["out_bass"]
        + in_low * curves["in_bass"]
        + out_percussive * curves["out_drums"]
        + in_percussive * curves["in_drums"]
        + out_harmonic * curves["out_harmonic"]
        + in_harmonic * curves["in_harmonic"]
        + _transition_glue_layer(out_harmonic, in_harmonic, render_method, vocal_conflict)
        + _rhythm_bridge_layer(out_percussive, render_method, vocal_conflict)
    )
    if vocal_conflict > 0.45:
        mix *= 0.92
    return np.clip(mix, -1, 1).astype(np.float32), shift, vocal_handoff, energy_handoff, frequency_handoff, render_method, samples


def _stem_overlap_from_paths(stem_paths: dict[str, Any], before_samples: int, overlap_samples: int, method: str, vocal_conflict: float) -> tuple[np.ndarray, int, dict[str, float], dict[str, float], dict[str, float], str, int]:
    outgoing_stems = _load_stem_set(stem_paths.get("outgoing") or {}, before_samples + overlap_samples)
    incoming_stems = _load_stem_set(stem_paths.get("incoming") or {}, overlap_samples)
    out_slice = slice(before_samples, before_samples + overlap_samples)

    outgoing_vocals = outgoing_stems["vocals"][:, out_slice]
    outgoing_drums = outgoing_stems["drums"][:, out_slice]
    outgoing_bass = outgoing_stems["bass"][:, out_slice]
    outgoing_other = outgoing_stems["other"][:, out_slice]
    incoming_vocals = incoming_stems["vocals"][:, :overlap_samples]
    incoming_drums = incoming_stems["drums"][:, :overlap_samples]
    incoming_bass = incoming_stems["bass"][:, :overlap_samples]
    incoming_other = incoming_stems["other"][:, :overlap_samples]
    vocal_handoff = _vocal_handoff_timing(outgoing_vocals, incoming_vocals, vocal_conflict, method)
    shift = _estimate_transient_shift_samples(outgoing_drums, incoming_drums)
    incoming_drums = _shift_audio(incoming_drums, shift)
    incoming_bass = _shift_audio(incoming_bass, shift)
    incoming_other = _shift_audio(incoming_other, shift)
    energy_handoff = _energy_handoff_profile(outgoing_drums + outgoing_bass + outgoing_other * 0.5, incoming_drums + incoming_bass + incoming_other * 0.5, method)
    frequency_handoff = _frequency_handoff_profile(
        outgoing_drums + outgoing_bass + outgoing_other + outgoing_vocals * 0.35,
        incoming_drums + incoming_bass + incoming_other + incoming_vocals * 0.35,
        method,
    )
    render_method = _adapt_render_method(method, vocal_conflict, vocal_handoff, energy_handoff, frequency_handoff)
    active_samples = _active_overlap_samples(overlap_samples, render_method, method)
    if active_samples < overlap_samples:
        out_slice = slice(before_samples + overlap_samples - active_samples, before_samples + overlap_samples)
        outgoing_vocals = outgoing_stems["vocals"][:, out_slice]
        outgoing_drums = outgoing_stems["drums"][:, out_slice]
        outgoing_bass = outgoing_stems["bass"][:, out_slice]
        outgoing_other = outgoing_stems["other"][:, out_slice]
        incoming_vocals = incoming_stems["vocals"][:, :active_samples]
        incoming_drums = incoming_stems["drums"][:, :active_samples]
        incoming_bass = incoming_stems["bass"][:, :active_samples]
        incoming_other = incoming_stems["other"][:, :active_samples]
        incoming_drums = _shift_audio(incoming_drums, shift)
        incoming_bass = _shift_audio(incoming_bass, shift)
        incoming_other = _shift_audio(incoming_other, shift)
    if render_method != method:
        vocal_handoff = _vocal_handoff_timing(outgoing_vocals, incoming_vocals, vocal_conflict, render_method)
        energy_handoff = _energy_handoff_profile(outgoing_drums + outgoing_bass + outgoing_other * 0.5, incoming_drums + incoming_bass + incoming_other * 0.5, render_method)
        frequency_handoff = _frequency_handoff_profile(
            outgoing_drums + outgoing_bass + outgoing_other + outgoing_vocals * 0.35,
            incoming_drums + incoming_bass + incoming_other + incoming_vocals * 0.35,
            render_method,
        )
    curves = _automation_curves(active_samples, render_method, vocal_conflict, vocal_handoff, energy_handoff, frequency_handoff)
    if render_method == "echo_out":
        outgoing_vocals = _apply_echo_tail(outgoing_vocals, feedback=0.34, wet=0.42)
        outgoing_other = _apply_echo_tail(outgoing_other, feedback=0.24, wet=0.3)

    mix = (
        outgoing_drums * curves["out_drums"]
        + incoming_drums * curves["in_drums"]
        + outgoing_bass * curves["out_bass"]
        + incoming_bass * curves["in_bass"]
        + outgoing_other * curves["out_other"]
        + incoming_other * curves["in_other"]
        + outgoing_vocals * curves["out_vocals"]
        + incoming_vocals * curves["in_vocals"]
        + _transition_glue_layer(outgoing_other + outgoing_vocals * 0.12, incoming_other, render_method, vocal_conflict)
        + _rhythm_bridge_layer(outgoing_drums, render_method, vocal_conflict)
    )
    return np.clip(mix, -1, 1).astype(np.float32), shift, vocal_handoff, energy_handoff, frequency_handoff, render_method, active_samples


def _load_stem_set(paths: dict[str, Any], samples: int) -> dict[str, np.ndarray]:
    stems: dict[str, np.ndarray] = {}
    for name in ("vocals", "drums", "bass", "other"):
        path = paths.get(name)
        if path and Path(path).exists():
            stems[name] = _fit_length(_load_stereo(Path(path)), samples)
        else:
            stems[name] = np.zeros((2, samples), dtype=np.float32)
    if not np.any(stems["other"]) and paths.get("accompaniment") and Path(paths["accompaniment"]).exists():
        stems["other"] = _fit_length(_load_stereo(Path(paths["accompaniment"])), samples)
    return stems


def _fit_length(buffer: np.ndarray, samples: int) -> np.ndarray:
    if buffer.shape[1] == samples:
        return buffer.astype(np.float32)
    if buffer.shape[1] > samples:
        return np.ascontiguousarray(buffer[:, :samples], dtype=np.float32)
    padded = np.zeros((2, samples), dtype=np.float32)
    padded[:, : buffer.shape[1]] = buffer[:2]
    return padded


def _automation_curves(
    samples: int,
    method: str,
    vocal_conflict: float,
    vocal_handoff: dict[str, float] | None = None,
    energy_handoff: dict[str, float] | None = None,
    frequency_handoff: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    x = np.linspace(0, 1, samples, dtype=np.float32)
    high_vocal_conflict = vocal_conflict > 0.45
    if method == "quick_cut":
        bass_start, bass_end = 0.42, 0.58
        in_drum_start, in_drum_end = 0.2, 0.72
        out_drum_start = 0.7
        out_vocal_end = 0.18 if high_vocal_conflict else 0.32
        in_vocal_start = 0.66 if high_vocal_conflict else 0.38
    elif method == "echo_out":
        bass_start, bass_end = 0.22, 0.45
        in_drum_start, in_drum_end = 0.14, 0.72
        out_drum_start = 0.72
        out_vocal_end = 0.16 if high_vocal_conflict else 0.28
        in_vocal_start = 0.7 if high_vocal_conflict else 0.48
    else:
        bass_start, bass_end = 0.58, 0.74
        in_drum_start, in_drum_end = 0.12, 0.72
        out_drum_start = 0.82
        out_vocal_end = 0.24 if high_vocal_conflict else 0.48
        in_vocal_start = 0.68 if high_vocal_conflict else 0.34

    if vocal_handoff:
        out_vocal_end = float(vocal_handoff.get("outgoingEndFraction", out_vocal_end))
        in_vocal_start = float(vocal_handoff.get("incomingStartFraction", in_vocal_start))
        out_vocal_end = max(0.08, min(0.72, out_vocal_end))
        in_vocal_start = max(0.12, min(0.9, in_vocal_start))
    out_vocal_start = 0.02
    if vocal_handoff and vocal_handoff.get("outgoingFadeStartFraction") is not None:
        out_vocal_start = float(vocal_handoff["outgoingFadeStartFraction"])
        out_vocal_start = max(0.0, min(out_vocal_end - 0.02, out_vocal_start))

    out_bass = 1.0 - 0.96 * _smoothstep(_window(x, bass_start, bass_end))
    in_bass = 0.04 + 0.96 * _smoothstep(_window(x, bass_start, min(1.0, bass_end + 0.08)))
    out_drums = 1.0 - 0.92 * _smoothstep(_window(x, out_drum_start, 1.0))
    in_drums = 0.08 + 0.92 * _smoothstep(_window(x, in_drum_start, in_drum_end))
    out_vocals = 1.0 - _smoothstep(_window(x, out_vocal_start, out_vocal_end))
    in_vocals = _smoothstep(_window(x, in_vocal_start, 0.98))
    out_other_equal, in_other_equal = _fade_curves(samples, "equal_power")
    out_other = np.maximum(out_other_equal * 0.88, out_vocals * 0.22)
    in_other = np.maximum(in_other_equal, _smoothstep(_window(x, 0.08, 0.78)) * 0.72)
    energy_gate = _incoming_energy_gate(samples, energy_handoff)
    low_gate, mid_gate, high_gate = _incoming_band_gates(samples, frequency_handoff)
    in_bass *= energy_gate * low_gate
    in_drums *= energy_gate * high_gate
    in_other *= energy_gate * np.minimum(mid_gate, high_gate * 1.08)
    in_vocals *= mid_gate
    out_harmonic = out_vocals if high_vocal_conflict else out_other
    in_harmonic = in_vocals if high_vocal_conflict else in_other
    return {
        "out_bass": out_bass.reshape(1, -1).astype(np.float32),
        "in_bass": in_bass.reshape(1, -1).astype(np.float32),
        "out_drums": out_drums.reshape(1, -1).astype(np.float32),
        "in_drums": in_drums.reshape(1, -1).astype(np.float32),
        "out_vocals": out_vocals.reshape(1, -1).astype(np.float32),
        "in_vocals": in_vocals.reshape(1, -1).astype(np.float32),
        "out_other": out_other.reshape(1, -1).astype(np.float32),
        "in_other": in_other.reshape(1, -1).astype(np.float32),
        "out_harmonic": out_harmonic.reshape(1, -1).astype(np.float32),
        "in_harmonic": in_harmonic.reshape(1, -1).astype(np.float32),
    }


def _window(x: np.ndarray, start: float, end: float) -> np.ndarray:
    if end <= start:
        return (x >= end).astype(np.float32)
    return np.clip((x - start) / (end - start), 0, 1).astype(np.float32)


def _effective_overlap(suggested: float, method: str | None) -> float:
    floors = {
        "beatmix": 8.0,
        "bass_swap": 8.0,
        "breakdown_switch": 4.0,
        "filter_sweep": 4.0,
        "fade": 3.0,
        "echo_out": 3.0,
        "quick_cut": 1.5,
    }
    return max(0.5, float(suggested or 0), floors.get(method or "", 4.0))


def _adapt_render_method(
    method: str,
    vocal_conflict: float,
    vocal_handoff: dict[str, float],
    energy_handoff: dict[str, float],
    frequency_handoff: dict[str, float],
) -> str:
    if method not in {"beatmix", "bass_swap", "filter_sweep", "loop_build"}:
        return method
    guarded = bool(vocal_handoff.get("outgoingGuarded"))
    energy_trim = abs(float(energy_handoff.get("incomingTrimDb") or 0))
    band_trim = max(abs(float(frequency_handoff.get(key) or 0)) for key in ("lowTrimDb", "midTrimDb", "highTrimDb"))
    incoming_delay = float(vocal_handoff.get("incomingStartFraction") or 0)
    if vocal_conflict >= 0.72 and (guarded or incoming_delay >= 0.65):
        return "echo_out"
    if vocal_conflict >= 0.55 and (guarded or energy_trim >= 4.5 or band_trim >= 4.5):
        return "echo_out"
    if guarded and energy_trim >= 3.5:
        return "echo_out"
    return method


def _active_overlap_samples(samples: int, render_method: str, requested_method: str) -> int:
    if render_method == requested_method:
        return samples
    limits = {
        "quick_cut": 2.25,
        "echo_out": 8.0,
        "fade": 3.0,
        "breakdown_switch": 4.0,
    }
    limit = limits.get(render_method)
    if not limit:
        return samples
    return max(1, min(samples, _seconds_to_samples(limit)))


def _vocal_handoff_timing(outgoing_vocals: np.ndarray, incoming_vocals: np.ndarray, vocal_conflict: float, method: str) -> dict[str, float]:
    high_conflict = vocal_conflict > 0.45
    default_out = 0.16 if method in {"quick_cut", "echo_out"} else (0.24 if high_conflict else 0.46)
    default_in = 0.72 if high_conflict else (0.42 if method != "echo_out" else 0.5)
    outgoing_end = _last_vocal_release_fraction(outgoing_vocals, default_out, high_conflict)
    incoming_start = _first_vocal_phrase_fraction(incoming_vocals, default_in, high_conflict)
    next_outgoing_start = _next_vocal_reentry_fraction(outgoing_vocals)
    guard_active = False
    if next_outgoing_start is not None and next_outgoing_start <= min(0.92, incoming_start + 0.16):
        guarded_end = max(0.08, next_outgoing_start - 0.035)
        if guarded_end < outgoing_end or high_conflict:
            outgoing_end = min(outgoing_end, guarded_end)
            guard_active = True
    if incoming_start <= outgoing_end + 0.16 and high_conflict:
        incoming_start = min(0.9, outgoing_end + 0.22)
    fade_start = max(0.0, outgoing_end - (0.075 if guard_active else 0.22))
    return {
        "outgoingEndFraction": round(float(outgoing_end), 3),
        "outgoingFadeStartFraction": round(float(fade_start), 3),
        "outgoingNextVocalStartFraction": round(float(next_outgoing_start), 3) if next_outgoing_start is not None else None,
        "outgoingGuarded": bool(guard_active),
        "incomingStartFraction": round(float(incoming_start), 3),
    }


def _energy_handoff_profile(outgoing: np.ndarray, incoming: np.ndarray, method: str) -> dict[str, float]:
    outgoing_rms = _early_rms(outgoing)
    incoming_rms = _early_rms(incoming)
    diff_db = 20 * math.log10((incoming_rms + 1e-9) / (outgoing_rms + 1e-9))
    trim_db = min(6.0, max(0.0, diff_db - 1.5))
    if method == "quick_cut":
        hold_fraction, release_fraction = 0.18, 0.42
    elif method == "echo_out":
        hold_fraction, release_fraction = 0.28, 0.62
    else:
        hold_fraction, release_fraction = 0.38, 0.78
    if trim_db <= 0.25:
        hold_fraction, release_fraction = 0.0, 0.0
    return {
        "incomingEnergyDiffDb": round(float(diff_db), 3),
        "incomingTrimDb": round(float(-trim_db), 3),
        "holdFraction": round(float(hold_fraction), 3),
        "releaseFraction": round(float(release_fraction), 3),
    }


def _early_rms(buffer: np.ndarray) -> float:
    samples = max(1, min(buffer.shape[1], int(buffer.shape[1] * 0.4)))
    segment = buffer[:, :samples]
    return float(np.sqrt(np.mean(segment * segment)) + 1e-9)


def _incoming_energy_gate(samples: int, energy_handoff: dict[str, float] | None) -> np.ndarray:
    if not energy_handoff:
        return np.ones(samples, dtype=np.float32)
    trim_db = float(energy_handoff.get("incomingTrimDb") or 0)
    if trim_db >= -0.25:
        return np.ones(samples, dtype=np.float32)
    hold = float(energy_handoff.get("holdFraction") or 0)
    release = float(energy_handoff.get("releaseFraction") or hold)
    trim_gain = 10 ** (trim_db / 20)
    x = np.linspace(0, 1, samples, dtype=np.float32)
    release_curve = _smoothstep(_window(x, hold, max(hold + 0.01, release)))
    return (trim_gain + (1 - trim_gain) * release_curve).astype(np.float32)


def _frequency_handoff_profile(outgoing: np.ndarray, incoming: np.ndarray, method: str) -> dict[str, float]:
    out_low, out_mid, out_high = _split_bands(outgoing)
    in_low, in_mid, in_high = _split_bands(incoming)
    hold, release = _band_release_window(method)
    low_trim = _band_trim_db(_early_rms(out_low), _early_rms(in_low), threshold_db=0.5)
    mid_trim = _band_trim_db(_early_rms(out_mid), _early_rms(in_mid), threshold_db=1.25)
    high_trim = _band_trim_db(_early_rms(out_high), _early_rms(in_high), threshold_db=1.75)
    return {
        "lowTrimDb": round(low_trim, 3),
        "midTrimDb": round(mid_trim, 3),
        "highTrimDb": round(high_trim, 3),
        "holdFraction": round(hold, 3),
        "releaseFraction": round(release, 3),
    }


def _split_bands(buffer: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = _sos_filter(buffer, "lowpass", 180)
    high = _sos_filter(buffer, "highpass", 3800)
    mid = _sos_filter(buffer, "bandpass", (220, 3400))
    return low, mid, high


def _band_trim_db(outgoing_rms: float, incoming_rms: float, threshold_db: float) -> float:
    diff_db = 20 * math.log10((incoming_rms + 1e-9) / (outgoing_rms + 1e-9))
    return -min(6.0, max(0.0, diff_db - threshold_db))


def _band_release_window(method: str) -> tuple[float, float]:
    if method == "quick_cut":
        return 0.12, 0.36
    if method == "echo_out":
        return 0.24, 0.58
    return 0.32, 0.74


def _incoming_band_gates(samples: int, frequency_handoff: dict[str, float] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not frequency_handoff:
        ones = np.ones(samples, dtype=np.float32)
        return ones, ones, ones
    hold = float(frequency_handoff.get("holdFraction") or 0)
    release = float(frequency_handoff.get("releaseFraction") or hold)
    return (
        _band_gate(samples, float(frequency_handoff.get("lowTrimDb") or 0), hold, release),
        _band_gate(samples, float(frequency_handoff.get("midTrimDb") or 0), hold, max(hold + 0.01, release - 0.08)),
        _band_gate(samples, float(frequency_handoff.get("highTrimDb") or 0), max(0.0, hold - 0.08), max(hold + 0.01, release - 0.16)),
    )


def _band_gate(samples: int, trim_db: float, hold: float, release: float) -> np.ndarray:
    if trim_db >= -0.25:
        return np.ones(samples, dtype=np.float32)
    trim_gain = 10 ** (trim_db / 20)
    x = np.linspace(0, 1, samples, dtype=np.float32)
    release_curve = _smoothstep(_window(x, hold, max(hold + 0.01, release)))
    return (trim_gain + (1 - trim_gain) * release_curve).astype(np.float32)


def _transition_glue_layer(outgoing_texture: np.ndarray, incoming_texture: np.ndarray, method: str, vocal_conflict: float) -> np.ndarray:
    samples = min(outgoing_texture.shape[1], incoming_texture.shape[1])
    if samples <= 1:
        return np.zeros_like(outgoing_texture[:, :samples])
    outgoing_texture = outgoing_texture[:, :samples]
    incoming_texture = incoming_texture[:, :samples]
    wet = _glue_wet_for_method(method, vocal_conflict)
    if wet <= 0:
        return np.zeros_like(outgoing_texture)
    out_space = _sos_filter(outgoing_texture, "highpass", 450)
    in_space = _sos_filter(incoming_texture, "highpass", 450)
    out_space = _sos_filter(out_space, "lowpass", 9000)
    in_space = _sos_filter(in_space, "lowpass", 9000)
    out_space = _apply_short_diffusion(out_space, feedback=0.18 if method == "quick_cut" else 0.24)
    in_space = _apply_short_diffusion(in_space, feedback=0.12)
    x = np.linspace(0, 1, samples, dtype=np.float32)
    bridge = np.sin(np.pi * x).reshape(1, -1).astype(np.float32)
    fade_out, fade_in = _fade_curves(samples, "equal_power")
    glue = (out_space * fade_out + in_space * fade_in) * bridge * wet
    return np.clip(glue, -0.35, 0.35).astype(np.float32)


def _rhythm_bridge_layer(outgoing_drums: np.ndarray, method: str, vocal_conflict: float) -> np.ndarray:
    samples = outgoing_drums.shape[1]
    wet = _rhythm_bridge_wet_for_method(method, vocal_conflict)
    if samples <= 1 or wet <= 0:
        return np.zeros_like(outgoing_drums)
    bridge_samples = min(samples, _seconds_to_samples(_rhythm_bridge_duration(method)))
    high_percussion = _sos_filter(outgoing_drums[:, :bridge_samples], "highpass", 2200)
    high_percussion = _sos_filter(high_percussion, "lowpass", 11000)
    x = np.linspace(0, 1, bridge_samples, dtype=np.float32)
    tail = np.power(1 - _smoothstep(x), 1.4).reshape(1, -1).astype(np.float32)
    bridge = np.zeros_like(outgoing_drums)
    bridge[:, :bridge_samples] = high_percussion * tail * wet
    return np.clip(bridge, -0.28, 0.28).astype(np.float32)


def _rhythm_bridge_duration(method: str) -> float:
    if method == "quick_cut":
        return 0.65
    if method == "echo_out":
        return 1.15
    return 1.35


def _rhythm_bridge_wet_for_method(method: str, vocal_conflict: float) -> float:
    if vocal_conflict > 0.78:
        return 0.045
    if method == "quick_cut":
        return 0.055
    if method == "echo_out":
        return 0.075
    return 0.065


def _glue_wet_for_method(method: str, vocal_conflict: float) -> float:
    if method == "quick_cut":
        return 0.1 if vocal_conflict < 0.75 else 0.07
    if method == "echo_out":
        return 0.16
    return 0.12 if vocal_conflict < 0.55 else 0.09


def _apply_short_diffusion(buffer: np.ndarray, feedback: float) -> np.ndarray:
    out = buffer.astype(np.float32).copy()
    for delay_ms, gain in ((23, feedback), (47, feedback * 0.65), (71, feedback * 0.42)):
        delay = max(1, int(SAMPLE_RATE * delay_ms / 1000))
        if out.shape[1] > delay:
            out[:, delay:] += buffer[:, :-delay] * gain
    return np.clip(out, -1, 1).astype(np.float32)


def _first_vocal_phrase_fraction(buffer: np.ndarray, default_fraction: float, high_conflict: bool) -> float:
    envelope = _rms_envelope(buffer)
    if envelope.size < 2 or float(np.max(envelope)) < 1e-5:
        return default_fraction
    threshold = _activity_threshold(envelope)
    active = envelope >= threshold
    min_fraction = 0.5 if high_conflict else 0.22
    min_index = min(active.size - 1, max(0, int(active.size * min_fraction)))
    for index in range(min_index, active.size):
        previous_inactive = index == 0 or not bool(active[index - 1])
        if active[index] and previous_inactive:
            return max(default_fraction if high_conflict else min_fraction, index / active.size)
    if bool(active[min_index]):
        return default_fraction
    active_indexes = np.flatnonzero(active[min_index:])
    if active_indexes.size:
        return max(default_fraction if high_conflict else min_fraction, (min_index + int(active_indexes[0])) / active.size)
    return default_fraction


def _last_vocal_release_fraction(buffer: np.ndarray, default_fraction: float, high_conflict: bool) -> float:
    envelope = _rms_envelope(buffer)
    if envelope.size < 2 or float(np.max(envelope)) < 1e-5:
        return default_fraction
    threshold = _activity_threshold(envelope)
    active = envelope >= threshold
    search_end = min(active.size - 1, int(active.size * (0.42 if high_conflict else 0.58)))
    for index in range(search_end, 1, -1):
        if active[index - 1] and not active[index]:
            return max(0.08, min(0.72, index / active.size))
    if bool(active[0]) or bool(active[min(search_end, active.size - 1)]):
        return default_fraction
    return min(default_fraction, 0.22)


def _next_vocal_reentry_fraction(buffer: np.ndarray) -> float | None:
    regions = _vocal_active_regions(buffer)
    if len(regions) < 2:
        return None
    first_start, first_end = regions[0]
    for start, end in regions[1:]:
        if start - first_end >= 0.035 and end - start >= 0.035:
            return start
    return None


def _vocal_active_regions(buffer: np.ndarray) -> list[tuple[float, float]]:
    envelope = _rms_envelope(buffer)
    if envelope.size < 2 or float(np.max(envelope)) < 1e-5:
        return []
    active = envelope >= _activity_threshold(envelope)
    regions: list[tuple[float, float]] = []
    start: int | None = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= 2:
                regions.append((start / active.size, index / active.size))
            start = None
    if start is not None and active.size - start >= 2:
        regions.append((start / active.size, 1.0))
    return regions


def _rms_envelope(buffer: np.ndarray, frame_ms: float = 40.0) -> np.ndarray:
    mono = np.mean(buffer, axis=0)
    frame = max(32, int(SAMPLE_RATE * frame_ms / 1000))
    if mono.size < frame:
        return np.array([float(np.sqrt(np.mean(mono * mono)))], dtype=np.float32)
    frame_count = max(1, mono.size // frame)
    trimmed = mono[: frame_count * frame].reshape(frame_count, frame)
    envelope = np.sqrt(np.mean(trimmed * trimmed, axis=1))
    smooth = min(5, envelope.size)
    if smooth > 1:
        kernel = np.ones(smooth, dtype=np.float32) / smooth
        envelope = np.convolve(envelope, kernel, mode="same")
    return envelope.astype(np.float32)


def _activity_threshold(envelope: np.ndarray) -> float:
    peak = float(np.max(envelope))
    if peak <= 1e-8:
        return 1.0
    return max(peak * 0.18, float(np.percentile(envelope, 70)) * 0.55, 0.005)


def _estimate_transient_shift_samples(reference: np.ndarray, target: np.ndarray, max_shift_ms: float = 45.0) -> int:
    samples = min(reference.shape[1], target.shape[1])
    if samples < 32:
        return 0
    max_shift = min(int(SAMPLE_RATE * max_shift_ms / 1000), samples // 3)
    if max_shift < 2:
        return 0
    ref_env = _onset_envelope(reference[:, :samples])
    target_env = _onset_envelope(target[:, :samples])
    if float(np.max(ref_env)) < 1e-6 or float(np.max(target_env)) < 1e-6:
        return 0
    ref_env = ref_env - float(np.mean(ref_env))
    target_env = target_env - float(np.mean(target_env))
    correlation = signal.correlate(ref_env, target_env, mode="full", method="fft")
    lags = signal.correlation_lags(ref_env.size, target_env.size, mode="full")
    mask = np.abs(lags) <= max_shift
    if not np.any(mask):
        return 0
    best_lag = int(lags[mask][int(np.argmax(correlation[mask]))])
    return best_lag


def _onset_envelope(buffer: np.ndarray) -> np.ndarray:
    mono = np.mean(buffer, axis=0)
    envelope = np.maximum(np.diff(np.abs(mono), prepend=mono[0]), 0)
    window = max(8, int(SAMPLE_RATE * 0.003))
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(envelope, kernel, mode="same").astype(np.float32)


def _shift_audio(buffer: np.ndarray, samples: int) -> np.ndarray:
    if samples == 0:
        return buffer
    shifted = np.zeros_like(buffer)
    if samples > 0:
        shifted[:, samples:] = buffer[:, :-samples]
    else:
        shifted[:, :samples] = buffer[:, -samples:]
    return shifted.astype(np.float32)


def _smooth_transition_edges(outgoing_pre: np.ndarray, overlap: np.ndarray, after: np.ndarray, fade_ms: float = 12.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fade_samples = min(_seconds_to_samples(fade_ms / 1000), overlap.shape[1] // 4)
    if fade_samples <= 1:
        return outgoing_pre, overlap, after
    overlap = overlap.copy()
    if outgoing_pre.shape[1] >= fade_samples:
        fade = _smoothstep(np.linspace(0, 1, fade_samples, dtype=np.float32)).reshape(1, -1)
        start_blend = outgoing_pre[:, -fade_samples:] * (1 - fade) + overlap[:, :fade_samples] * fade
        overlap[:, :fade_samples] = start_blend
    if after.shape[1] >= fade_samples:
        fade = _smoothstep(np.linspace(0, 1, fade_samples, dtype=np.float32)).reshape(1, -1)
        end_blend = overlap[:, -fade_samples:] * (1 - fade) + after[:, :fade_samples] * fade
        overlap[:, -fade_samples:] = end_blend
    return outgoing_pre, overlap, after


def _maybe_separate_transition_stems(outgoing: np.ndarray, incoming: np.ndarray, workspace: Path, options: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    if not options.get("useStemSeparation"):
        return {"used": False, "paths": None}
    engine = options.get("stemEngine")
    if engine == "spleeter":
        return _separate_transition_stems_with_spleeter(outgoing, incoming, workspace, warnings)
    if engine != "demucs":
        warnings.append("Stem separation engine is disabled; using full mix fallback.")
        return {"used": False, "paths": None}
    if not _demucs_available():
        warnings.append("Demucs is not available; using full mix fallback.")
        return {"used": False, "paths": None}
    try:
        device = _resolve_torch_device(options.get("device", "auto"))
        out_path = workspace / "outgoing_segment.wav"
        in_path = workspace / "incoming_segment.wav"
        sf.write(out_path, outgoing.T, SAMPLE_RATE, subtype="PCM_16")
        sf.write(in_path, incoming.T, SAMPLE_RATE, subtype="PCM_16")
        return {
            "used": True,
            "paths": {
                "outgoing": _separate_stems_with_demucs_api(out_path, workspace / "outgoing_stems", device),
                "incoming": _separate_stems_with_demucs_api(in_path, workspace / "incoming_stems", device),
            },
        }
    except Exception as exc:
        warnings.append(f"Demucs stem separation failed; using full mix fallback. Detail: {exc}")
        return {"used": False, "paths": None}


def _separate_transition_stems_with_spleeter(outgoing: np.ndarray, incoming: np.ndarray, workspace: Path, warnings: list[str]) -> dict[str, Any]:
    if not _spleeter_available():
        warnings.append("Spleeter is not available; using full mix fallback.")
        return {"used": False, "paths": None}
    try:
        out_path = workspace / "outgoing_segment.wav"
        in_path = workspace / "incoming_segment.wav"
        sf.write(out_path, outgoing.T, SAMPLE_RATE, subtype="PCM_16")
        sf.write(in_path, incoming.T, SAMPLE_RATE, subtype="PCM_16")
        return {
            "used": True,
            "paths": {
                "outgoing": _separate_stems_with_spleeter_cli(out_path, workspace / "spleeter_outgoing"),
                "incoming": _separate_stems_with_spleeter_cli(in_path, workspace / "spleeter_incoming"),
            },
        }
    except Exception as exc:
        warnings.append(f"Spleeter stem separation failed; using full mix fallback. Detail: {exc}")
        return {"used": False, "paths": None}


def _spleeter_available() -> bool:
    if shutil.which("spleeter"):
        return True
    try:
        import spleeter  # noqa: F401
    except Exception:
        return False
    return True


def _separate_stems_with_spleeter_cli(input_path: Path, workspace: Path) -> dict[str, Path]:
    workspace.mkdir(parents=True, exist_ok=True)
    command = ["spleeter"] if shutil.which("spleeter") else ["python", "-m", "spleeter"]
    subprocess.run(
        [*command, "separate", "-p", "spleeter:4stems", "-o", str(workspace), str(input_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    stem_dir = workspace / input_path.stem
    stems = {stem: stem_dir / f"{stem}.wav" for stem in ("vocals", "drums", "bass", "other")}
    if all(path.exists() for path in stems.values()):
        return stems
    raise RuntimeError("Spleeter did not produce vocals/drums/bass/other stems.")


def _time_stretch_buffer(buffer: np.ndarray, ratio: float, workspace: Path, options: dict[str, Any], warnings: list[str]) -> np.ndarray:
    command = _rubberband_command()
    if command and options.get("timeStretchEngine") != "librosa_fallback":
        input_path = workspace / "tempo_in.wav"
        output_path = workspace / "tempo_out.wav"
        sf.write(input_path, buffer.T, SAMPLE_RATE, subtype="PCM_16")
        args = [command, "-3", "--fine", "-t", str(ratio), str(input_path), str(output_path)]
        try:
            subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            return _load_stereo(output_path)
        except subprocess.CalledProcessError as exc:
            warnings.append(f"Rubber Band time-stretch failed; using librosa fallback. Detail: {(exc.stderr or '').strip()[-160:]}")
    else:
        warnings.append("Rubber Band not available; using librosa time-stretch fallback.")
    return np.ascontiguousarray(librosa.effects.time_stretch(buffer, rate=ratio), dtype=np.float32)


def _pitch_shift_buffer(buffer: np.ndarray, semitones: int, workspace: Path, options: dict[str, Any], warnings: list[str]) -> np.ndarray:
    command = _rubberband_command()
    if command and options.get("timeStretchEngine") != "librosa_fallback":
        input_path = workspace / "pitch_in.wav"
        output_path = workspace / "pitch_out.wav"
        sf.write(input_path, buffer.T, SAMPLE_RATE, subtype="PCM_16")
        args = [command, "-3", "--fine", "-p", str(semitones)]
        if options.get("preserveFormants", True):
            args.append("-F")
        args.extend([str(input_path), str(output_path)])
        try:
            subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            return _load_stereo(output_path)
        except subprocess.CalledProcessError as exc:
            warnings.append(f"Rubber Band pitch-shift failed; using librosa fallback. Detail: {(exc.stderr or '').strip()[-160:]}")
    else:
        warnings.append("Rubber Band not available for pitch-shift; using librosa fallback.")
    return np.ascontiguousarray(librosa.effects.pitch_shift(y=buffer, sr=SAMPLE_RATE, n_steps=semitones), dtype=np.float32)


def _load_segment(path: Path, start: float, duration: float) -> np.ndarray:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=False, offset=max(0, start), duration=max(0.25, duration))
    if y.ndim == 1:
        y = np.vstack([y, y])
    if y.shape[0] > 2:
        y = y[:2]
    return np.ascontiguousarray(y, dtype=np.float32)


def _load_stereo(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=False)
    if y.ndim == 1:
        y = np.vstack([y, y])
    if y.shape[0] > 2:
        y = y[:2]
    return np.ascontiguousarray(y, dtype=np.float32)


def _match_loudness_report(outgoing: np.ndarray, incoming: np.ndarray) -> dict[str, float]:
    out_lufs = loudness_metrics(outgoing, SAMPLE_RATE)["lufs"]
    in_lufs = loudness_metrics(incoming, SAMPLE_RATE)["lufs"]
    return {"outgoingGainDb": 0.0, "incomingGainDb": round(out_lufs - in_lufs, 3), "loudnessDifferenceDb": round(in_lufs - out_lufs, 3)}


def _vocal_conflict(outgoing: dict[str, Any], incoming: dict[str, Any], outgoing_time: float, incoming_time: float, overlap: float) -> float:
    out_curve = outgoing.get("vocal_density_curve") or []
    in_curve = incoming.get("vocal_density_curve") or []
    values = []
    for offset in np.linspace(0, overlap, max(2, int(overlap // 2) + 1)):
        values.append(math.sqrt(_curve_value(out_curve, outgoing_time + float(offset), "density") * _curve_value(in_curve, incoming_time + float(offset), "density")))
    return round(float(np.mean(values)) if values else 0.0, 3)


def _curve_value(curve: list[dict], time_value: float, key: str) -> float:
    if not curve:
        return 0.35
    nearest = min(curve, key=lambda item: abs(float(item.get("time", 0)) - time_value))
    return max(0.0, min(1.0, float(nearest.get(key, 0.35))))


def _curve_average(curve: list[dict], start_time: float, duration: float, key: str) -> float:
    if duration <= 0:
        return _curve_value(curve, start_time, key)
    sample_count = max(2, min(16, int(math.ceil(duration)) + 1))
    values = [_curve_value(curve, start_time + float(offset), key) for offset in np.linspace(0, duration, sample_count)]
    return float(np.mean(values)) if values else _curve_value(curve, start_time, key)


def _curve_peak(curve: list[dict], start_time: float, duration: float, key: str) -> float:
    if duration <= 0:
        return _curve_value(curve, start_time, key)
    sample_count = max(2, min(18, int(math.ceil(duration * 2)) + 1))
    values = [_curve_value(curve, start_time + float(offset), key) for offset in np.linspace(0, duration, sample_count)]
    return float(np.max(values)) if values else _curve_value(curve, start_time, key)


def _snap_to_phrase(track: dict[str, Any], cue_time: float) -> dict[str, Any]:
    bars = [float(value) for value in track.get("bars") or []]
    if not bars:
        return {"time": cue_time, "downbeat": cue_time, "phrase": False}
    phrases = [float(value) for value in track.get("phrases") or []] or bars[::8]
    phrase_time = min(phrases, key=lambda value: abs(value - cue_time))
    bar_time = min(bars, key=lambda value: abs(value - cue_time))
    if abs(phrase_time - cue_time) <= 8:
        return {"time": phrase_time, "downbeat": bar_time, "phrase": True}
    return {"time": bar_time, "downbeat": bar_time, "phrase": False}


def _candidate_time(track: dict[str, Any], key: str) -> float:
    candidates = track.get("transition_candidates") or {}
    if key in candidates:
        return float(candidates[key])
    return float(track.get("duration") or 0) * (0.8 if key == "outro" else 0.1)


def _fade_curves(samples: int, curve: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 1, samples, dtype=np.float32)
    if curve == "smoothstep":
        y = _smoothstep(x)
        return (1 - y).astype(np.float32), y.astype(np.float32)
    return np.cos(x * np.pi / 2).astype(np.float32), np.sin(x * np.pi / 2).astype(np.float32)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    return (x * x * (3 - 2 * x)).astype(np.float32)


def _hpss(buffer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        harmonic, percussive = librosa.effects.hpss(buffer, kernel_size=(31, 31), margin=(1.0, 4.0))
        return harmonic.astype(np.float32), percussive.astype(np.float32)
    except Exception:
        return buffer * 0.5, buffer * 0.5


def _sos_filter(buffer: np.ndarray, kind: str, freq: float) -> np.ndarray:
    sos = signal.butter(2, freq, btype=kind, fs=SAMPLE_RATE, output="sos")
    return signal.sosfilt(sos, buffer, axis=1).astype(np.float32)


def _apply_echo_tail(buffer: np.ndarray, feedback: float, wet: float) -> np.ndarray:
    delay = max(1, int(0.5 * SAMPLE_RATE))
    out = buffer.copy()
    for channel in range(out.shape[0]):
        for index in range(delay, out.shape[1]):
            out[channel, index] += out[channel, index - delay] * feedback * wet
    return np.clip(out, -1, 1).astype(np.float32)


def _limit_peak(buffer: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    peak = float(np.max(np.abs(buffer)) + 1e-12)
    if peak <= ceiling:
        return buffer
    return (buffer * (ceiling / peak)).astype(np.float32)


def _seconds_to_samples(seconds: float) -> int:
    return max(1, int(round(seconds * SAMPLE_RATE)))


def _curve_for_method(method: str | None) -> str:
    if method == "quick_cut":
        return "smoothstep"
    if method == "fade":
        return "logarithmic"
    return "equal_power"


def _estimate_after_conflict(before: float, rec: dict[str, Any]) -> float:
    method = rec.get("method")
    if method in {"quick_cut", "echo_out"}:
        return round(before * 0.25, 3)
    return round(before * 0.55, 3)


def _risk_score(tempo: dict[str, Any], pitch: dict[str, Any], vocal: float, stems_used: bool) -> float:
    tempo_risk = {"low": 0.05, "medium": 0.18, "high": 0.38}[tempo["risk"]]
    pitch_risk = {"low": 0.04, "medium": 0.16, "high": 0.3}[pitch["risk"]]
    stem_penalty = 0 if stems_used else 0.12
    return round(min(1.0, tempo_risk + pitch_risk + vocal * 0.35 + stem_penalty), 3)


def _processing_explanation(
    rec: dict[str, Any],
    tempo: dict[str, Any],
    pitch: dict[str, Any],
    stems_used: bool,
    vocal: float,
    loudness: dict[str, float],
    transient_shift_samples: int = 0,
    vocal_handoff: dict[str, float] | None = None,
    energy_handoff: dict[str, float] | None = None,
    frequency_handoff: dict[str, float] | None = None,
    render_method: str | None = None,
    overlap: float = 0,
) -> str:
    method = rec.get("method", "beatmix")
    parts = [f"系统先把两首歌的乐句强拍对齐，再按 {method} 方式生成可试听的过渡片段。"]
    if render_method and render_method != method:
        parts.append(f"由于人声/能量风险较高，实际渲染已自动改成 {render_method}，避免硬做长时间叠加。")
    if tempo["shouldStretch"]:
        parts.append(f"B 歌速度按 {tempo['tempoChangePercent']}% 做轻微同步，避免鼓点慢慢漂移。")
    if pitch["shouldPitchShift"]:
        parts.append(f"B 歌做 {pitch['semitones']} 个半音的小范围变调，并尽量保留人声 formant。")
    if transient_shift_samples:
        shift_ms = round(transient_shift_samples / SAMPLE_RATE * 1000, 1)
        parts.append(f"系统检测到鼓组瞬态有约 {shift_ms}ms 的偏移，并对 B 歌鼓组/低频做了微调。")
    if vocal_handoff and vocal_handoff.get("incomingStartFraction") is not None and overlap:
        delay_ms = round(float(vocal_handoff["incomingStartFraction"]) * overlap * 1000)
        parts.append(f"B 歌人声会延后约 {delay_ms}ms 再打开，尽量贴近一句新歌词或新乐句进入。")
        if vocal_handoff.get("outgoingGuarded"):
            parts.append("系统检测到 A 歌下一句歌词可能会闯入过渡，已在下一句起唱前快速收掉 A 歌人声。")
    if energy_handoff and float(energy_handoff.get("incomingTrimDb") or 0) < -0.25:
        parts.append(f"B 歌前半段能量先压低约 {abs(float(energy_handoff['incomingTrimDb'])):.1f}dB，再在乐句后半段释放。")
    if frequency_handoff:
        active_bands = [
            label
            for label, key in (("低频", "lowTrimDb"), ("中频", "midTrimDb"), ("高频", "highTrimDb"))
            if float(frequency_handoff.get(key) or 0) < -0.25
        ]
        if active_bands:
            parts.append(f"频段接管会先压住 B 歌的{'、'.join(active_bands)}，再分层释放，避免整首歌一起冲进来。")
    parts.append("系统还会加入很轻的氛围胶水层，用 other/空间感盖住切换边界，让两首歌更像在同一个声场里交接。")
    parts.append("B 歌刚进来时会短暂保留一点 A 歌高频打击乐 ghost，保留律动空气感，但不会保留低频鼓。")
    if vocal > 0.45:
        parts.append("检测到人声冲突，过渡里会更快收掉 A 歌人声，并延后 B 歌人声进入。")
    parts.append("低频采用 bass swap：前半段保留 A 歌低频，后半段再交给 B 歌，避免两个 bass 同时顶满。")
    parts.append(f"过渡段响度差约 {loudness['loudnessDifferenceDb']} dB，导出前已做整体响度归一化。")
    if stems_used:
        parts.append("本次启用了 stem 分离，会分别控制 vocals、drums、bass、other，不再整首歌一起切换。")
    else:
        parts.append("本次使用 full mix fallback，会用 HPSS 和滤波模拟分层，但精度不如 Demucs/Spleeter。")
    return "".join(parts)


def _small_camelot_shift(outgoing_key: str, incoming_key: str) -> int | None:
    # Conservative placeholder: only same parsed pitch class can be shifted.
    if outgoing_key == incoming_key:
        return 0
    return None


def _fallback_recommendation(outgoing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": "beatmix",
        "overlapDuration": 8,
        "outgoingCue": {"time": _candidate_time(outgoing, "outro"), "role": "outro", "sectionType": "outro"},
        "incomingCue": {"time": _candidate_time(incoming, "intro"), "role": "entry", "sectionType": "intro"},
    }


def _convert_to_mp3(wav_path: Path) -> Path:
    mp3_path = wav_path.with_suffix(".mp3")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return mp3_path
