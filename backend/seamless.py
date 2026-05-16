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
        loudness = _match_loudness_report(outgoing_segment[:, : _seconds_to_samples(before)], incoming_segment[:, _seconds_to_samples(overlap) :])
        rendered = normalize_loudness(rendered, SAMPLE_RATE, -16)
        rendered = _limit_peak(rendered)

        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        wav_path = EXPORT_DIR / f"{uuid.uuid4().hex}_transition.wav"
        sf.write(wav_path, rendered.T, SAMPLE_RATE, subtype="PCM_16")
        output_path = wav_path
        if opts.get("exportFormat") == "mp3":
            output_path = _convert_to_mp3(wav_path)

    vocal_conflict_after = _estimate_after_conflict(vocal_conflict_before, rec)
    report = {
        "bpmBefore": {"outgoing": outgoing_analysis.get("bpm"), "incoming": incoming_analysis.get("bpm")},
        "bpmAfter": {"outgoing": outgoing_analysis.get("bpm"), "incoming": tempo["targetBpm"] if tempo["shouldStretch"] else incoming_analysis.get("bpm")},
        "keyBefore": {"outgoing": outgoing_analysis.get("camelot") or outgoing_analysis.get("key"), "incoming": incoming_analysis.get("camelot") or incoming_analysis.get("key")},
        "keyAfter": {"outgoing": outgoing_analysis.get("camelot") or outgoing_analysis.get("key"), "incoming": pitch["targetKey"] if pitch["shouldPitchShift"] else incoming_analysis.get("camelot") or incoming_analysis.get("key")},
        "tempoChangePercent": tempo["tempoChangePercent"],
        "pitchShiftSemitones": pitch["semitones"] if pitch["shouldPitchShift"] else 0,
        "usedStemSeparation": stem_report["used"],
        "usedFormantPreservation": bool(opts.get("preserveFormants") and pitch["shouldPitchShift"]),
        "crossfadeCurve": _curve_for_method(rec.get("method")),
        "overlapDuration": overlap,
        "vocalConflictBefore": vocal_conflict_before,
        "vocalConflictAfter": vocal_conflict_after,
        "loudnessMatchDb": loudness["loudnessDifferenceDb"],
        "transientShiftMs": round((stem_report.get("transientShiftSamples") or 0) / SAMPLE_RATE * 1000, 2),
        "incomingVocalDelayMs": round(((stem_report.get("vocalHandoff") or {}).get("incomingStartFraction") or 0) * overlap * 1000, 2),
        "incomingEnergyTrimDb": round(float((stem_report.get("energyHandoff") or {}).get("incomingTrimDb") or 0), 2),
        "incomingBandTrimDb": stem_report.get("frequencyHandoff") or {},
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
            overlap,
        ),
    }
    return {
        "audioPath": str(output_path),
        "url": f"/api/exports/{output_path.name}",
        "previewStartTime": out_start,
        "previewEndTime": out_start + rendered.shape[1] / SAMPLE_RATE,
        "outgoingCue": rec.get("outgoingCue"),
        "incomingCue": rec.get("incomingCue"),
        "alignment": alignment,
        "method": rec.get("method", "beatmix"),
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
    after = incoming[:, overlap_samples:]
    outgoing_pre = outgoing[:, :before_samples]
    out_overlap = outgoing[:, before_samples : before_samples + overlap_samples]
    in_overlap = incoming[:, :overlap_samples]
    method = rec.get("method", "beatmix")

    if method == "echo_out":
        out_overlap = _apply_echo_tail(out_overlap, feedback=0.36, wet=0.42)
    if stem_report.get("used") and stem_report.get("paths"):
        overlap_mix, shift, vocal_handoff, energy_handoff, frequency_handoff = _stem_overlap_from_paths(stem_report["paths"], before_samples, overlap_samples, method, vocal_conflict)
    else:
        overlap_mix, shift, vocal_handoff, energy_handoff, frequency_handoff = _stem_like_overlap(out_overlap, in_overlap, method, vocal_conflict, False)
    stem_report["transientShiftSamples"] = int(shift)
    stem_report["vocalHandoff"] = vocal_handoff
    stem_report["energyHandoff"] = energy_handoff
    stem_report["frequencyHandoff"] = frequency_handoff
    outgoing_pre, overlap_mix, after = _smooth_transition_edges(outgoing_pre, overlap_mix, after)
    return np.concatenate([outgoing_pre, overlap_mix, after], axis=1)


def _stem_like_overlap(outgoing: np.ndarray, incoming: np.ndarray, method: str, vocal_conflict: float, stems_used: bool) -> tuple[np.ndarray, int, dict[str, float], dict[str, float], dict[str, float]]:
    samples = min(outgoing.shape[1], incoming.shape[1])
    outgoing = outgoing[:, :samples]
    incoming = incoming[:, :samples]
    out_low = _sos_filter(outgoing, "lowpass", 160)
    in_low = _sos_filter(incoming, "lowpass", 160)
    out_harmonic, out_percussive = _hpss(outgoing - out_low)
    in_harmonic, in_percussive = _hpss(incoming - in_low)
    vocal_handoff = _vocal_handoff_timing(out_harmonic, in_harmonic, vocal_conflict, method)
    shift = _estimate_transient_shift_samples(out_percussive, in_percussive)
    in_low = _shift_audio(in_low, shift)
    in_percussive = _shift_audio(in_percussive, shift)
    energy_handoff = _energy_handoff_profile(out_low + out_percussive + out_harmonic * 0.35, in_low + in_percussive + in_harmonic * 0.35, method)
    frequency_handoff = _frequency_handoff_profile(outgoing, incoming, method)
    curves = _automation_curves(samples, method, vocal_conflict, vocal_handoff, energy_handoff, frequency_handoff)
    mix = (
        out_low * curves["out_bass"]
        + in_low * curves["in_bass"]
        + out_percussive * curves["out_drums"]
        + in_percussive * curves["in_drums"]
        + out_harmonic * curves["out_harmonic"]
        + in_harmonic * curves["in_harmonic"]
    )
    if vocal_conflict > 0.45:
        mix *= 0.92
    return np.clip(mix, -1, 1).astype(np.float32), shift, vocal_handoff, energy_handoff, frequency_handoff


def _stem_overlap_from_paths(stem_paths: dict[str, Any], before_samples: int, overlap_samples: int, method: str, vocal_conflict: float) -> tuple[np.ndarray, int, dict[str, float], dict[str, float], dict[str, float]]:
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
    curves = _automation_curves(overlap_samples, method, vocal_conflict, vocal_handoff, energy_handoff, frequency_handoff)
    if method == "echo_out":
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
    )
    return np.clip(mix, -1, 1).astype(np.float32), shift, vocal_handoff, energy_handoff, frequency_handoff


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
        in_drum_start, in_drum_end = 0.38, 0.86
        out_drum_start = 0.42
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

    out_bass = 1.0 - 0.96 * _smoothstep(_window(x, bass_start, bass_end))
    in_bass = 0.04 + 0.96 * _smoothstep(_window(x, bass_start, min(1.0, bass_end + 0.08)))
    out_drums = 1.0 - 0.92 * _smoothstep(_window(x, out_drum_start, 1.0))
    in_drums = 0.08 + 0.92 * _smoothstep(_window(x, in_drum_start, in_drum_end))
    out_vocals = 1.0 - _smoothstep(_window(x, 0.02, out_vocal_end))
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


def _vocal_handoff_timing(outgoing_vocals: np.ndarray, incoming_vocals: np.ndarray, vocal_conflict: float, method: str) -> dict[str, float]:
    high_conflict = vocal_conflict > 0.45
    default_out = 0.16 if method in {"quick_cut", "echo_out"} else (0.24 if high_conflict else 0.46)
    default_in = 0.72 if high_conflict else (0.42 if method != "echo_out" else 0.5)
    outgoing_end = _last_vocal_release_fraction(outgoing_vocals, default_out, high_conflict)
    incoming_start = _first_vocal_phrase_fraction(incoming_vocals, default_in, high_conflict)
    if incoming_start <= outgoing_end + 0.16 and high_conflict:
        incoming_start = min(0.9, outgoing_end + 0.22)
    return {
        "outgoingEndFraction": round(float(outgoing_end), 3),
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
    overlap: float = 0,
) -> str:
    method = rec.get("method", "beatmix")
    parts = [f"系统先把两首歌的乐句强拍对齐，再按 {method} 方式生成可试听的过渡片段。"]
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
    if energy_handoff and float(energy_handoff.get("incomingTrimDb") or 0) < -0.25:
        parts.append(f"B 歌前半段能量先压低约 {abs(float(energy_handoff['incomingTrimDb'])):.1f}dB，再在乐句后半段释放。")
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
