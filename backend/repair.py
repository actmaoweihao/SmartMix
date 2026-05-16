from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from .analysis import analyze_audio
from .loudness import normalize_loudness
from .matching import camelot_key_distance, evaluate_track_match
from .storage import EXPORT_DIR
from .tuning import SAMPLE_RATE, harmonic_targets, normalize_camelot, semitones_between_camelot


@dataclass(frozen=True)
class MatchRepairOptions:
    process_target: str = "auto"
    include_key: bool = True
    include_tempo: bool = True
    include_energy: bool = True
    max_tempo_change_percent: float = 10.0
    max_pitch_shift_semitones: int = 4
    format: str = "wav"


def repair_track_for_match(
    track_a: dict[str, Any],
    track_b: dict[str, Any],
    options: MatchRepairOptions,
) -> dict[str, Any]:
    original_match = evaluate_track_match(track_a, track_b)
    plan = build_match_repair_plan(track_a, track_b, options)
    if not plan["operations"]:
        return {
            "original_match": original_match,
            "repair_plan": plan,
            "repaired_track": None,
            "repaired_match": original_match,
            "warnings": ["The pair is already close enough, or requested limits do not allow safe processing."],
        }

    source_track = track_a if plan["source"] == "track_a" else track_b
    output_path = render_match_repair(Path(source_track["path"]), plan, options)
    repaired_analysis = analyze_audio(output_path)
    repaired_track = {
        **source_track,
        **repaired_analysis,
        "id": uuid.uuid4().hex,
        "name": _processed_name(source_track.get("name") or "track", output_path.suffix),
        "path": str(output_path),
        "content_type": "audio/mpeg" if output_path.suffix.lower() == ".mp3" else "audio/wav",
        "repair": plan,
    }
    repaired_pair = (repaired_track, track_b) if plan["source"] == "track_a" else (track_a, repaired_track)
    repaired_match = evaluate_track_match(repaired_pair[0], repaired_pair[1])

    return {
        "original_match": original_match,
        "repair_plan": plan,
        "repaired_track": repaired_track,
        "repaired_match": repaired_match,
        "warnings": plan["warnings"],
    }


def build_match_repair_plan(track_a: dict[str, Any], track_b: dict[str, Any], options: MatchRepairOptions) -> dict[str, Any]:
    candidates = []
    if options.process_target in {"auto", "track_a", "a"}:
        candidates.append(_candidate_plan("track_a", track_a, "track_b", track_b, options))
    if options.process_target in {"auto", "track_b", "b", "incoming"}:
        candidates.append(_candidate_plan("track_b", track_b, "track_a", track_a, options))
    if not candidates:
        raise ValueError("process_target must be auto, track_a, track_b, a, b, or incoming")
    return min(candidates, key=lambda item: (item["risk_score"], -len(item["operations"])))


def render_match_repair(input_path: Path, plan: dict[str, Any], options: MatchRepairOptions) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ".mp3" if options.format == "mp3" else ".wav"
    output_path = EXPORT_DIR / f"{uuid.uuid4().hex}_match_repair{suffix}"

    buffer = _load_stereo(input_path)
    tempo = plan.get("tempo") or {}
    pitch = plan.get("pitch") or {}
    energy = plan.get("energy") or {}

    tempo_rate = float(tempo.get("rate") or 1.0)
    if tempo.get("enabled") and abs(tempo_rate - 1.0) > 0.002:
        buffer = librosa.effects.time_stretch(buffer, rate=tempo_rate)

    semitones = int(pitch.get("semitones") or 0)
    if pitch.get("enabled") and semitones:
        buffer = librosa.effects.pitch_shift(y=buffer, sr=SAMPLE_RATE, n_steps=semitones, bins_per_octave=12)

    if energy.get("enabled"):
        buffer = _match_energy(buffer, energy)

    wav_path = output_path if output_path.suffix.lower() == ".wav" else output_path.with_suffix(".wav")
    sf.write(wav_path, buffer.T, SAMPLE_RATE, subtype="PCM_16")
    if output_path.suffix.lower() == ".mp3":
        return _convert_to_mp3(wav_path, output_path)
    return wav_path


def _candidate_plan(
    source_label: str,
    source: dict[str, Any],
    reference_label: str,
    reference: dict[str, Any],
    options: MatchRepairOptions,
) -> dict[str, Any]:
    warnings: list[str] = []
    operations: list[dict[str, Any]] = []
    pitch = _pitch_plan(source, reference, options, warnings)
    tempo = _tempo_plan(source, reference, options, warnings)
    energy = _energy_plan(source, reference, options, warnings)

    for item in (pitch, tempo, energy):
        if item.get("enabled"):
            operations.append(item)

    risk_score = (
        abs(int(pitch.get("semitones") or 0)) * 10
        + abs(float(tempo.get("change_percent") or 0)) * 2
        + abs(float(energy.get("lufs_delta") or 0)) * 3
        + abs(float(energy.get("low_gain") or 1) - 1) * 20
    )
    return {
        "source": source_label,
        "reference": reference_label,
        "source_track_id": source.get("id"),
        "source_track_name": source.get("name"),
        "reference_track_id": reference.get("id"),
        "reference_track_name": reference.get("name"),
        "pitch": pitch,
        "tempo": tempo,
        "energy": energy,
        "operations": operations,
        "risk_score": round(float(risk_score), 2),
        "warnings": warnings,
        "explanation": _plan_explanation(source_label, reference_label, operations),
    }


def _pitch_plan(source: dict[str, Any], reference: dict[str, Any], options: MatchRepairOptions, warnings: list[str]) -> dict[str, Any]:
    source_key = normalize_camelot(source.get("camelot"))
    reference_key = normalize_camelot(reference.get("camelot"))
    base = {"type": "pitch", "enabled": False, "from_camelot": source_key, "target_camelot": source_key, "semitones": 0}
    if not options.include_key:
        return base
    if not source_key or not reference_key:
        warnings.append("Missing Camelot key; skipping pitch repair.")
        return base
    if camelot_key_distance(source_key, reference_key)["relation"] != "clash":
        return base

    candidates = []
    for target in harmonic_targets(reference_key):
        semitones = semitones_between_camelot(source_key, target)
        if abs(semitones) <= options.max_pitch_shift_semitones:
            relation = camelot_key_distance(target, reference_key)
            candidates.append((abs(semitones), -relation["score"], semitones, target, relation["relation"]))
    if not candidates:
        warnings.append("No safe Camelot pitch shift found within the configured semitone limit.")
        return base

    _, _, semitones, target, relation = min(candidates)
    return {
        "type": "pitch",
        "enabled": semitones != 0,
        "from_camelot": source_key,
        "target_camelot": target,
        "reference_camelot": reference_key,
        "semitones": semitones,
        "expected_relation": relation,
        "reason": f"Shift {source_key} by {semitones:+d} semitone(s) toward {target} for Camelot compatibility with {reference_key}.",
    }


def _tempo_plan(source: dict[str, Any], reference: dict[str, Any], options: MatchRepairOptions, warnings: list[str]) -> dict[str, Any]:
    source_bpm = _float_or_none(source.get("bpm"))
    reference_bpm = _float_or_none(reference.get("bpm"))
    base = {"type": "tempo", "enabled": False, "from_bpm": source_bpm, "target_bpm": reference_bpm, "rate": 1.0, "change_percent": 0.0}
    if not options.include_tempo:
        return base
    if not source_bpm or not reference_bpm:
        warnings.append("Missing BPM; skipping tempo repair.")
        return base

    variants = [(source_bpm, 1.0), (source_bpm * 2, 1.0), (source_bpm / 2, 1.0)]
    compatible_bpm, _ = min(variants, key=lambda item: abs(item[0] - reference_bpm))
    if compatible_bpm != source_bpm and abs(compatible_bpm - reference_bpm) / reference_bpm <= 0.03:
        return {**base, "target_bpm": compatible_bpm, "reason": "BPM is already compatible through half/double-time interpretation."}

    change_percent = abs(reference_bpm - source_bpm) / max(source_bpm, 1) * 100
    if change_percent > options.max_tempo_change_percent:
        warnings.append("BPM gap is too large for safe automatic time-stretching.")
        return {**base, "change_percent": round(change_percent, 3)}
    rate = reference_bpm / source_bpm
    return {
        **base,
        "enabled": change_percent > 0.25,
        "target_bpm": reference_bpm,
        "rate": round(rate, 6),
        "change_percent": round(change_percent, 3),
        "reason": f"Time-stretch from {source_bpm:g} BPM toward {reference_bpm:g} BPM.",
    }


def _energy_plan(source: dict[str, Any], reference: dict[str, Any], options: MatchRepairOptions, warnings: list[str]) -> dict[str, Any]:
    source_profile = source.get("energy_profile") or {}
    reference_profile = reference.get("energy_profile") or {}
    source_lufs = _float_or_none(source_profile.get("lufs") or source.get("loudness_lufs"))
    reference_lufs = _float_or_none(reference_profile.get("lufs") or reference.get("loudness_lufs"))
    source_low = _float_or_none(source_profile.get("low_frequency_ratio"))
    reference_low = _float_or_none(reference_profile.get("low_frequency_ratio"))
    base = {"type": "energy", "enabled": False, "target_lufs": reference_lufs, "low_gain": 1.0, "lufs_delta": 0.0}
    if not options.include_energy:
        return base
    if source_lufs is None or reference_lufs is None:
        warnings.append("Missing LUFS data; skipping loudness repair.")
        return base

    lufs_delta = reference_lufs - source_lufs
    low_gain = 1.0
    if source_low and reference_low:
        low_gain = float(np.clip(reference_low / max(source_low, 1e-6), 0.72, 1.28))
    enabled = abs(lufs_delta) > 0.8 or abs(low_gain - 1.0) > 0.08
    return {
        **base,
        "enabled": enabled,
        "source_lufs": source_lufs,
        "target_lufs": reference_lufs,
        "lufs_delta": round(float(lufs_delta), 3),
        "source_low_ratio": source_low,
        "reference_low_ratio": reference_low,
        "low_gain": round(low_gain, 4),
        "reason": f"Match loudness toward {reference_lufs:g} LUFS and adjust low-frequency energy.",
    }


def _match_energy(buffer: np.ndarray, energy: dict[str, Any]) -> np.ndarray:
    low_gain = float(energy.get("low_gain") or 1.0)
    low = _sos_filter(buffer, "lowpass", 180)
    high = buffer - low
    adjusted = high + low * low_gain
    target_lufs = _float_or_none(energy.get("target_lufs"))
    if target_lufs is not None:
        adjusted = normalize_loudness(adjusted, SAMPLE_RATE, target_lufs, peak_ceiling=0.97)
    peak = float(np.max(np.abs(adjusted)) + 1e-12)
    if peak > 0.98:
        adjusted *= 0.98 / peak
    return adjusted.astype(np.float32)


def _load_stereo(path: Path) -> np.ndarray:
    try:
        audio, source_sr = sf.read(path, always_2d=True, dtype="float32")
        if source_sr != SAMPLE_RATE:
            audio = librosa.resample(audio.T, orig_sr=source_sr, target_sr=SAMPLE_RATE).T
        y = audio.T
    except Exception:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=False)
    if y.ndim == 1:
        y = np.vstack([y, y])
    if y.shape[0] > 2:
        y = y[:2]
    return np.ascontiguousarray(y, dtype=np.float32)


def _sos_filter(buffer: np.ndarray, kind: str, freq: float) -> np.ndarray:
    sos = signal.butter(2, freq, btype=kind, fs=SAMPLE_RATE, output="sos")
    return signal.sosfilt(sos, buffer, axis=1).astype(np.float32)


def _convert_to_mp3(wav_path: Path, mp3_path: Path) -> Path:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", "256k", str(mp3_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wav_path.unlink(missing_ok=True)
    return mp3_path


def _processed_name(name: str, suffix: str) -> str:
    stem = Path(name).stem
    return f"{stem}_match_repaired{suffix}"


def _plan_explanation(source: str, reference: str, operations: list[dict[str, Any]]) -> str:
    if not operations:
        return "No processing was selected because the tracks are already compatible or limits prevented safe changes."
    labels = ", ".join(item["type"] for item in operations)
    return f"Process {source} to better match {reference}: {labels}."


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
