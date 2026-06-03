from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from .loudness import normalize_loudness
from .services.stem_separation import (
    demucs_available as sdk_demucs_available,
    load_stereo as sdk_load_stereo,
    prepare_demucs_input as sdk_prepare_demucs_input,
    resolve_torch_device as sdk_resolve_torch_device,
    separate_prepared_demucs_input as sdk_separate_prepared_demucs_input,
)
from .storage import EXPORT_DIR


SAMPLE_RATE = 44100
TARGET_LUFS = -14.0
PEAK_CEILING = 0.97

_CAMELOT_TO_TONIC = {
    # Minor keys, Camelot A.
    "1A": 8,
    "2A": 3,
    "3A": 10,
    "4A": 5,
    "5A": 0,
    "6A": 7,
    "7A": 2,
    "8A": 9,
    "9A": 4,
    "10A": 11,
    "11A": 6,
    "12A": 1,
    # Major keys, Camelot B.
    "1B": 11,
    "2B": 6,
    "3B": 1,
    "4B": 8,
    "5B": 3,
    "6B": 10,
    "7B": 5,
    "8B": 0,
    "9B": 7,
    "10B": 2,
    "11B": 9,
    "12B": 4,
}


@dataclass(frozen=True)
class TuningRenderResult:
    path: Path
    method: str
    semitones: int
    source_camelot: str
    target_camelot: str
    used_stems: bool
    device: str
    warnings: list[str]


def normalize_camelot(code: str | None) -> str | None:
    if not code:
        return None
    value = code.strip().upper()
    if len(value) < 2:
        return None
    number = value[:-1]
    mode = value[-1]
    if mode not in {"A", "B"}:
        return None
    try:
        parsed = int(number)
    except ValueError:
        return None
    if parsed < 1 or parsed > 12:
        return None
    return f"{parsed}{mode}"


def semitones_between_camelot(source_camelot: str, target_camelot: str, prefer: str = "nearest") -> int:
    source = normalize_camelot(source_camelot)
    target = normalize_camelot(target_camelot)
    if source not in _CAMELOT_TO_TONIC or target not in _CAMELOT_TO_TONIC:
        raise ValueError(f"Unsupported Camelot pair: {source_camelot} -> {target_camelot}")

    diff = (_CAMELOT_TO_TONIC[target] - _CAMELOT_TO_TONIC[source]) % 12
    if prefer == "up":
        return diff
    if prefer == "down":
        return diff - 12 if diff else 0
    return diff - 12 if diff >= 6 else diff


def harmonic_targets(reference_camelot: str | None) -> list[str]:
    reference = normalize_camelot(reference_camelot)
    if not reference:
        return []
    number = int(reference[:-1])
    mode = reference[-1]
    other_mode = "B" if mode == "A" else "A"
    prev_number = 12 if number == 1 else number - 1
    next_number = 1 if number == 12 else number + 1
    return [
        f"{number}{mode}",
        f"{prev_number}{mode}",
        f"{next_number}{mode}",
        f"{number}{other_mode}",
    ]


def recommend_pair_tuning(track_a: dict[str, Any], track_b: dict[str, Any]) -> list[dict[str, Any]]:
    a_code = normalize_camelot(track_a.get("camelot"))
    b_code = normalize_camelot(track_b.get("camelot"))
    if not a_code or not b_code:
        return []
    if b_code in harmonic_targets(a_code):
        return []

    options: list[dict[str, Any]] = []
    for source_name, source_track, source_code, reference_track, reference_code in (
        ("track_a", track_a, a_code, track_b, b_code),
        ("track_b", track_b, b_code, track_a, a_code),
    ):
        for target in harmonic_targets(reference_code):
            semitones = semitones_between_camelot(source_code, target)
            if semitones == 0:
                continue
            if abs(semitones) > 6:
                continue
            score = _tuning_option_score(semitones, target, reference_code)
            options.append(
                {
                    "source": source_name,
                    "track_id": source_track.get("id"),
                    "track_name": source_track.get("name"),
                    "from_camelot": source_code,
                    "target_camelot": target,
                    "reference_camelot": reference_code,
                    "reference_track_id": reference_track.get("id"),
                    "reference_track_name": reference_track.get("name"),
                    "semitones": semitones,
                    "quality_risk": _quality_risk(semitones),
                    "score": score,
                    "reason": _recommendation_reason(source_track, target, semitones),
                }
            )

    deduped: dict[tuple[str, str, int], dict[str, Any]] = {}
    for option in options:
        key = (option["source"], option["target_camelot"], option["semitones"])
        current = deduped.get(key)
        if current is None or option["score"] > current["score"]:
            deduped[key] = option

    return sorted(deduped.values(), key=lambda item: (-item["score"], abs(item["semitones"])))[:6]


def render_harmonic_tune(
    input_path: Path,
    source_camelot: str,
    target_camelot: str,
    output_path: Path | None = None,
    prefer_direction: str = "nearest",
    fmt: str = "wav",
    device: str = "auto",
) -> TuningRenderResult:
    semitones = semitones_between_camelot(source_camelot, target_camelot, prefer_direction)
    if semitones == 0:
        raise ValueError("Source and target Camelot keys do not require pitch shifting")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        suffix = ".mp3" if fmt == "mp3" else ".wav"
        output_path = EXPORT_DIR / f"{uuid.uuid4().hex}_{normalize_camelot(source_camelot)}_to_{normalize_camelot(target_camelot)}{suffix}"

    warnings: list[str] = []
    with tempfile.TemporaryDirectory(prefix="smartmix_tune_") as tmp:
        workspace = Path(tmp)
        stems, used_device = _separate_stems(input_path, workspace, warnings, device)
        if stems:
            rendered = _render_from_stems(stems, workspace, semitones, warnings)
            used_stems = True
            method = "demucs+rubberband" if _rubberband_command() else "demucs+librosa"
        else:
            rendered = _render_master_fallback(input_path, workspace, semitones, warnings)
            used_stems = False
            method = "rubberband-master" if _rubberband_command() else "librosa-master"
            used_device = "none"

        rendered = _master_polish(rendered)
        wav_path = output_path if output_path.suffix.lower() == ".wav" else output_path.with_suffix(".wav")
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav_path, rendered.T, SAMPLE_RATE, subtype="PCM_16")

        final_path = wav_path
        if fmt == "mp3" or output_path.suffix.lower() == ".mp3":
            final_path = _convert_to_mp3(wav_path, output_path.with_suffix(".mp3"))
            if output_path.suffix.lower() == ".mp3":
                wav_path.unlink(missing_ok=True)

    return TuningRenderResult(
        path=final_path,
        method=method,
        semitones=semitones,
        source_camelot=normalize_camelot(source_camelot) or source_camelot,
        target_camelot=normalize_camelot(target_camelot) or target_camelot,
        used_stems=used_stems,
        device=used_device,
        warnings=warnings,
    )


def analyze_tuned_output(result: TuningRenderResult, original_meta: dict[str, Any]) -> dict[str, Any]:
    from .analysis import analyze_audio

    analysis = analyze_audio(result.path)
    return {
        **original_meta,
        "path": str(result.path),
        "content_type": "audio/mpeg" if result.path.suffix.lower() == ".mp3" else "audio/wav",
        "key": analysis.get("key"),
        "camelot": analysis.get("camelot") or result.target_camelot,
        "key_index": analysis.get("key_index"),
        "mode": analysis.get("mode"),
        "duration": analysis.get("duration"),
        "bpm": analysis.get("bpm"),
        "beats": analysis.get("beats"),
        "bars": analysis.get("bars"),
        "phrases": analysis.get("phrases"),
        "energy": analysis.get("energy"),
        "intro_low": analysis.get("intro_low"),
        "outro_low": analysis.get("outro_low"),
        "loudness_lufs": analysis.get("loudness_lufs"),
        "true_peak_db": analysis.get("true_peak_db"),
        "transition_candidates": analysis.get("transition_candidates"),
        "peaks": analysis.get("peaks"),
        "tuning": {
            "method": result.method,
            "from_camelot": result.source_camelot,
            "target_camelot": result.target_camelot,
            "semitones": result.semitones,
            "used_stems": result.used_stems,
            "device": result.device,
            "warnings": result.warnings,
        },
    }


def _separate_stems(
    input_path: Path,
    workspace: Path,
    warnings: list[str],
    requested_device: str,
) -> tuple[dict[str, Path] | None, str]:
    if not _demucs_available():
        warnings.append("Demucs is not installed; using whole-track pitch shifting fallback.")
        return None, "none"

    demucs_input = _prepare_demucs_input(input_path, workspace)
    try:
        device = _resolve_torch_device(requested_device)
        return _separate_stems_with_demucs_api(demucs_input, workspace, device), device
    except Exception as exc:
        warnings.append(f"Demucs separation failed; using whole-track fallback. Detail: {exc}")
        return None, "none"


def _separate_stems_with_demucs_api(input_path: Path, workspace: Path, device: str) -> dict[str, Path]:
    return sdk_separate_prepared_demucs_input(input_path, workspace, device)


def _resolve_torch_device(requested_device: str) -> str:
    return sdk_resolve_torch_device(requested_device)


def _prepare_demucs_input(input_path: Path, workspace: Path) -> Path:
    return sdk_prepare_demucs_input(input_path, workspace)


def _render_from_stems(stems: dict[str, Path], workspace: Path, semitones: int, warnings: list[str]) -> np.ndarray:
    processed_paths: list[Path] = []
    for stem, path in stems.items():
        if stem == "drums":
            processed_paths.append(path)
            continue
        out_path = workspace / f"{stem}_tuned.wav"
        formant = stem == "vocals"
        _pitch_shift_file(path, out_path, semitones, formant=formant, warnings=warnings)
        processed_paths.append(out_path)

    buffers = [_load_stereo(path) for path in processed_paths]
    length = max(buffer.shape[1] for buffer in buffers)
    mix = np.zeros((2, length), dtype=np.float32)
    gains = {
        "vocals_tuned": 1.0,
        "bass_tuned": 0.95,
        "other_tuned": 0.92,
        "drums": 1.0,
    }
    for path, buffer in zip(processed_paths, buffers):
        padded = _pad_to_length(buffer, length)
        gain = next((value for key, value in gains.items() if key in path.stem), 1.0)
        mix += padded * gain
    return np.clip(mix, -1, 1)


def _render_master_fallback(input_path: Path, workspace: Path, semitones: int, warnings: list[str]) -> np.ndarray:
    out_path = workspace / "master_tuned.wav"
    _pitch_shift_file(input_path, out_path, semitones, formant=True, warnings=warnings)
    return _load_stereo(out_path)


def _pitch_shift_file(input_path: Path, output_path: Path, semitones: int, formant: bool, warnings: list[str]) -> None:
    command = _rubberband_command()
    if command:
        args = [command, "-3", "-p", str(semitones)]
        if formant:
            args.append("-F")
        args.extend([str(input_path), str(output_path)])
        try:
            subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            return
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip().splitlines()[-1:] or [str(exc)]
            warnings.append(f"Rubber Band failed for {input_path.name}; using librosa fallback. Detail: {detail[0]}")

    warnings.append(f"Rubber Band is not available for {input_path.name}; using librosa fallback.")
    y = _load_stereo(input_path)
    shifted = librosa.effects.pitch_shift(y=y, sr=SAMPLE_RATE, n_steps=semitones, bins_per_octave=12)
    sf.write(output_path, shifted.T, SAMPLE_RATE, subtype="PCM_16")


def _master_polish(buffer: np.ndarray) -> np.ndarray:
    cleaned = _sos_filter(buffer, "highpass", 28)
    cleaned = _gentle_presence(cleaned)
    return normalize_loudness(cleaned, SAMPLE_RATE, TARGET_LUFS, peak_ceiling=PEAK_CEILING)


def _gentle_presence(buffer: np.ndarray) -> np.ndarray:
    high = _sos_filter(buffer, "highpass", 3600)
    low = _sos_filter(buffer, "lowpass", 180)
    out = buffer + high * 0.04 - low * 0.03
    return np.clip(out, -1, 1).astype(np.float32)


def _load_stereo(path: Path) -> np.ndarray:
    return sdk_load_stereo(path)


def _pad_to_length(buffer: np.ndarray, length: int) -> np.ndarray:
    if buffer.shape[1] >= length:
        return buffer[:, :length]
    return np.pad(buffer, ((0, 0), (0, length - buffer.shape[1])))


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
    return mp3_path


def _demucs_available() -> bool:
    return sdk_demucs_available()


def _rubberband_command() -> str | None:
    local_tools = Path(__file__).resolve().parents[1] / ".tools" / "rubberband"
    local_candidates = [
        local_tools / "rubberband-r3.exe",
        local_tools / "rubberband.exe",
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return str(candidate)
    nested_candidates = [
        *local_tools.rglob("rubberband-r3.exe"),
        *local_tools.rglob("rubberband.exe"),
    ] if local_tools.exists() else []
    if nested_candidates:
        return str(nested_candidates[0])
    return shutil.which("rubberband-r3") or shutil.which("rubberband")


def _tuning_option_score(semitones: int, target: str, reference: str) -> float:
    penalty = abs(semitones) * 8
    relationship_bonus = 8 if target == reference else 0
    return round(max(0, 100 - penalty + relationship_bonus), 1)


def _quality_risk(semitones: int) -> str:
    amount = abs(semitones)
    if amount <= 2:
        return "low"
    if amount <= 4:
        return "medium"
    return "high"


def _recommendation_reason(track: dict[str, Any], target: str, semitones: int) -> str:
    name = track.get("name") or "this track"
    direction = "up" if semitones > 0 else "down"
    return f"Tune {name} {direction} {abs(semitones)} semitone(s) to {target} for a harmonic Camelot relationship."
