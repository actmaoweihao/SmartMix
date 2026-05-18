from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from .loudness import normalize_loudness
from .storage import EXPORT_DIR
from .transition import plan_transition


SAMPLE_RATE = 44100


def render_mix(tracks: list[dict], settings: dict, fmt: str) -> Path:
    if not tracks:
        raise ValueError("没有可导出的曲目")

    buffers = [_load_stereo(Path(track["path"])) for track in tracks]
    if settings.get("beatSync"):
        buffers = _beat_sync(buffers, tracks)

    if settings.get("aiPrecision") or settings.get("loudnessNormalize"):
        buffers = [normalize_loudness(buffer, SAMPLE_RATE, float(settings.get("targetLufs", -16))) for buffer in buffers]

    buffers = [_apply_track_mixer(buffer, track) for buffer, track in zip(buffers, tracks)]
    buffers = [_apply_static_eq(buffer, settings.get("eq", {})) for buffer in buffers]
    mix = _crossfade(buffers, tracks, settings)
    if settings.get("aiPrecision") or settings.get("loudnessNormalize"):
        mix = normalize_loudness(mix, SAMPLE_RATE, float(settings.get("targetLufs", -16)))
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = EXPORT_DIR / f"{uuid.uuid4().hex}.wav"
    sf.write(wav_path, mix.T, SAMPLE_RATE, subtype="PCM_16")

    if fmt == "wav":
        return wav_path
    if fmt == "mp3":
        return _convert_to_mp3(wav_path)
    raise ValueError("不支持的导出格式")


def _load_stereo(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=False)
    if y.ndim == 1:
        y = np.vstack([y, y])
    if y.shape[0] > 2:
        y = y[:2]
    return np.ascontiguousarray(y, dtype=np.float32)


def _beat_sync(buffers: list[np.ndarray], tracks: list[dict]) -> list[np.ndarray]:
    bpms = [float(track.get("bpm") or 0) for track in tracks]
    valid = [bpm for bpm in bpms if bpm > 0]
    if not valid:
        return buffers
    target = float(np.median(valid))
    synced = []
    for buffer, bpm in zip(buffers, bpms):
        if bpm <= 0:
            synced.append(buffer)
            continue
        rate = np.clip(target / bpm, 0.88, 1.12)
        if abs(rate - 1) < 0.015:
            synced.append(buffer)
            continue
        stretched = librosa.effects.time_stretch(buffer, rate=rate)
        synced.append(np.ascontiguousarray(stretched, dtype=np.float32))
    return synced


def _apply_static_eq(buffer: np.ndarray, eq: dict) -> np.ndarray:
    low = float(eq.get("low", 0))
    mid = float(eq.get("mid", 0))
    high = float(eq.get("high", 0))
    if abs(low) < 0.01 and abs(mid) < 0.01 and abs(high) < 0.01:
        return buffer

    low_band = _sos_filter(buffer, "lowpass", 220)
    high_band = _sos_filter(buffer, "highpass", 3200)
    mid_band = buffer - low_band - high_band
    out = buffer + low_band * low + mid_band * mid + high_band * high
    return np.clip(out, -1, 1).astype(np.float32)


def _apply_track_mixer(buffer: np.ndarray, track: dict) -> np.ndarray:
    mixer = track.get("mixer") or {}
    eq = mixer.get("eq") or {}
    gain = float(mixer.get("gain", 1.0))
    out = _apply_static_eq(buffer, eq)
    return np.clip(out * gain, -1, 1).astype(np.float32)


def _sos_filter(buffer: np.ndarray, kind: str, freq: float) -> np.ndarray:
    sos = signal.butter(2, freq, btype=kind, fs=SAMPLE_RATE, output="sos")
    return signal.sosfilt(sos, buffer, axis=1).astype(np.float32)


def _crossfade(buffers: list[np.ndarray], tracks: list[dict], settings: dict) -> np.ndarray:
    requested = float(settings.get("crossfade", 8))
    auto = bool(settings.get("autoTransition", True))
    filter_mode = settings.get("filterMode", "none")
    ai_precision = bool(settings.get("aiPrecision", False))
    rendered = buffers[0]

    for index in range(1, len(buffers)):
        prev_track = tracks[index - 1]
        next_track = tracks[index]
        incoming = buffers[index]
        strategy = _resolve_mix_strategy(settings, prev_track, next_track) if ai_precision or filter_mode == "dynamicEq" else None
        plan = plan_transition(prev_track, next_track, {**settings, "crossfade": requested, "autoTransition": auto})
        preview = _transition_preview_for_pair(prev_track, next_track)
        prev_duration_samples = min(_seconds_to_samples(float(prev_track.get("duration") or 0)), rendered.shape[1])
        current_track_start = max(0, rendered.shape[1] - prev_duration_samples)
        if preview:
            rendered = _splice_transition_preview(rendered, incoming, current_track_start, preview)
            continue

        prev_overlap_start = current_track_start + _seconds_to_samples(plan.prev_overlap_start)
        next_overlap_start = _seconds_to_samples(plan.next_intro if strategy == "vocalHandoff" else plan.next_overlap_start)
        if prev_overlap_start >= rendered.shape[1]:
            prev_overlap_start = max(0, rendered.shape[1] - _seconds_to_samples(plan.seconds))
        if next_overlap_start >= incoming.shape[1]:
            next_overlap_start = 0
        samples = min(
            _seconds_to_samples(plan.seconds),
            rendered.shape[1] - prev_overlap_start,
            incoming.shape[1] - next_overlap_start,
        )
        if samples <= 0:
            rendered = np.concatenate([rendered, incoming], axis=1)
            continue

        head = rendered[:, :prev_overlap_start]
        outgoing_tail = rendered[:, prev_overlap_start : prev_overlap_start + samples]
        incoming_head = incoming[:, next_overlap_start : next_overlap_start + samples]
        tail = incoming[:, next_overlap_start + samples :]

        if ai_precision or filter_mode == "dynamicEq":
            overlap = _dynamic_eq_overlap(outgoing_tail, incoming_head, strategy or "bassSwap")
        else:
            if filter_mode == "lowpassSweep":
                outgoing_tail = _sos_filter(outgoing_tail, "lowpass", 1800)
            elif filter_mode == "highpassLift":
                incoming_head = _sos_filter(incoming_head, "highpass", 180)
            fade_out, fade_in = _fade_curves(samples, equal_power=bool(settings.get("equalPowerFade", ai_precision)))
            overlap = outgoing_tail * fade_out + incoming_head * fade_in
        rendered = np.concatenate([head, overlap, tail], axis=1)

    return np.clip(rendered, -1, 1)


def _transition_preview_for_pair(prev_track: dict, next_track: dict) -> dict | None:
    preview = next_track.get("appliedTransitionPreview") or {}
    if not preview.get("url") and not preview.get("audioPath"):
        return None
    if preview.get("outgoingTrackId") and preview.get("outgoingTrackId") != prev_track.get("id"):
        return None
    if preview.get("incomingTrackId") and preview.get("incomingTrackId") != next_track.get("id"):
        return None
    outgoing_time = _preview_cue_time(preview, "outgoingCue", "outgoingExitTime")
    incoming_time = _preview_cue_time(preview, "incomingCue", "incomingEntryTime")
    if outgoing_time is None or incoming_time is None:
        return None
    return preview


def _splice_transition_preview(rendered: np.ndarray, incoming: np.ndarray, current_track_start: int, preview: dict) -> np.ndarray:
    preview_audio = _load_transition_preview_audio(preview)
    outgoing_time = _preview_cue_time(preview, "outgoingCue", "outgoingExitTime")
    incoming_time = _preview_cue_time(preview, "incomingCue", "incomingEntryTime")
    preview_start_time = float(preview.get("previewStartTime") or max(0.0, float(outgoing_time or 0) - 8.0))
    before_seconds = max(0.0, float(outgoing_time or preview_start_time) - preview_start_time)
    preview_start = max(0, min(rendered.shape[1], current_track_start + _seconds_to_samples(preview_start_time)))
    before_samples = min(_seconds_to_samples(before_seconds), preview_audio.shape[1])
    preview_audio = _match_preview_level(rendered, preview_audio, preview_start, before_samples)
    incoming_resume = _seconds_to_samples(float(incoming_time or 0) + preview_audio.shape[1] / SAMPLE_RATE - before_samples / SAMPLE_RATE)
    incoming_resume = max(0, min(incoming.shape[1], incoming_resume))
    head = rendered[:, :preview_start]
    tail = incoming[:, incoming_resume:]
    return np.concatenate([head, preview_audio, tail], axis=1)


def _load_transition_preview_audio(preview: dict) -> np.ndarray:
    path_value = preview.get("audioPath")
    if path_value:
        path = Path(path_value)
    else:
        path = EXPORT_DIR / Path(str(preview.get("url", "")).split("?")[0]).name
    resolved = path.resolve()
    export_root = EXPORT_DIR.resolve()
    if export_root not in resolved.parents and resolved != export_root:
        raise ValueError("Transition preview audio must be in the export directory")
    if not resolved.exists():
        raise ValueError("Transition preview audio is missing; regenerate the seamless preview before export")
    return _load_stereo(resolved)


def _match_preview_level(rendered: np.ndarray, preview_audio: np.ndarray, preview_start: int, before_samples: int) -> np.ndarray:
    reference_samples = min(before_samples, rendered.shape[1] - preview_start, preview_audio.shape[1], SAMPLE_RATE * 2)
    if reference_samples < SAMPLE_RATE // 4:
        return preview_audio
    reference = rendered[:, preview_start : preview_start + reference_samples]
    preview_reference = preview_audio[:, :reference_samples]
    reference_rms = float(np.sqrt(np.mean(reference * reference)) + 1e-9)
    preview_rms = float(np.sqrt(np.mean(preview_reference * preview_reference)) + 1e-9)
    gain_db = max(-6.0, min(6.0, 20 * np.log10(reference_rms / preview_rms)))
    if abs(gain_db) < 0.15:
        return preview_audio
    return np.clip(preview_audio * (10 ** (gain_db / 20)), -1, 1).astype(np.float32)


def _preview_cue_time(preview: dict, cue_key: str, alignment_key: str) -> float | None:
    cue = preview.get(cue_key) or {}
    value = cue.get("time")
    if value is None:
        value = (preview.get("alignment") or {}).get(alignment_key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _seconds_to_samples(seconds: float) -> int:
    return max(0, int(round(seconds * SAMPLE_RATE)))


def _fade_curves(samples: int, equal_power: bool) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 1, samples, dtype=np.float32)
    if equal_power:
        return np.cos(x * np.pi / 2).astype(np.float32), np.sin(x * np.pi / 2).astype(np.float32)
    return (1 - x).astype(np.float32), x.astype(np.float32)


def _dynamic_eq_overlap(outgoing: np.ndarray, incoming: np.ndarray, strategy: str = "bassSwap") -> np.ndarray:
    if strategy == "vocalHandoff":
        return _vocal_handoff_overlap(outgoing, incoming)

    samples = outgoing.shape[1]
    fade_out, fade_in = _fade_curves(samples, equal_power=True)
    x = np.linspace(0, 1, samples, dtype=np.float32)

    prev_low = _sos_filter(outgoing, "lowpass", 220)
    next_low = _sos_filter(incoming, "lowpass", 220)
    prev_high = _sos_filter(outgoing, "highpass", 3200)
    next_high = _sos_filter(incoming, "highpass", 3200)
    prev_mid = outgoing - prev_low - prev_high
    next_mid = incoming - next_low - next_high

    curves = _strategy_curves(strategy, x, fade_out, fade_in)

    overlap = (
        prev_low * curves["prev_low"]
        + next_low * curves["next_low"]
        + prev_mid * curves["prev_mid"]
        + next_mid * curves["next_mid"]
        + prev_high * curves["prev_high"]
        + next_high * curves["next_high"]
    )
    return np.clip(overlap, -1, 1).astype(np.float32)


def _resolve_mix_strategy(settings: dict, prev_track: dict, next_track: dict) -> str:
    selected = settings.get("mixStrategy") or "auto"
    if selected != "auto":
        return selected
    prev_candidates = prev_track.get("transition_candidates") or {}
    next_candidates = next_track.get("transition_candidates") or {}
    prev_vocal = float(prev_candidates.get("outro_vocal_density") or 0)
    next_vocal = float(next_candidates.get("intro_vocal_density") or 0)
    bpm_delta = abs(float(prev_track.get("bpm") or 0) - float(next_track.get("bpm") or 0))
    prev_energy = float(prev_candidates.get("outro_energy") or prev_track.get("energy") or 0)
    next_energy = float(next_candidates.get("intro_energy") or next_track.get("energy") or 0)
    energy_lift = float(next_track.get("energy") or 0) - float(prev_track.get("energy") or 0)
    if bpm_delta > 20:
        return "smooth"
    if next_vocal >= 0.32 and prev_energy >= 0.25 and next_energy >= 0.22 and bpm_delta <= 18:
        return "vocalHandoff"
    if prev_vocal > 0.55 or next_vocal > 0.55:
        return "vocalSafe"
    if bpm_delta <= 4 and energy_lift > 0.08:
        return "bassSwap"
    if bpm_delta > 12:
        return "smooth"
    return "bassSwap"


def _vocal_handoff_overlap(outgoing: np.ndarray, incoming: np.ndarray) -> np.ndarray:
    samples = min(outgoing.shape[1], incoming.shape[1])
    if samples <= 0:
        return incoming

    outgoing = outgoing[:, :samples]
    incoming = incoming[:, :samples]
    x = np.linspace(0, 1, samples, dtype=np.float32)
    fast_entry = _smoothstep(np.clip(x / 0.16, 0, 1))
    late = np.clip((x - 0.68) / 0.32, 0, 1)
    very_late = np.clip((x - 0.78) / 0.22, 0, 1)

    prev_harmonic, prev_percussive = _harmonic_percussive(outgoing)
    next_harmonic, next_percussive = _harmonic_percussive(incoming)
    prev_low = _sos_filter(outgoing, "lowpass", 180)
    next_low = _sos_filter(incoming, "lowpass", 180)
    prev_vocal_region = _sos_filter(prev_harmonic, "highpass", 180)
    next_vocal_region = _sos_filter(next_harmonic, "highpass", 180)

    overlap = (
        prev_percussive * (0.98 - 0.2 * x)
        + prev_low * np.power(1 - x, 1.1) * 0.6
        + prev_vocal_region * np.power(1 - x, 5.0) * 0.22
        + next_vocal_region * fast_entry
        + next_percussive * np.power(late, 1.7) * 0.82
        + next_low * np.power(very_late, 1.8) * 0.88
    )
    return np.clip(overlap, -1, 1).astype(np.float32)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    return (x * x * (3 - 2 * x)).astype(np.float32)


def _harmonic_percussive(buffer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        harmonic, percussive = librosa.effects.hpss(buffer, kernel_size=(31, 31), margin=(1.0, 4.0))
    except Exception:
        low = _sos_filter(buffer, "lowpass", 220)
        high = _sos_filter(buffer, "highpass", 2600)
        harmonic = buffer - high
        percussive = high + low * 0.4
    return np.ascontiguousarray(harmonic, dtype=np.float32), np.ascontiguousarray(percussive, dtype=np.float32)


def _strategy_curves(strategy: str, x: np.ndarray, fade_out: np.ndarray, fade_in: np.ndarray) -> dict[str, np.ndarray]:
    if strategy == "vocalHandoff":
        return {
            "prev_low": (0.85 - 0.45 * x),
            "next_low": np.power(np.clip((x - 0.72) / 0.28, 0, 1), 1.5),
            "prev_mid": np.power(1 - x, 2.4) * 0.45,
            "next_mid": np.power(x, 0.62),
            "prev_high": np.power(1 - x, 1.9) * 0.55,
            "next_high": np.power(x, 0.75),
        }
    if strategy == "vocalSafe":
        return {
            "prev_low": np.power(1 - x, 1.8),
            "next_low": np.power(x, 0.75),
            "prev_mid": np.power(1 - x, 0.75),
            "next_mid": np.power(x, 1.9),
            "prev_high": np.power(1 - x, 0.85),
            "next_high": np.power(x, 1.45),
        }
    if strategy == "smooth":
        return {
            "prev_low": fade_out,
            "next_low": fade_in * 0.9,
            "prev_mid": fade_out,
            "next_mid": np.power(x, 1.25),
            "prev_high": fade_out,
            "next_high": np.power(x, 1.25),
        }
    if strategy == "quickCut":
        return {
            "prev_low": np.power(1 - x, 3.0),
            "next_low": np.power(x, 0.45),
            "prev_mid": np.power(1 - x, 2.0),
            "next_mid": np.power(x, 0.75),
            "prev_high": np.power(1 - x, 1.6),
            "next_high": np.power(x, 0.8),
        }
    return {
        "prev_low": np.power(1 - x, 2.2),
        "next_low": np.sqrt(x),
        "prev_mid": fade_out,
        "next_mid": np.power(x, 1.35),
        "prev_high": np.power(1 - x, 0.85),
        "next_high": np.power(x, 1.15),
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
