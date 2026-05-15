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
        plan = plan_transition(prev_track, next_track, {**settings, "crossfade": requested, "autoTransition": auto})
        prev_duration_samples = min(_seconds_to_samples(float(prev_track.get("duration") or 0)), rendered.shape[1])
        current_track_start = max(0, rendered.shape[1] - prev_duration_samples)
        prev_overlap_start = current_track_start + _seconds_to_samples(plan.prev_overlap_start)
        next_overlap_start = _seconds_to_samples(plan.next_overlap_start)
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
            overlap = _dynamic_eq_overlap(outgoing_tail, incoming_head)
        else:
            if filter_mode == "lowpassSweep":
                outgoing_tail = _sos_filter(outgoing_tail, "lowpass", 1800)
            elif filter_mode == "highpassLift":
                incoming_head = _sos_filter(incoming_head, "highpass", 180)
            fade_out, fade_in = _fade_curves(samples, equal_power=bool(settings.get("equalPowerFade", ai_precision)))
            overlap = outgoing_tail * fade_out + incoming_head * fade_in
        rendered = np.concatenate([head, overlap, tail], axis=1)

    return np.clip(rendered, -1, 1)


def _seconds_to_samples(seconds: float) -> int:
    return max(0, int(round(seconds * SAMPLE_RATE)))


def _fade_curves(samples: int, equal_power: bool) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 1, samples, dtype=np.float32)
    if equal_power:
        return np.cos(x * np.pi / 2).astype(np.float32), np.sin(x * np.pi / 2).astype(np.float32)
    return (1 - x).astype(np.float32), x.astype(np.float32)


def _dynamic_eq_overlap(outgoing: np.ndarray, incoming: np.ndarray) -> np.ndarray:
    samples = outgoing.shape[1]
    fade_out, fade_in = _fade_curves(samples, equal_power=True)
    x = np.linspace(0, 1, samples, dtype=np.float32)

    prev_low = _sos_filter(outgoing, "lowpass", 220)
    next_low = _sos_filter(incoming, "lowpass", 220)
    prev_high = _sos_filter(outgoing, "highpass", 3200)
    next_high = _sos_filter(incoming, "highpass", 3200)
    prev_mid = outgoing - prev_low - prev_high
    next_mid = incoming - next_low - next_high

    # DJ-style frequency avoidance: let the incoming kick/bass establish early,
    # while vocals/synths enter more slowly to reduce masking.
    prev_low_curve = np.power(1 - x, 2.2)
    next_low_curve = np.sqrt(x)
    prev_mid_curve = fade_out
    next_mid_curve = np.power(x, 1.35)
    prev_high_curve = np.power(1 - x, 0.85)
    next_high_curve = np.power(x, 1.15)

    overlap = (
        prev_low * prev_low_curve
        + next_low * next_low_curve
        + prev_mid * prev_mid_curve
        + next_mid * next_mid_curve
        + prev_high * prev_high_curve
        + next_high * next_high_curve
    )
    return np.clip(overlap, -1, 1).astype(np.float32)


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
