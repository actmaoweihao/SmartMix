from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from scipy import signal

from .storage import EXPORT_DIR


SAMPLE_RATE = 44100


def render_mix(tracks: list[dict], settings: dict, fmt: str) -> Path:
    if not tracks:
        raise ValueError("没有可导出的曲目")

    buffers = [_load_stereo(Path(track["path"])) for track in tracks]
    if settings.get("beatSync"):
        buffers = _beat_sync(buffers, tracks)

    buffers = [_apply_static_eq(buffer, settings.get("eq", {})) for buffer in buffers]
    mix = _crossfade(buffers, tracks, settings)
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
        rate = np.clip(bpm / target, 0.88, 1.12)
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


def _sos_filter(buffer: np.ndarray, kind: str, freq: float) -> np.ndarray:
    sos = signal.butter(2, freq, btype=kind, fs=SAMPLE_RATE, output="sos")
    return signal.sosfilt(sos, buffer, axis=1).astype(np.float32)


def _crossfade(buffers: list[np.ndarray], tracks: list[dict], settings: dict) -> np.ndarray:
    requested = float(settings.get("crossfade", 8))
    auto = bool(settings.get("autoTransition", True))
    filter_mode = settings.get("filterMode", "none")
    rendered = buffers[0]

    for index in range(1, len(buffers)):
      prev_track = tracks[index - 1]
      next_track = tracks[index]
      incoming = buffers[index]
      transition = _transition_seconds(prev_track, next_track, requested, auto)
      samples = min(
          int(transition * SAMPLE_RATE),
          rendered.shape[1] // 2,
          incoming.shape[1] // 2,
      )
      if samples <= 0:
          rendered = np.concatenate([rendered, incoming], axis=1)
          continue

      head = rendered[:, :-samples]
      outgoing_tail = rendered[:, -samples:]
      incoming_head = incoming[:, :samples]
      tail = incoming[:, samples:]

      if filter_mode == "lowpassSweep":
          outgoing_tail = _sos_filter(outgoing_tail, "lowpass", 1800)
      elif filter_mode == "highpassLift":
          incoming_head = _sos_filter(incoming_head, "highpass", 180)

      fade_out = np.linspace(1, 0, samples, dtype=np.float32)
      fade_in = np.linspace(0, 1, samples, dtype=np.float32)
      overlap = outgoing_tail * fade_out + incoming_head * fade_in
      rendered = np.concatenate([head, overlap, tail], axis=1)

    return np.clip(rendered, -1, 1)


def _transition_seconds(prev_track: dict, next_track: dict, requested: float, auto: bool) -> float:
    prev_duration = float(prev_track.get("duration") or 0)
    next_duration = float(next_track.get("duration") or 0)
    max_by_length = max(0.5, min(prev_duration, next_duration) * 0.35)
    prev_out = prev_track.get("outroPoint")
    next_in = next_track.get("introPoint")
    if isinstance(prev_out, (int, float)) and isinstance(next_in, (int, float)):
        handle_value = min(max(0.5, prev_duration - float(prev_out)), max(0.5, float(next_in)))
        requested = min(requested, handle_value)
    if auto:
        structural = max(2, min(requested, float(prev_track.get("outro_low") or 0) + float(next_track.get("intro_low") or 0) + 2))
        return min(structural, max_by_length)
    return min(requested, max_by_length)


def _convert_to_mp3(wav_path: Path) -> Path:
    mp3_path = wav_path.with_suffix(".mp3")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        segment = AudioSegment.from_wav(wav_path)
        segment.export(mp3_path, format="mp3", bitrate="192k", parameters=["-ac", "2"])
    except Exception:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return mp3_path

