from __future__ import annotations

import math

import numpy as np


DEFAULT_PEAK_CEILING = 0.98


def loudness_metrics(buffer: np.ndarray, sample_rate: int) -> dict[str, float]:
    audio = _to_frames_channels(buffer)
    lufs = _integrated_lufs(audio, sample_rate)
    peak = float(np.max(np.abs(audio)) + 1e-12)
    peak_db = 20 * math.log10(peak)
    return {"lufs": round(lufs, 2), "peak_db": round(peak_db, 2)}


def normalize_loudness(
    buffer: np.ndarray,
    sample_rate: int,
    target_lufs: float,
    peak_ceiling: float = DEFAULT_PEAK_CEILING,
) -> np.ndarray:
    audio = _to_frames_channels(buffer)
    current_lufs = _integrated_lufs(audio, sample_rate)
    gain = 10 ** ((target_lufs - current_lufs) / 20)
    out = audio * gain
    peak = float(np.max(np.abs(out)) + 1e-12)
    if peak > peak_ceiling:
        out *= peak_ceiling / peak
    return _restore_layout(out, buffer).astype(np.float32)


def _integrated_lufs(audio: np.ndarray, sample_rate: int) -> float:
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sample_rate)
        value = float(meter.integrated_loudness(audio))
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return _rms_lufs(audio)


def _rms_lufs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
    return 20 * math.log10(rms)


def _to_frames_channels(buffer: np.ndarray) -> np.ndarray:
    audio = np.asarray(buffer, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
        return audio.T
    return audio


def _restore_layout(audio: np.ndarray, original: np.ndarray) -> np.ndarray:
    source = np.asarray(original)
    if source.ndim == 1:
        return np.asarray(audio).reshape(-1)
    if source.shape[0] <= 8 and source.shape[0] < source.shape[1]:
        return np.asarray(audio).T
    return np.asarray(audio)
