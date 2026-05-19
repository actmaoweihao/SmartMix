#!/usr/bin/env python3
"""Reference-guided automatic mixing MVP.

This is a rule-based DSP prototype inspired by Diff-MST's idea of steering
interpretable mixing-console parameters from reference audio features. It does
not train, download, or run a neural network.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy import signal

try:  # Optional but recommended.
    import librosa
except Exception:  # pragma: no cover - exercised only in minimal envs.
    librosa = None

try:
    import pyloudnorm as pyln
except Exception:  # pragma: no cover - exercised only in minimal envs.
    pyln = None

try:
    from pedalboard import Compressor, Gain, Limiter, Pedalboard, Reverb

    PEDALBOARD_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only in minimal envs.
    PEDALBOARD_AVAILABLE = False


EPS = 1e-9
BANDS = {
    "sub": (20.0, 60.0),
    "bass": (60.0, 250.0),
    "low_mid": (250.0, 500.0),
    "mid": (500.0, 2000.0),
    "high_mid": (2000.0, 6000.0),
    "high": (6000.0, 16000.0),
}
AUDIO_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".ogg", ".mp3", ".m4a"}


class AutoMixError(RuntimeError):
    """User-facing error raised by the CLI."""


@dataclass
class LoadedStem:
    name: str
    path: Path
    audio: np.ndarray
    sr: int


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def amp_to_db(amp: float) -> float:
    return float(20.0 * np.log10(max(float(amp), EPS)))


def as_float32(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(y, -4.0, 4.0).astype(np.float32, copy=False)


def ensure_2d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2:
        raise AutoMixError(f"Expected audio shape (samples, channels), got {y.shape}")
    if y.shape[0] == 0:
        raise AutoMixError("Audio file contains no samples")
    return as_float32(y)


def ensure_stereo(y: np.ndarray) -> np.ndarray:
    y = ensure_2d(y)
    if y.shape[1] == 1:
        return np.repeat(y, 2, axis=1)
    if y.shape[1] > 2:
        return y[:, :2]
    return y


def mono_mix(y: np.ndarray) -> np.ndarray:
    y = ensure_2d(y)
    return as_float32(np.mean(y, axis=1))


def load_audio(path: str | Path, target_sr: int = 44100) -> Tuple[np.ndarray, int]:
    """Read mono/stereo audio, resample, and return (samples, channels)."""
    path = Path(path)
    if not path.exists():
        raise AutoMixError(f"Audio file not found: {path}")
    try:
        y, sr = sf.read(path, always_2d=True, dtype="float32")
    except Exception as exc:
        raise AutoMixError(f"Could not read audio file {path}: {exc}") from exc

    y = ensure_2d(y)
    if y.shape[1] > 2:
        y = y[:, :2]
    if sr <= 0:
        raise AutoMixError(f"Invalid sample rate in {path}: {sr}")
    if target_sr and sr != target_sr:
        if librosa is None:
            raise AutoMixError(
                f"{path} is {sr} Hz, but librosa is unavailable for resampling"
            )
        try:
            y = librosa.resample(y.T, orig_sr=sr, target_sr=target_sr).T
        except Exception as exc:
            raise AutoMixError(f"Could not resample {path}: {exc}") from exc
        sr = target_sr
    return as_float32(y), int(sr)


def approximate_lra(y: np.ndarray, sr: int) -> Optional[float]:
    """Small loudness range approximation using 3-second RMS blocks."""
    mono = mono_mix(y)
    block = max(1, int(sr * 3.0))
    hop = max(1, int(sr * 1.0))
    if len(mono) < block:
        return 0.0
    values = []
    for start in range(0, len(mono) - block + 1, hop):
        chunk = mono[start : start + block]
        rms = float(np.sqrt(np.mean(chunk * chunk) + EPS))
        values.append(amp_to_db(rms))
    if not values:
        return 0.0
    return float(np.percentile(values, 95) - np.percentile(values, 10))


def measure_lufs(y: np.ndarray, sr: int) -> Tuple[float, Optional[float]]:
    if pyln is None:
        rms = float(np.sqrt(np.mean(np.square(mono_mix(y))) + EPS))
        return amp_to_db(rms) - 0.691, approximate_lra(y, sr)
    try:
        stereo = ensure_stereo(y)
        meter = pyln.Meter(sr)
        loudness = float(meter.integrated_loudness(stereo))
        if not np.isfinite(loudness):
            loudness = -70.0
        return loudness, approximate_lra(y, sr)
    except Exception:
        rms = float(np.sqrt(np.mean(np.square(mono_mix(y))) + EPS))
        return amp_to_db(rms) - 0.691, approximate_lra(y, sr)


def calculate_band_energy(y: np.ndarray, sr: int) -> Dict[str, float]:
    mono = mono_mix(y)
    if len(mono) < 32:
        return {band: 0.0 for band in BANDS}
    nperseg = min(4096, max(256, 2 ** int(np.floor(np.log2(len(mono))))))
    noverlap = min(nperseg // 2, nperseg - 1)
    freqs, _, zxx = signal.stft(
        mono,
        fs=sr,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    power = np.mean(np.abs(zxx) ** 2, axis=1)
    total = float(np.sum(power[(freqs >= 20.0) & (freqs <= min(16000.0, sr / 2))]) + EPS)
    result: Dict[str, float] = {}
    for name, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < min(high, sr / 2))
        result[name] = float(np.sum(power[mask]) / total) if np.any(mask) else 0.0
    return result


def spectral_fallback(y: np.ndarray, sr: int) -> Tuple[float, float]:
    mono = mono_mix(y)
    if len(mono) < 32:
        return 0.0, 0.0
    window = np.hanning(len(mono))
    spectrum = np.abs(np.fft.rfft(mono * window)) ** 2
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / sr)
    total = float(np.sum(spectrum) + EPS)
    centroid = float(np.sum(freqs * spectrum) / total)
    cumulative = np.cumsum(spectrum)
    idx = int(np.searchsorted(cumulative, 0.85 * cumulative[-1]))
    rolloff = float(freqs[min(idx, len(freqs) - 1)])
    return centroid, rolloff


def stereo_metrics(y: np.ndarray) -> Tuple[float, float]:
    y = ensure_2d(y)
    if y.shape[1] < 2:
        return 0.0, 0.0
    stereo = ensure_stereo(y)
    left = stereo[:, 0]
    right = stereo[:, 1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    mid_energy = float(np.mean(mid * mid) + EPS)
    side_energy = float(np.mean(side * side))
    width = float(np.sqrt(side_energy / mid_energy))
    l_rms = float(np.sqrt(np.mean(left * left) + EPS))
    r_rms = float(np.sqrt(np.mean(right * right) + EPS))
    imbalance = amp_to_db(l_rms) - amp_to_db(r_rms)
    return clamp(width, 0.0, 2.5), float(imbalance)


def analyze_audio(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Analyze loudness, dynamics, spectrum, and stereo features."""
    y = ensure_2d(y)
    mono = mono_mix(y)
    duration = float(len(y) / sr)
    peak = float(np.max(np.abs(y)) if len(y) else 0.0)
    rms = float(np.sqrt(np.mean(mono * mono) + EPS))
    peak_dbfs = amp_to_db(peak)
    rms_db = amp_to_db(rms)
    integrated_lufs, lra = measure_lufs(y, sr)
    crest = float(peak_dbfs - rms_db)

    if librosa is not None and len(mono) >= 64:
        try:
            centroid = float(np.mean(librosa.feature.spectral_centroid(y=mono, sr=sr)))
            rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=mono, sr=sr)))
        except Exception:
            centroid, rolloff = spectral_fallback(y, sr)
    else:
        centroid, rolloff = spectral_fallback(y, sr)

    width, imbalance = stereo_metrics(y)
    result = {
        "duration": duration,
        "peak_dbfs": peak_dbfs,
        "rms_db": rms_db,
        "integrated_lufs": integrated_lufs,
        "lra": lra,
        "crest_factor_db": crest,
        "spectral_centroid_mean": centroid,
        "spectral_rolloff_mean": rolloff,
        "band_energy": calculate_band_energy(y, sr),
        "stereo_width": width,
        "stereo_imbalance": imbalance,
    }
    return json_safe(result)


def infer_role(name: str) -> str:
    n = name.lower()
    if any(token in n for token in ("vocal", "vox", "sing")):
        return "vocal"
    if any(token in n for token in ("drums", "drum", "beat", "kick", "snare")):
        return "drums"
    if "bass" in n:
        return "bass"
    if any(token in n for token in ("guitar", "gtr")):
        return "guitar"
    if any(token in n for token in ("keys", "piano", "synth", "pad")):
        return "harmonic"
    return "other"


def analyze_stems(stems: Iterable[LoadedStem | Dict[str, Any]], sr: Optional[int] = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for stem in stems:
        if isinstance(stem, LoadedStem):
            name = stem.name
            audio = stem.audio
            sample_rate = stem.sr
            path = str(stem.path)
        else:
            name = str(stem["name"])
            audio = stem["audio"]
            sample_rate = int(stem.get("sr", sr or 44100))
            path = str(stem.get("path", ""))
        features = analyze_audio(audio, sample_rate)
        features["role"] = infer_role(name)
        if path:
            features["path"] = path
        result[name] = features
    return result


def role_target_rms(role: str) -> float:
    return {
        "vocal": -18.0,
        "drums": -19.0,
        "bass": -20.0,
        "guitar": -23.0,
        "harmonic": -24.0,
        "other": -26.0,
    }.get(role, -26.0)


def role_compressor(role: str, ref_crest: float) -> Dict[str, float]:
    dense_ref = ref_crest < 10.0
    if role == "vocal":
        return {
            "threshold_db": -24.0 if dense_ref else -22.0,
            "ratio": 3.5 if dense_ref else 2.6,
            "attack_ms": 8.0,
            "release_ms": 90.0,
        }
    if role == "drums":
        return {
            "threshold_db": -18.0 if dense_ref else -15.0,
            "ratio": 3.0 if dense_ref else 2.0,
            "attack_ms": 18.0,
            "release_ms": 120.0,
        }
    if role == "bass":
        return {
            "threshold_db": -22.0 if dense_ref else -20.0,
            "ratio": 3.0 if dense_ref else 2.4,
            "attack_ms": 25.0,
            "release_ms": 160.0,
        }
    return {
        "threshold_db": -20.0 if dense_ref else -18.0,
        "ratio": 2.0 if dense_ref else 1.5,
        "attack_ms": 20.0,
        "release_ms": 140.0,
    }


def derive_mix_params(
    reference_features: Dict[str, Any],
    stem_features: Dict[str, Any],
    style: str = "auto",
) -> Dict[str, Any]:
    ref_bands = reference_features.get("band_energy", {})
    ref_sub_bass = float(ref_bands.get("sub", 0.0) + ref_bands.get("bass", 0.0))
    ref_low_mid = float(ref_bands.get("low_mid", 0.0))
    ref_air = float(ref_bands.get("high_mid", 0.0) + ref_bands.get("high", 0.0))
    ref_width = float(reference_features.get("stereo_width", 0.0))
    ref_crest = float(reference_features.get("crest_factor_db", 12.0))
    target_lufs = clamp(float(reference_features.get("integrated_lufs", -14.0)), -18.0, -9.0)

    sorted_names = sorted(stem_features.keys())
    lateral_index = 0
    stems: Dict[str, Any] = {}
    for name in sorted_names:
        feats = stem_features[name]
        role = feats.get("role") or infer_role(name)
        rms_db = float(feats.get("rms_db", -40.0))
        gain_db = clamp(role_target_rms(role) - rms_db, -18.0, 12.0)

        highpass = {
            "vocal": 90.0,
            "drums": 30.0,
            "bass": 30.0,
            "guitar": 90.0,
            "harmonic": 120.0,
            "other": 130.0,
        }.get(role, 120.0)
        if ref_low_mid > 0.22 and role not in ("bass", "drums"):
            highpass += 35.0
        if role == "bass" and ref_sub_bass > 0.32:
            gain_db += 1.5
        if role == "drums" and ref_sub_bass > 0.32:
            gain_db += 0.8
        if role not in ("bass", "drums") and ref_sub_bass > 0.35:
            highpass += 25.0

        low_shelf = 0.0
        if role == "bass":
            low_shelf = 1.5 if ref_sub_bass > 0.28 else -0.8
        elif role == "drums":
            low_shelf = 0.8 if ref_sub_bass > 0.30 else 0.0
        elif ref_low_mid > 0.22:
            low_shelf = -1.5

        presence = 0.0
        high_shelf = 0.0
        if ref_air > 0.18:
            if role in ("vocal", "guitar", "harmonic"):
                presence = 1.5
                high_shelf = 1.2
            elif role == "other":
                high_shelf = 0.6
        elif ref_air < 0.08:
            high_shelf = -1.5 if role != "bass" else -0.5

        if role in ("vocal", "bass", "drums"):
            pan = 0.0
        else:
            base = 0.55 if ref_width > 0.45 else 0.35
            pan = base if lateral_index % 2 else -base
            if role == "other":
                pan *= 0.7
            lateral_index += 1

        reverb_send = estimate_reverb_send(role, style, ref_width, ref_air)
        stems[name] = {
            "role": role,
            "gain_db": round(float(gain_db), 3),
            "pan": round(float(pan), 3),
            "eq": {
                "highpass_hz": round(float(highpass), 2),
                "low_shelf_db": round(float(low_shelf), 3),
                "presence_db": round(float(presence), 3),
                "high_shelf_db": round(float(high_shelf), 3),
            },
            "compressor": role_compressor(role, ref_crest),
            "reverb_send": round(float(reverb_send), 3),
        }

    master_ratio = 2.2 if ref_crest < 9.5 else (1.35 if ref_crest > 15.0 else 1.7)
    master_threshold = -18.0 if ref_crest < 9.5 else (-12.0 if ref_crest > 15.0 else -15.0)
    params = {
        "stems": stems,
        "master": {
            "target_lufs": round(float(target_lufs), 3),
            "master_gain_db": 0.0,
            "master_compressor": {
                "threshold_db": master_threshold,
                "ratio": master_ratio,
                "attack_ms": 25.0,
                "release_ms": 180.0,
            },
            "limiter_threshold_db": -1.0,
            "stereo_width": round(clamp(0.85 + ref_width, 0.85, 1.55), 3),
            "brightness_shift_db": 0.0,
            "bass_shift_db": 0.0,
        },
    }
    return params


def estimate_reverb_send(role: str, style: str, ref_width: float, ref_air: float) -> float:
    if style in ("lofi", "cinematic"):
        base = 0.22
    elif style in ("club", "edm"):
        base = 0.08
    elif style in ("pop", "modern"):
        base = 0.11
    else:
        base = 0.10 + 0.12 * clamp(ref_width, 0.0, 1.0) + 0.05 * clamp(ref_air, 0.0, 0.3)
    if role == "vocal":
        return clamp(base + 0.04, 0.04, 0.30)
    if role in ("bass",):
        return clamp(base * 0.20, 0.0, 0.06)
    if role == "drums":
        return clamp(base * 0.35, 0.0, 0.09)
    return clamp(base, 0.03, 0.28)


def rbj_filter(
    y: np.ndarray,
    sr: int,
    kind: str,
    freq: float,
    gain_db: float = 0.0,
    q: float = 0.707,
) -> np.ndarray:
    freq = clamp(freq, 20.0, sr * 0.45)
    a = math.sqrt(db_to_amp(gain_db))
    omega = 2.0 * math.pi * freq / sr
    sin_w = math.sin(omega)
    cos_w = math.cos(omega)
    alpha = sin_w / (2.0 * q)

    if kind == "peaking":
        b0 = 1 + alpha * a
        b1 = -2 * cos_w
        b2 = 1 - alpha * a
        a0 = 1 + alpha / a
        a1 = -2 * cos_w
        a2 = 1 - alpha / a
    elif kind == "lowshelf":
        sqrt_a = math.sqrt(a)
        two_sqrt_a_alpha = 2 * sqrt_a * alpha
        b0 = a * ((a + 1) - (a - 1) * cos_w + two_sqrt_a_alpha)
        b1 = 2 * a * ((a - 1) - (a + 1) * cos_w)
        b2 = a * ((a + 1) - (a - 1) * cos_w - two_sqrt_a_alpha)
        a0 = (a + 1) + (a - 1) * cos_w + two_sqrt_a_alpha
        a1 = -2 * ((a - 1) + (a + 1) * cos_w)
        a2 = (a + 1) + (a - 1) * cos_w - two_sqrt_a_alpha
    elif kind == "highshelf":
        sqrt_a = math.sqrt(a)
        two_sqrt_a_alpha = 2 * sqrt_a * alpha
        b0 = a * ((a + 1) + (a - 1) * cos_w + two_sqrt_a_alpha)
        b1 = -2 * a * ((a - 1) + (a + 1) * cos_w)
        b2 = a * ((a + 1) + (a - 1) * cos_w - two_sqrt_a_alpha)
        a0 = (a + 1) - (a - 1) * cos_w + two_sqrt_a_alpha
        a1 = 2 * ((a - 1) - (a + 1) * cos_w)
        a2 = (a + 1) - (a - 1) * cos_w - two_sqrt_a_alpha
    else:
        raise ValueError(f"Unsupported RBJ filter kind: {kind}")

    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a_coeff = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return as_float32(signal.lfilter(b, a_coeff, y, axis=0))


def highpass(y: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0.0:
        return y
    cutoff_hz = clamp(cutoff_hz, 10.0, sr * 0.45)
    sos = signal.butter(2, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return as_float32(signal.sosfilt(sos, y, axis=0))


def apply_eq(y: np.ndarray, sr: int, eq: Dict[str, float]) -> np.ndarray:
    out = ensure_stereo(y)
    out = highpass(out, sr, float(eq.get("highpass_hz", 0.0)))
    low = float(eq.get("low_shelf_db", 0.0))
    pres = float(eq.get("presence_db", 0.0))
    high = float(eq.get("high_shelf_db", 0.0))
    if abs(low) > 0.05:
        out = rbj_filter(out, sr, "lowshelf", 110.0, low, q=0.707)
    if abs(pres) > 0.05:
        out = rbj_filter(out, sr, "peaking", 3500.0, pres, q=0.9)
    if abs(high) > 0.05:
        out = rbj_filter(out, sr, "highshelf", 8500.0, high, q=0.707)
    return as_float32(out)


def soft_compressor(y: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    threshold = db_to_amp(float(params.get("threshold_db", -18.0)))
    ratio = max(1.0, float(params.get("ratio", 1.0)))
    if ratio <= 1.01:
        return as_float32(y)
    level = np.max(np.abs(y), axis=1, keepdims=True)
    over = np.maximum(level - threshold, 0.0)
    compressed = threshold + over / ratio
    gain = np.where(level > threshold, compressed / (level + EPS), 1.0)
    # A light smoothing pass keeps the MVP compressor stable and inexpensive.
    alpha = 0.01
    smoothed = np.empty_like(gain)
    current = 1.0
    for i, value in enumerate(gain[:, 0]):
        current = (1.0 - alpha) * current + alpha * float(value)
        smoothed[i, 0] = current
    return as_float32(y * smoothed)


def apply_gain_pedalboard(y: np.ndarray, sr: int, gain_db: float) -> np.ndarray:
    if not PEDALBOARD_AVAILABLE:
        return as_float32(y * db_to_amp(gain_db))
    board = Pedalboard([Gain(gain_db=gain_db)])
    processed = board(y.T, sr).T
    return as_float32(processed)


def apply_compressor_pedalboard(y: np.ndarray, sr: int, params: Dict[str, float]) -> np.ndarray:
    if not PEDALBOARD_AVAILABLE:
        return soft_compressor(y, params)
    try:
        board = Pedalboard(
            [
                Compressor(
                    threshold_db=float(params.get("threshold_db", -18.0)),
                    ratio=float(params.get("ratio", 2.0)),
                    attack_ms=float(params.get("attack_ms", 20.0)),
                    release_ms=float(params.get("release_ms", 120.0)),
                )
            ]
        )
        return as_float32(board(y.T, sr).T)
    except Exception:
        return soft_compressor(y, params)


def simple_reverb(y: np.ndarray, sr: int) -> np.ndarray:
    out = np.zeros_like(y)
    taps = [
        (0.029, 0.35),
        (0.043, 0.25),
        (0.071, 0.18),
        (0.113, 0.12),
        (0.173, 0.08),
    ]
    for seconds, gain in taps:
        delay = int(seconds * sr)
        if delay < len(y):
            out[delay:] += y[:-delay] * gain
    return as_float32(out)


def apply_reverb_send(y: np.ndarray, sr: int, send: float) -> np.ndarray:
    send = clamp(send, 0.0, 1.0)
    if send <= 0.001:
        return as_float32(y)
    wet: np.ndarray
    if PEDALBOARD_AVAILABLE:
        try:
            board = Pedalboard(
                [Reverb(room_size=0.45, damping=0.45, wet_level=1.0, dry_level=0.0)]
            )
            wet = as_float32(board(y.T, sr).T)
        except Exception:
            wet = simple_reverb(y, sr)
    else:
        wet = simple_reverb(y, sr)
    return as_float32(y * (1.0 - send) + wet * send)


def pan_stereo(y: np.ndarray, pan: float) -> np.ndarray:
    """Constant-power pan. pan=-1 left, 0 center, +1 right."""
    stereo = ensure_stereo(y)
    pan = clamp(pan, -1.0, 1.0)
    angle = (pan + 1.0) * math.pi / 4.0
    left_gain = math.cos(angle) * math.sqrt(2.0)
    right_gain = math.sin(angle) * math.sqrt(2.0)
    mono = mono_mix(stereo)
    return as_float32(np.column_stack([mono * left_gain, mono * right_gain]))


def peak_cap(y: np.ndarray, ceiling_db: float) -> np.ndarray:
    peak = float(np.max(np.abs(y)) + EPS)
    ceiling = db_to_amp(ceiling_db)
    if peak > ceiling:
        y = y * (ceiling / peak)
    return as_float32(y)


def process_stem(y: np.ndarray, sr: int, params: Dict[str, Any]) -> np.ndarray:
    out = ensure_stereo(y)
    out = apply_eq(out, sr, params.get("eq", {}))
    out = apply_gain_pedalboard(out, sr, float(params.get("gain_db", 0.0)))
    out = apply_compressor_pedalboard(out, sr, params.get("compressor", {}))
    out = apply_reverb_send(out, sr, float(params.get("reverb_send", 0.0)))
    out = pan_stereo(out, float(params.get("pan", 0.0)))
    return peak_cap(out, -6.0)


def sum_mix(processed_stems: Iterable[np.ndarray]) -> np.ndarray:
    stems = [ensure_stereo(y) for y in processed_stems]
    if not stems:
        raise AutoMixError("No stems were processed")
    min_len = min(len(y) for y in stems)
    mix = np.zeros((min_len, 2), dtype=np.float32)
    for y in stems:
        mix += y[:min_len]
    return peak_cap(mix, -3.0)


def stereo_width_process(y: np.ndarray, width_factor: float) -> np.ndarray:
    stereo = ensure_stereo(y)
    width_factor = clamp(width_factor, 0.0, 2.0)
    left = stereo[:, 0]
    right = stereo[:, 1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right) * width_factor
    return as_float32(np.column_stack([mid + side, mid - side]))


def normalize_loudness(y: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    target_lufs = clamp(target_lufs, -24.0, -8.0)
    if pyln is None:
        current, _ = measure_lufs(y, sr)
        gain = clamp(target_lufs - current, -18.0, 18.0)
        return as_float32(y * db_to_amp(gain))
    try:
        meter = pyln.Meter(sr)
        current = float(meter.integrated_loudness(ensure_stereo(y)))
        if not np.isfinite(current):
            return as_float32(y)
        gain = clamp(target_lufs - current, -18.0, 18.0)
        return as_float32(y * db_to_amp(gain))
    except Exception:
        return as_float32(y)


def apply_limiter(y: np.ndarray, sr: int, threshold_db: float) -> np.ndarray:
    if PEDALBOARD_AVAILABLE:
        try:
            board = Pedalboard([Limiter(threshold_db=threshold_db, release_ms=80.0)])
            return peak_cap(as_float32(board(y.T, sr).T), threshold_db)
        except Exception:
            pass
    return peak_cap(np.tanh(y), threshold_db)


def master_process(mix: np.ndarray, sr: int, master_params: Dict[str, Any]) -> np.ndarray:
    out = ensure_stereo(mix)
    out = apply_gain_pedalboard(out, sr, float(master_params.get("master_gain_db", 0.0)))

    brightness = float(master_params.get("brightness_shift_db", 0.0))
    bass = float(master_params.get("bass_shift_db", 0.0))
    if abs(bass) > 0.05:
        out = rbj_filter(out, sr, "lowshelf", 120.0, bass, q=0.707)
    if abs(brightness) > 0.05:
        out = rbj_filter(out, sr, "highshelf", 8000.0, brightness, q=0.707)

    out = apply_compressor_pedalboard(out, sr, master_params.get("master_compressor", {}))
    out = stereo_width_process(out, float(master_params.get("stereo_width", 1.0)))
    out = normalize_loudness(out, sr, float(master_params.get("target_lufs", -14.0)))
    out = apply_limiter(out, sr, float(master_params.get("limiter_threshold_db", -1.0)))
    return as_float32(out)


def feature_distance(output_features: Dict[str, Any], reference_features: Dict[str, Any]) -> float:
    lufs_diff = float(output_features.get("integrated_lufs", -70.0)) - float(
        reference_features.get("integrated_lufs", -70.0)
    )
    crest_diff = float(output_features.get("crest_factor_db", 0.0)) - float(
        reference_features.get("crest_factor_db", 0.0)
    )
    out_bands = output_features.get("band_energy", {})
    ref_bands = reference_features.get("band_energy", {})
    band_distance = sum(
        abs(float(out_bands.get(name, 0.0)) - float(ref_bands.get(name, 0.0)))
        for name in BANDS
    )
    width_diff = float(output_features.get("stereo_width", 0.0)) - float(
        reference_features.get("stereo_width", 0.0)
    )
    return float(
        1.0 * abs(lufs_diff)
        + 0.8 * abs(crest_diff)
        + 1.2 * band_distance
        + 0.6 * abs(width_diff)
    )


def render_mix(stems: List[LoadedStem], sr: int, params: Dict[str, Any]) -> np.ndarray:
    processed = []
    for stem in stems:
        stem_params = params["stems"].get(stem.name)
        if stem_params is None:
            continue
        processed.append(process_stem(stem.audio, sr, stem_params))
    return master_process(sum_mix(processed), sr, params["master"])


def optimize_params(
    stems: List[LoadedStem],
    sr: int,
    params: Dict[str, Any],
    reference_features: Dict[str, Any],
    seconds: float = 30.0,
    trials: int = 24,
) -> Tuple[Dict[str, Any], float, float]:
    preview_len = min(min(len(stem.audio) for stem in stems), int(seconds * sr))
    preview_stems = [
        LoadedStem(stem.name, stem.path, stem.audio[:preview_len], stem.sr) for stem in stems
    ]

    baseline = render_mix(preview_stems, sr, copy.deepcopy(params))
    baseline_features = analyze_audio(baseline, sr)
    best_distance = feature_distance(baseline_features, reference_features)
    before_distance = best_distance
    best_params = copy.deepcopy(params)

    grid = [
        {
            "master_gain_db": gain,
            "compression_shift": comp,
            "stereo_width": width,
            "brightness_shift_db": bright,
            "bass_shift_db": bass,
        }
        for gain in (-1.5, 0.0, 1.5)
        for comp in (-0.25, 0.0, 0.35)
        for width in (0.85, 1.0, 1.2)
        for bright in (-1.5, 0.0, 1.5)
        for bass in (-1.5, 0.0, 1.5)
    ]
    random.Random(7).shuffle(grid)

    for candidate in grid[:trials]:
        trial_params = copy.deepcopy(params)
        master = trial_params["master"]
        master["master_gain_db"] = candidate["master_gain_db"]
        master["brightness_shift_db"] = candidate["brightness_shift_db"]
        master["bass_shift_db"] = candidate["bass_shift_db"]
        master["stereo_width"] = clamp(
            float(params["master"].get("stereo_width", 1.0)) * candidate["stereo_width"],
            0.75,
            1.8,
        )
        comp = master.get("master_compressor", {})
        comp["ratio"] = clamp(float(comp.get("ratio", 1.7)) + candidate["compression_shift"], 1.1, 3.2)
        comp["threshold_db"] = float(comp.get("threshold_db", -15.0)) - 2.0 * max(
            candidate["compression_shift"], 0.0
        )

        rendered = render_mix(preview_stems, sr, trial_params)
        features = analyze_audio(rendered, sr)
        distance = feature_distance(features, reference_features)
        if distance < best_distance:
            best_distance = distance
            best_params = trial_params
    return best_params, float(before_distance), float(best_distance)


def find_stem_files(stems_dir: Path) -> List[Path]:
    if not stems_dir.exists() or not stems_dir.is_dir():
        raise AutoMixError(f"Stems directory not found: {stems_dir}")
    paths = sorted(
        p for p in stems_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not paths:
        raise AutoMixError(f"No audio stems found in {stems_dir}")
    return paths


def load_stems(stems_dir: Path, target_sr: int) -> Tuple[List[LoadedStem], int]:
    loaded: List[LoadedStem] = []
    for path in find_stem_files(stems_dir):
        audio, sr = load_audio(path, target_sr)
        loaded.append(LoadedStem(path.stem, path, audio, sr))
    min_len = min(len(stem.audio) for stem in loaded)
    if min_len <= 0:
        raise AutoMixError("All stems are empty")
    loaded = [
        LoadedStem(stem.name, stem.path, stem.audio[:min_len], stem.sr) for stem in loaded
    ]
    return loaded, int(loaded[0].sr)


def write_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    return value


def build_warnings() -> List[str]:
    warnings = []
    if librosa is None:
        warnings.append("librosa is unavailable; resampling and some spectral features are limited.")
    if pyln is None:
        warnings.append("pyloudnorm is unavailable; LUFS is approximated from RMS.")
    if not PEDALBOARD_AVAILABLE:
        warnings.append("pedalboard is unavailable; using scipy/numpy fallback DSP.")
    warnings.append("TODO: add chunked processing for very large sessions.")
    return warnings


def run_cli(args: argparse.Namespace) -> Dict[str, Any]:
    stems_dir = Path(args.stems)
    reference_path = Path(args.reference)
    output_path = Path(args.out)
    report_path = Path(args.report) if args.report else output_path.with_name("mix_report.json")

    reference_audio, ref_sr = load_audio(reference_path, args.sample_rate)
    stems, sr = load_stems(stems_dir, args.sample_rate or ref_sr)
    if sr != ref_sr:
        reference_audio, ref_sr = load_audio(reference_path, sr)

    reference_features = analyze_audio(reference_audio, ref_sr)
    stem_features = analyze_stems(stems, sr)
    params = derive_mix_params(reference_features, stem_features, style=args.style)

    before_distance = None
    after_distance = None
    if args.optimize:
        params, before_distance, after_distance = optimize_params(
            stems,
            sr,
            params,
            reference_features,
            seconds=args.optimize_seconds,
            trials=args.optimize_trials,
        )

    mix = render_mix(stems, sr, params)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, mix, sr, subtype="PCM_24")
    final_features = analyze_audio(mix, sr)

    report = {
        "reference": str(reference_path),
        "stems_dir": str(stems_dir),
        "output": str(output_path),
        "sample_rate": sr,
        "style": args.style,
        "reference_features": reference_features,
        "stem_features": stem_features,
        "derived_params": params,
        "final_features": final_features,
        "feature_distance_before": before_distance,
        "feature_distance_after": after_distance
        if after_distance is not None
        else feature_distance(final_features, reference_features),
        "warnings": build_warnings(),
    }
    write_report(report_path, report)
    return report


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reference-guided automatic mixing MVP")
    parser.add_argument("--stems", required=True, help="Directory containing stem audio files")
    parser.add_argument("--reference", required=True, help="Reference track path")
    parser.add_argument("--out", required=True, help="Output stereo WAV path")
    parser.add_argument("--report", default=None, help="Optional mix_report.json path")
    parser.add_argument("--style", default="auto", help="auto, pop, modern, lofi, cinematic, club, edm")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Target sample rate")
    parser.add_argument("--optimize", dest="optimize", action="store_true", default=True)
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")
    parser.add_argument("--optimize-seconds", type=float, default=30.0)
    parser.add_argument("--optimize-trials", type=int, default=24)
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    try:
        report = run_cli(args)
    except AutoMixError as exc:
        parser.exit(2, f"auto_mix error: {exc}\n")
    print(f"Wrote mix: {report['output']}")
    print(f"Wrote report: {args.report or Path(args.out).with_name('mix_report.json')}")
    print(f"Final LUFS: {report['final_features']['integrated_lufs']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
