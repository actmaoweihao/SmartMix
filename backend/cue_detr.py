from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from scipy.signal import find_peaks


OVERLAP = 0.75
W_WIN = 355
PADDING = 266


def cue_detr_enabled() -> bool:
    return os.getenv("SMARTMIX_ENABLE_CUE_DETR", "").strip().lower() in {"1", "true", "yes", "on"}


def predict_cue_points(
    path: Path,
    *,
    checkpoint: str | None = None,
    sensitivity: float = 0.9,
    min_distance: int = 16,
) -> list[dict[str, Any]]:
    if not cue_detr_enabled():
        return []

    try:
        import torch
        from matplotlib import cm
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(f"CUE-DETR dependencies are missing: {exc}") from exc

    image_processor, model, device = _load_model(checkpoint or os.getenv("SMARTMIX_CUE_DETR_CHECKPOINT", "disco-eth/cue-detr"))
    y, _ = librosa.load(path, sr=22050, mono=True)
    if y.size == 0:
        return []

    mel = librosa.feature.melspectrogram(y=y, sr=22050, n_fft=2048)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    image = _spectrogram_image(mel_db, cm, Image)
    images, borders = _sliding_windows(image)
    if not images:
        return []

    encoding = image_processor.preprocess(images, do_resize=False, return_tensors="pt")
    pixel_values = encoding["pixel_values"].to(device)
    with torch.no_grad():
        outputs = model(pixel_values)

    to_pixel = [(128, 355)] * pixel_values.shape[0]
    predictions = image_processor.post_process_object_detection(outputs, 0, to_pixel)
    scores: list[float] = []
    positions: list[int] = []
    for prediction, left in zip(predictions, borders):
        boxes = prediction["boxes"]
        if boxes.numel() == 0:
            continue
        scores.extend(float(value) for value in prediction["scores"].tolist())
        centers = (boxes[:, 0] + boxes[:, 2]) // 2 + left
        positions.extend(int(value) for value in centers.long().tolist())
    if not positions:
        return []

    ordered = sorted(zip(positions, _normalize(scores)), key=lambda item: item[0])
    ordered_positions = [item[0] for item in ordered]
    ordered_scores = [item[1] for item in ordered]
    peak_idx, props = find_peaks(ordered_scores, height=float(sensitivity), distance=int(min_distance))
    cues = []
    for idx, height in zip(peak_idx, props.get("peak_heights", [])):
        position = max(0, int(ordered_positions[int(idx)]))
        time_value = float(librosa.frames_to_time(position, sr=22050))
        cues.append({"time": round(time_value, 3), "score": round(float(height) * 100, 1), "source": "cue_detr"})
    return cues


@lru_cache(maxsize=2)
def _load_model(checkpoint: str):
    try:
        import torch
        from transformers import DetrForObjectDetection, DetrImageProcessor
    except Exception as exc:
        raise RuntimeError(f"CUE-DETR requires torch and transformers: {exc}") from exc

    image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DetrForObjectDetection.from_pretrained(checkpoint)
    model.to(device)
    model.eval()
    return image_processor, model, device


def _spectrogram_image(mel_db: np.ndarray, cm: Any, image_module: Any) -> np.ndarray:
    arr = mel_db[::-1]
    mapper = cm.ScalarMappable(cmap="viridis")
    mapper.set_clim(float(arr.min()), float(arr.max()))
    rgba = mapper.to_rgba(arr, bytes=True)
    rgb_shape = (rgba.shape[1], rgba.shape[0])
    rgba = np.require(rgba, requirements="C")
    image = image_module.frombuffer("RGBA", rgb_shape, rgba, "raw", "RGBA", 0, 1)
    return np.asarray(image)[:, :, :3]


def _sliding_windows(image: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    image_w = int(image.shape[1]) + PADDING
    n_windows = int(np.floor(image_w / (W_WIN * (1 - OVERLAP))))
    images: list[np.ndarray] = []
    borders: list[int] = []
    for index in range(n_windows):
        left = int(np.floor(index * W_WIN * (1 - OVERLAP))) - PADDING
        right = left + W_WIN
        borders.append(left)
        if left < 0:
            segment = image[:, :right]
            segment = np.pad(segment, ((0, 0), (-left, 0), (0, 0)), mode="linear_ramp")
        elif right > image.shape[1]:
            segment = image[:, left:]
            segment = np.pad(segment, ((0, 0), (0, right - left - segment.shape[1]), (0, 0)), mode="linear_ramp")
        else:
            segment = image[:, left:right]
        images.append(segment)
    return images, borders


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]
