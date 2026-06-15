from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .cue_detr import cue_detr_enabled, predict_cue_points


def collect_learned_cue_points(path: Path) -> dict[str, Any]:
    """Collect cue candidates from optional learned providers.

    Providers are deliberately additive: model cues become candidates that the
    deterministic cue scorer can re-score and filter for phrase/vocal safety.
    """

    cues: list[dict[str, Any]] = []
    providers: list[dict[str, Any]] = []

    if cue_detr_enabled():
        try:
            cue_detr_cues = predict_cue_points(path)
            cues.extend(_normalize_cues(cue_detr_cues, "cue_detr"))
            providers.append({"provider": "cue_detr", "enabled": True, "used": bool(cue_detr_cues), "count": len(cue_detr_cues)})
        except Exception as exc:
            providers.append({"provider": "cue_detr", "enabled": True, "used": False, "error": str(exc)})

    sidecar_path = _sidecar_path(path)
    if sidecar_path and sidecar_path.exists():
        try:
            sidecar_cues = _load_sidecar_cues(sidecar_path)
            cues.extend(_normalize_cues(sidecar_cues, "self_trained_sidecar"))
            providers.append({"provider": "self_trained_sidecar", "enabled": True, "used": bool(sidecar_cues), "count": len(sidecar_cues), "path": str(sidecar_path)})
        except Exception as exc:
            providers.append({"provider": "self_trained_sidecar", "enabled": True, "used": False, "error": str(exc), "path": str(sidecar_path)})

    command = os.getenv("SMARTMIX_CUE_MODEL_COMMAND", "").strip()
    if command:
        try:
            command_cues = _run_command_provider(command, path)
            cues.extend(_normalize_cues(command_cues, "self_trained_command"))
            providers.append({"provider": "self_trained_command", "enabled": True, "used": bool(command_cues), "count": len(command_cues)})
        except Exception as exc:
            providers.append({"provider": "self_trained_command", "enabled": True, "used": False, "error": str(exc)})

    deduped: dict[int, dict[str, Any]] = {}
    for cue in cues:
        key = int(round(float(cue["time"]) * 4))
        if key not in deduped or float(cue.get("score") or 0) > float(deduped[key].get("score") or 0):
            deduped[key] = cue

    return {
        "cues": sorted(deduped.values(), key=lambda cue: float(cue["time"])),
        "providers": providers,
    }


def _sidecar_path(path: Path) -> Path | None:
    explicit_dir = os.getenv("SMARTMIX_LEARNED_CUE_DIR", "").strip()
    if explicit_dir:
        return Path(explicit_dir).expanduser().resolve() / f"{path.stem}.cues.json"
    explicit_path = os.getenv("SMARTMIX_LEARNED_CUE_FILE", "").strip()
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    candidate = path.with_suffix(".cues.json")
    return candidate if candidate.exists() else None


def _load_sidecar_cues(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return list(payload.get("cues") or payload.get("cue_candidates") or [])
    return []


def _run_command_provider(command: str, path: Path) -> list[dict[str, Any]]:
    args = [part.format(path=str(path), stem=path.stem) for part in shlex.split(command)]
    if not args:
        return []
    completed = subprocess.run(args, capture_output=True, text=True, timeout=90, check=True)
    payload = json.loads(completed.stdout or "[]")
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return list(payload.get("cues") or payload.get("cue_candidates") or [])
    return []


def _normalize_cues(cues: list[dict[str, Any]], provider: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for cue in cues:
        if not isinstance(cue, dict):
            continue
        time_value = cue.get("time", cue.get("timeSec"))
        if not isinstance(time_value, (int, float)):
            continue
        score = cue.get("score", cue.get("confidence", cue.get("probability", 0.7)))
        score_value = float(score)
        if score_value <= 1.0:
            score_value *= 100.0
        normalized.append(
            {
                "time": round(float(time_value), 3),
                "score": round(max(0.0, min(100.0, score_value)), 1),
                "role": cue.get("role"),
                "source": cue.get("source") or provider,
                "provider": provider,
                "raw": {key: value for key, value in cue.items() if key not in {"time", "timeSec", "score", "confidence", "probability"}},
            }
        )
    return normalized

