from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile

from ..analysis import analyze_audio
from ..storage import STEM_DIR, UPLOAD_DIR, read_json, write_json
from ..vocal_activity import analyze_vocal_stem, merge_vocal_activity_into_analysis


AUDIO_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".weba",
    ".webm",
}
STEM_NAMES = ("vocals", "drums", "bass", "other")


def save_and_analyze_upload(file: UploadFile) -> dict:
    track_id = uuid.uuid4().hex
    suffix = Path(file.filename or "track").suffix or ".audio"
    content_type = file.content_type or "application/octet-stream"
    is_audio_mime = content_type.startswith("audio/")
    is_audio_extension = suffix.lower() in AUDIO_EXTENSIONS
    if not is_audio_mime and not is_audio_extension:
        raise HTTPException(status_code=400, detail="Unsupported audio file type")

    path = UPLOAD_DIR / f"{track_id}{suffix}"
    with path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    try:
        analysis = analyze_audio(path)
        cached = cached_stem_paths(track_id)
        if cached and cached.get("vocals"):
            report = analyze_vocal_stem(cached["vocals"], analysis.get("duration"))
            analysis = merge_vocal_activity_into_analysis(analysis, report)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Audio analysis failed: {exc}") from exc

    payload = {
        "id": track_id,
        "name": file.filename or path.name,
        "path": str(path),
        "content_type": content_type,
        **analysis,
    }
    write_json(UPLOAD_DIR / f"{track_id}.json", payload)
    return payload


def read_track_meta(track_id: str) -> dict:
    meta_path = UPLOAD_DIR / f"{track_id}.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Track not found: {track_id}")
    return read_json(meta_path)


def stem_paths(track_id: str) -> dict[str, Path]:
    return {stem: STEM_DIR / track_id / "demucs_api" / f"{stem}.wav" for stem in STEM_NAMES}


def cached_stem_paths(track_id: str) -> dict[str, Path] | None:
    paths = stem_paths(track_id)
    if all(path.exists() for path in paths.values()):
        return paths
    return None


def stem_response(track_id: str, paths: dict[str, Path], device: str, cached: bool) -> dict:
    return {
        "trackId": track_id,
        "engine": "demucs",
        "device": device,
        "cached": cached,
        "stems": {
            stem: {
                "url": f"/api/tracks/{track_id}/stems/{stem}/audio",
                "path": str(paths[stem]),
            }
            for stem in STEM_NAMES
        },
    }


def refresh_vocal_activity_metadata(track_id: str, paths: dict[str, Path]) -> dict:
    meta = read_track_meta(track_id)
    vocals = paths.get("vocals")
    if not vocals or not vocals.exists():
        return meta
    report = analyze_vocal_stem(vocals, meta.get("duration"))
    updated = merge_vocal_activity_into_analysis(meta, report)
    write_json(UPLOAD_DIR / f"{track_id}.json", updated)
    return updated
