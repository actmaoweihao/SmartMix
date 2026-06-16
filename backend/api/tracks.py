from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..reference_mix import render_reference_mix
from ..services.tracks import (
    STEM_NAMES,
    cached_stem_paths,
    read_track_meta,
    refresh_vocal_activity_metadata,
    save_and_analyze_upload,
    stem_paths,
    stem_response,
)
from ..storage import EXPORT_DIR, STEM_DIR, UPLOAD_DIR, read_json, write_json
from ..services.stem_separation import demucs_available, separate_demucs_stems
from ..tuning import (
    analyze_tuned_output,
    normalize_camelot,
    render_playback_transform,
    render_harmonic_tune,
)


router = APIRouter(prefix="/api/tracks", tags=["tracks"])


class StemSeparationRequest(BaseModel):
    device: str = "auto"
    force: bool = False


class ReferenceMixRequest(BaseModel):
    referenceTrackId: str
    style: str = "auto"
    optimize: bool = True
    optimizeSeconds: float = 30.0
    optimizeTrials: int = 18


class TuneTrackRequest(BaseModel):
    targetCamelot: str
    sourceCamelot: str | None = None
    direction: str = "nearest"
    format: str = "wav"
    device: str = "auto"


class PlaybackTransformRequest(BaseModel):
    speed: float = 1.0
    pitchSemitones: float = 0.0
    preserveFormants: bool = True


@router.post("")
async def upload_track(file: UploadFile = File(...)) -> dict:
    return save_and_analyze_upload(file)


@router.get("/{track_id}/audio")
def track_audio(track_id: str) -> FileResponse:
    payload = read_track_meta(track_id)
    return FileResponse(payload["path"], media_type=payload.get("content_type") or "audio/mpeg", filename=payload["name"])


@router.post("/{track_id}/playback-transform")
def playback_transform(track_id: str, request: PlaybackTransformRequest) -> dict:
    meta = read_track_meta(track_id)
    if request.speed < 0.5 or request.speed > 1.5:
        raise HTTPException(status_code=400, detail="speed must be between 0.5 and 1.5")
    if request.pitchSemitones < -12 or request.pitchSemitones > 12:
        raise HTTPException(status_code=400, detail="pitchSemitones must be between -12 and 12")

    try:
        result = render_playback_transform(
            Path(meta["path"]),
            track_id=track_id,
            speed=request.speed,
            pitch_semitones=request.pitchSemitones,
            preserve_formants=request.preserveFormants,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Playback transform failed: {exc}") from exc

    return {
        "url": f"/api/exports/{result['filename']}",
        **result,
    }


@router.post("/{track_id}/stems")
def separate_track_stems(track_id: str, request: StemSeparationRequest) -> dict:
    meta = read_track_meta(track_id)
    if request.device not in {"auto", "cuda", "cpu"}:
        raise HTTPException(status_code=400, detail="device must be auto, cuda, or cpu")

    cached = cached_stem_paths(track_id)
    if cached and not request.force:
        updated_meta = refresh_vocal_activity_metadata(track_id, cached)
        return {**stem_response(track_id, cached, device="cached", cached=True), "vocalActivity": updated_meta.get("vocal_activity"), "track": updated_meta}

    if not demucs_available():
        raise HTTPException(status_code=503, detail="Demucs is not available. Install backend/requirements-tuning.txt first.")

    source_path = Path(meta["path"])
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Track audio file not found")

    workspace = STEM_DIR / track_id
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        result = separate_demucs_stems(source_path, workspace, request.device)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Demucs separation failed: {exc}") from exc

    updated_meta = refresh_vocal_activity_metadata(track_id, result.stems)
    return {**stem_response(track_id, result.stems, device=result.device, cached=False), "vocalActivity": updated_meta.get("vocal_activity"), "track": updated_meta}


@router.get("/{track_id}/stems/{stem_name}/audio")
def track_stem_audio(track_id: str, stem_name: str) -> FileResponse:
    if stem_name not in STEM_NAMES:
        raise HTTPException(status_code=404, detail="Stem not found")
    path = stem_paths(track_id)[stem_name]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Stem not found")
    try:
        original_name = Path(read_track_meta(track_id).get("name") or track_id).stem
    except HTTPException:
        original_name = track_id
    safe_name = "".join(char if char.isalnum() or char in {"-", "_", " "} else "_" for char in original_name).strip()
    return FileResponse(path, media_type="audio/wav", filename=f"{safe_name or track_id}-{stem_name}.wav")


@router.post("/{track_id}/reference-mix")
def reference_mix(track_id: str, request: ReferenceMixRequest) -> dict:
    read_track_meta(track_id)
    reference = read_track_meta(request.referenceTrackId)
    if request.referenceTrackId == track_id:
        raise HTTPException(status_code=400, detail="Reference track must be different from the stem track")
    if request.style not in {"auto", "pop", "modern", "lofi", "cinematic", "club", "edm"}:
        raise HTTPException(status_code=400, detail="Unsupported style")
    try:
        return render_reference_mix(
            track_id,
            reference,
            style=request.style,
            optimize=request.optimize,
            optimize_seconds=request.optimizeSeconds,
            optimize_trials=request.optimizeTrials,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reference mix failed: {exc}") from exc


@router.post("/{track_id}/tune")
def tune_track(track_id: str, request: TuneTrackRequest) -> dict:
    meta_path = UPLOAD_DIR / f"{track_id}.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Track not found")

    meta = read_json(meta_path)
    source_camelot = normalize_camelot(request.sourceCamelot or meta.get("camelot"))
    target_camelot = normalize_camelot(request.targetCamelot)
    if not source_camelot or not target_camelot:
        raise HTTPException(status_code=400, detail="A valid source and target Camelot key are required")
    if request.direction not in {"nearest", "up", "down"}:
        raise HTTPException(status_code=400, detail="direction must be nearest, up, or down")
    if request.format not in {"wav", "mp3"}:
        raise HTTPException(status_code=400, detail="format must be wav or mp3")
    if request.device not in {"auto", "cuda", "cpu"}:
        raise HTTPException(status_code=400, detail="device must be auto, cuda, or cpu")

    new_track_id = uuid.uuid4().hex
    original_name = Path(meta.get("name") or meta_path.name)
    suffix = ".mp3" if request.format == "mp3" else ".wav"
    output_path = EXPORT_DIR / f"{new_track_id}_{source_camelot}_to_{target_camelot}{suffix}"

    try:
        result = render_harmonic_tune(
            Path(meta["path"]),
            source_camelot=source_camelot,
            target_camelot=target_camelot,
            output_path=output_path,
            prefer_direction=request.direction,
            fmt=request.format,
            device=request.device,
        )
        tuned_meta = analyze_tuned_output(
            result,
            {
                "id": new_track_id,
                "name": f"{original_name.stem}_{source_camelot}_to_{target_camelot}{suffix}",
                "original_track_id": track_id,
                "original_name": meta.get("name"),
            },
        )
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Tuning failed: {exc}") from exc

    write_json(UPLOAD_DIR / f"{new_track_id}.json", tuned_meta)
    return {
        **tuned_meta,
        "url": f"/api/tracks/{new_track_id}/audio",
    }
