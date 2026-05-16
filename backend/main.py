from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .analysis import analyze_audio
from .matching import evaluate_track_match
from .mixing import render_mix
from .repair import MatchRepairOptions, repair_track_for_match
from .seamless import generate_seamless_transition
from .storage import EXPORT_DIR, PROJECT_DIR, UPLOAD_DIR, ensure_dirs, read_json, write_json
from .tuning import analyze_tuned_output, normalize_camelot, render_harmonic_tune


ensure_dirs()

app = FastAPI(title="SmartMix API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_origin_regex=r"http://(127\.0\.0\.1|localhost|0\.0\.0\.0|\[::1\])(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


class ExportRequest(BaseModel):
    trackIds: list[str]
    tracks: list[dict]
    settings: dict
    format: str = "wav"


class ProjectRequest(BaseModel):
    name: str
    tracks: list[dict]
    settings: dict


class TuneTrackRequest(BaseModel):
    targetCamelot: str
    sourceCamelot: str | None = None
    direction: str = "nearest"
    format: str = "wav"
    device: str = "auto"


class TransitionPreviewRequest(BaseModel):
    outgoingTrackId: str
    incomingTrackId: str
    recommendation: dict | None = None
    options: dict = {}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/tracks")
async def upload_track(file: UploadFile = File(...)) -> dict:
    return _save_and_analyze_upload(file)


@app.post("/api/match")
async def match_two_tracks(file_a: UploadFile = File(...), file_b: UploadFile = File(...)) -> dict:
    track_a = _save_and_analyze_upload(file_a)
    track_b = _save_and_analyze_upload(file_b)
    return evaluate_track_match(track_a, track_b)


@app.post("/api/match/repair")
async def repair_match(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    process_target: str = Form("auto"),
    include_key: bool = Form(True),
    include_tempo: bool = Form(True),
    include_energy: bool = Form(True),
    max_tempo_change_percent: float = Form(10.0),
    max_pitch_shift_semitones: int = Form(4),
    format: str = Form("wav"),
) -> dict:
    if format not in {"wav", "mp3"}:
        raise HTTPException(status_code=400, detail="format must be wav or mp3")
    if max_tempo_change_percent <= 0 or max_tempo_change_percent > 20:
        raise HTTPException(status_code=400, detail="max_tempo_change_percent must be between 0 and 20")
    if max_pitch_shift_semitones < 0 or max_pitch_shift_semitones > 6:
        raise HTTPException(status_code=400, detail="max_pitch_shift_semitones must be between 0 and 6")

    track_a = _save_and_analyze_upload(file_a)
    track_b = _save_and_analyze_upload(file_b)
    options = MatchRepairOptions(
        process_target=process_target,
        include_key=include_key,
        include_tempo=include_tempo,
        include_energy=include_energy,
        max_tempo_change_percent=max_tempo_change_percent,
        max_pitch_shift_semitones=max_pitch_shift_semitones,
        format=format,
    )
    try:
        result = repair_track_for_match(track_a, track_b, options)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Match repair failed: {exc}") from exc

    repaired_track = result.get("repaired_track")
    if repaired_track:
        write_json(UPLOAD_DIR / f"{repaired_track['id']}.json", repaired_track)
        result["repaired_track"] = {
            **repaired_track,
            "url": f"/api/tracks/{repaired_track['id']}/audio",
        }
    return result


def _save_and_analyze_upload(file: UploadFile) -> dict:
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


@app.get("/api/tracks/{track_id}/audio")
def track_audio(track_id: str) -> FileResponse:
    meta_path = UPLOAD_DIR / f"{track_id}.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Track not found")
    payload = read_json(meta_path)
    return FileResponse(payload["path"], media_type=payload.get("content_type") or "audio/mpeg", filename=payload["name"])


@app.post("/api/tracks/{track_id}/tune")
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


@app.post("/api/export")
def export_mix(request: ExportRequest) -> dict:
    metas = []
    by_id = {track["id"]: track for track in request.tracks}
    for track_id in request.trackIds:
        meta_path = UPLOAD_DIR / f"{track_id}.json"
        if not meta_path.exists():
            raise HTTPException(status_code=404, detail=f"Track not found: {track_id}")
        meta = read_json(meta_path)
        meta.update(by_id.get(track_id, {}))
        metas.append(meta)

    try:
        output = render_mix(metas, request.settings, request.format)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Export failed: {exc}") from exc

    return {"url": f"/api/exports/{output.name}", "filename": output.name}


@app.post("/api/transition-preview")
def transition_preview(request: TransitionPreviewRequest) -> dict:
    outgoing = _read_track_meta(request.outgoingTrackId)
    incoming = _read_track_meta(request.incomingTrackId)
    try:
        return generate_seamless_transition(
            Path(outgoing["path"]),
            Path(incoming["path"]),
            outgoing,
            incoming,
            request.recommendation,
            request.options,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transition preview failed: {exc}") from exc


@app.get("/api/exports/{filename}")
def download_export(filename: str) -> FileResponse:
    path = EXPORT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Export file not found")
    media_type = "audio/mpeg" if path.suffix == ".mp3" else "audio/wav"
    return FileResponse(path, media_type=media_type, filename=filename)


def _read_track_meta(track_id: str) -> dict:
    meta_path = UPLOAD_DIR / f"{track_id}.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Track not found: {track_id}")
    return read_json(meta_path)


@app.post("/api/projects")
def save_project(request: ProjectRequest) -> dict:
    project_id = uuid.uuid4().hex
    payload = request.model_dump()
    payload["id"] = project_id
    write_json(PROJECT_DIR / f"{project_id}.json", payload)
    return {"id": project_id, "name": request.name}


@app.get("/api/projects")
def list_projects() -> dict:
    projects = []
    for path in sorted(PROJECT_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        payload = read_json(path)
        projects.append({"id": payload["id"], "name": payload["name"]})
    return {"projects": projects}


@app.get("/api/projects/{project_id}")
def load_project(project_id: str) -> dict:
    path = PROJECT_DIR / f"{project_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    return read_json(path)
