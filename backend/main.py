from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .analysis import analyze_audio
from .mixing import render_mix
from .storage import EXPORT_DIR, PROJECT_DIR, UPLOAD_DIR, ensure_dirs, read_json, write_json


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


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/tracks")
async def upload_track(file: UploadFile = File(...)) -> dict:
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


@app.get("/api/exports/{filename}")
def download_export(filename: str) -> FileResponse:
    path = EXPORT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Export file not found")
    media_type = "audio/mpeg" if path.suffix == ".mp3" else "audio/wav"
    return FileResponse(path, media_type=media_type, filename=filename)


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
