from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..storage import PROJECT_DIR, read_json, write_json


router = APIRouter(prefix="/api/projects", tags=["projects"])


class ProjectRequest(BaseModel):
    name: str
    tracks: list[dict]
    settings: dict


@router.post("")
def save_project(request: ProjectRequest) -> dict:
    project_id = uuid.uuid4().hex
    payload = request.model_dump()
    payload["id"] = project_id
    write_json(PROJECT_DIR / f"{project_id}.json", payload)
    return {"id": project_id, "name": request.name}


@router.get("")
def list_projects() -> dict:
    projects = []
    for path in sorted(PROJECT_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        payload = read_json(path)
        projects.append({"id": payload["id"], "name": payload["name"]})
    return {"projects": projects}


@router.get("/{project_id}")
def load_project(project_id: str) -> dict:
    path = PROJECT_DIR / f"{project_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    return read_json(path)
