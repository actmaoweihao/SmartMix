from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .api.projects import router as projects_router
from .api.segmentation import router as segmentation_router
from .api.tracks import router as tracks_router
from .auto_handoff import build_auto_handoff_plan
from .matching import evaluate_track_match
from .mashup import analyze_mashup_tracks, build_mashup_plan, render_mashup_plan
from .mixing import render_mix
from .repair import MatchRepairOptions, repair_track_for_match
from .seamless import generate_seamless_transition
from .services.tracks import read_track_meta, save_and_analyze_upload
from .storage import EXPORT_DIR, UPLOAD_DIR, ensure_dirs, read_json, write_json


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
app.include_router(projects_router)
app.include_router(segmentation_router)
app.include_router(tracks_router)


class ExportRequest(BaseModel):
    trackIds: list[str]
    tracks: list[dict]
    settings: dict
    format: str = "wav"


class TransitionPreviewRequest(BaseModel):
    outgoingTrackId: str
    incomingTrackId: str
    recommendation: dict | None = None
    options: dict = {}


class AutoHandoffPlanRequest(BaseModel):
    trackIds: list[str]
    tracks: list[dict] = []
    settings: dict = {}


class MashupAnalyzeRequest(BaseModel):
    trackAId: str
    trackBId: str
    barsPerSegment: int = 16
    useStems: bool = True
    segmentationAnalyzer: str = "hybrid"


class MashupPlanRequest(BaseModel):
    trackAId: str
    trackBId: str
    mode: str = "auto"
    targetDurationSec: float = 180
    barsPerSegment: int = 16
    useStems: bool = True
    transitionStrictness: str = "balanced"
    stemUsage: str = "auto"
    vocalPriority: str = "auto"
    energyCurve: str = "smooth"
    bedPreference: str = "auto"
    allowHybridBed: bool = True
    allowVocalPitchShift: bool = False
    maxVocalStretch: float = 1.06
    returnAlternatives: bool = True


class MashupRenderRequest(BaseModel):
    plan: list[dict] | dict
    format: str = "wav"
    targetLufs: float = -14
    useStems: bool = True


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/match")
async def match_two_tracks(file_a: UploadFile = File(...), file_b: UploadFile = File(...)) -> dict:
    track_a = save_and_analyze_upload(file_a)
    track_b = save_and_analyze_upload(file_b)
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

    track_a = save_and_analyze_upload(file_a)
    track_b = save_and_analyze_upload(file_b)
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


@app.post("/api/auto-handoff/plan")
def auto_handoff_plan(request: AutoHandoffPlanRequest) -> dict:
    by_id = {track["id"]: track for track in request.tracks if track.get("id")}
    metas = []
    for track_id in request.trackIds:
        meta_path = UPLOAD_DIR / f"{track_id}.json"
        if not meta_path.exists():
            raise HTTPException(status_code=404, detail=f"Track not found: {track_id}")
        meta = read_json(meta_path)
        meta.update(by_id.get(track_id, {}))
        metas.append(meta)

    try:
        return build_auto_handoff_plan(metas, request.settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Auto handoff planning failed: {exc}") from exc


@app.post("/api/transition-preview")
def transition_preview(request: TransitionPreviewRequest) -> dict:
    outgoing = read_track_meta(request.outgoingTrackId)
    incoming = read_track_meta(request.incomingTrackId)
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


@app.post("/api/mashup/analyze")
def mashup_analyze(request: MashupAnalyzeRequest) -> dict:
    track_a = read_track_meta(request.trackAId)
    track_b = read_track_meta(request.trackBId)
    try:
        return analyze_mashup_tracks(track_a, track_b, request.barsPerSegment, request.useStems, request.segmentationAnalyzer)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Mashup analysis failed: {exc}") from exc


@app.post("/api/mashup/plan")
def mashup_plan(request: MashupPlanRequest) -> dict:
    track_a = read_track_meta(request.trackAId)
    track_b = read_track_meta(request.trackBId)
    try:
        return build_mashup_plan(
            track_a,
            track_b,
            mode=request.mode,
            target_duration_sec=request.targetDurationSec,
            bars_per_segment=request.barsPerSegment,
            use_stems=request.useStems,
            transition_strictness=request.transitionStrictness,
            stem_usage=request.stemUsage,
            vocal_priority=request.vocalPriority,
            energy_curve=request.energyCurve,
            bed_preference=request.bedPreference,
            allow_hybrid_bed=request.allowHybridBed,
            allow_vocal_pitch_shift=request.allowVocalPitchShift,
            max_vocal_stretch=request.maxVocalStretch,
            return_alternatives=request.returnAlternatives,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Mashup plan failed: {exc}") from exc


@app.post("/api/mashup/render")
def mashup_render(request: MashupRenderRequest) -> dict:
    plan_items = request.plan.get("layers", []) + request.plan.get("items", request.plan.get("plan", [])) if isinstance(request.plan, dict) else request.plan
    track_ids = {str(item.get("trackId") or "") for item in plan_items if item.get("trackId")}
    if isinstance(request.plan, dict) and request.plan.get("groovePlan"):
        groove = request.plan["groovePlan"]
        bed = groove.get("bed") or {}
        for key in ("drumsTrackId", "bassTrackId", "otherTrackId"):
            if bed.get(key):
                track_ids.add(str(bed[key]))
        for event in groove.get("vocalEvents", []):
            if event.get("trackId"):
                track_ids.add(str(event["trackId"]))
    track_ids = sorted(track_id for track_id in track_ids if track_id)
    tracks_by_id = {track_id: read_track_meta(track_id) for track_id in track_ids}
    try:
        return render_mashup_plan(
            request.plan,
            tracks_by_id,
            fmt=request.format,
            target_lufs=request.targetLufs,
            use_stems=request.useStems,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Mashup render failed: {exc}") from exc


@app.get("/api/exports/{filename}")
def download_export(filename: str) -> FileResponse:
    path = EXPORT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Export file not found")
    if path.suffix == ".json":
        media_type = "application/json"
    else:
        media_type = "audio/mpeg" if path.suffix == ".mp3" else "audio/wav"
    return FileResponse(path, media_type=media_type, filename=filename)
