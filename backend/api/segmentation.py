from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..sdk.section_analysis import SectionAnalysisConfig, analyze_song_sections
from ..services.segment_analysis import SegmentAnalysisOptions, analyze_track_segments, msaf_algorithms
from ..services.tracks import cached_stem_paths, read_track_meta


router = APIRouter(prefix="/api/segmentation", tags=["segmentation"])


class SegmentAnalysisRequest(BaseModel):
    analyzer: str = "hybrid"
    boundariesId: str = "scluster"
    labelsId: str | None = "scluster"
    feature: str = "pcp"
    useStems: bool = True
    nJobs: int = 1


@router.get("/msaf/algorithms")
def get_msaf_algorithms() -> dict:
    return msaf_algorithms()


@router.post("/tracks/{track_id}")
def analyze_track_segments_endpoint(track_id: str, request: SegmentAnalysisRequest) -> dict:
    track = read_track_meta(track_id)
    stems = cached_stem_paths(track_id) if request.useStems else None
    try:
        return analyze_track_segments(
            track,
            "A",
            stems=stems,
            options=SegmentAnalysisOptions(
                analyzer=request.analyzer,
                boundaries_id=request.boundariesId,
                labels_id=request.labelsId,
                feature=request.feature,
                use_stems=request.useStems,
                n_jobs=request.nJobs,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {exc}") from exc


@router.post("/tracks/{track_id}/sections")
def analyze_track_sections_endpoint(track_id: str, request: SegmentAnalysisRequest) -> dict:
    track = read_track_meta(track_id)
    stems = cached_stem_paths(track_id) if request.useStems else None
    try:
        return analyze_song_sections(
            track["path"],
            track_id=track_id,
            name=track.get("name"),
            metadata=track,
            stems=stems,
            config=SectionAnalysisConfig(
                analyzer=request.analyzer,
                boundaries_id=request.boundariesId,
                labels_id=request.labelsId,
                feature=request.feature,
                use_stems=request.useStems,
                n_jobs=request.nJobs,
                source="A",
                enrich_metadata=False,
                include_report=True,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Section analysis failed: {exc}") from exc