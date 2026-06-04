from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from ..segmentation import (
    SAMPLE_RATE,
    annotate_structural_groups,
    analyze_track_segmentation,
    compute_similarity_matrices,
    extract_bar_features,
    summarize_structural_groups,
    _load_stems,
    _make_section,
    _mono,
    _nearest_index,
    _safe_load_audio,
)
from .section_labeler import refine_section_labels


MSAF_BOUNDARY_ALGORITHMS = {"cnmf", "example", "foote", "olda", "scluster", "sf", "vmo"}
MSAF_LABEL_ALGORITHMS = {"cnmf", "fmc2d", "scluster", "vmo"}
MSAF_FEATURES = {"pcp", "mfcc", "tonnetz"}


@dataclass(frozen=True)
class SegmentAnalysisOptions:
    analyzer: str = "hybrid"
    boundaries_id: str = "scluster"
    labels_id: str | None = "scluster"
    feature: str = "pcp"
    use_stems: bool = True
    n_jobs: int = 1


def msaf_available() -> bool:
    try:
        _import_msaf()
    except Exception:
        return False
    return True


def msaf_algorithms() -> dict[str, Any]:
    try:
        msaf = _import_msaf()
        boundaries = sorted(msaf.get_all_boundary_algorithms())
        labels = sorted(msaf.get_all_label_algorithms())
    except Exception:
        boundaries = sorted(MSAF_BOUNDARY_ALGORITHMS)
        labels = sorted(MSAF_LABEL_ALGORITHMS)
    return {
        "available": msaf_available(),
        "boundaries": boundaries,
        "labels": labels,
        "features": sorted(MSAF_FEATURES),
        "default": {"boundariesId": "scluster", "labelsId": "scluster", "feature": "pcp"},
    }


def analyze_track_segments(
    track: dict[str, Any],
    source: str = "A",
    *,
    stems: dict[str, Any] | None = None,
    options: SegmentAnalysisOptions | None = None,
) -> dict[str, Any]:
    options = _normalize_options(options or SegmentAnalysisOptions())
    audio_path = Path(str(track.get("path") or ""))
    if not audio_path.exists():
        raise ValueError("Track audio file not found")

    if options.analyzer == "smartmix":
        audio = _safe_load_audio(audio_path, SAMPLE_RATE)
        return analyze_track_segmentation(track, source, audio=audio, sr=SAMPLE_RATE, stems=stems if options.use_stems else None)

    try:
        msaf_report = analyze_with_msaf_package(track, source, stems=stems if options.use_stems else None, options=options)
    except Exception as exc:
        if options.analyzer == "msaf":
            raise RuntimeError(f"MSAF segmentation failed: {exc}") from exc
        audio = _safe_load_audio(audio_path, SAMPLE_RATE)
        fallback = analyze_track_segmentation(track, source, audio=audio, sr=SAMPLE_RATE, stems=stems if options.use_stems else None)
        fallback["method"] = f"{fallback.get('method', 'smartmix')}_fallback_after_msaf_error"
        fallback.setdefault("warnings", []).append(f"MSAF failed; used SmartMix fallback. Detail: {exc}")
        return fallback

    if options.analyzer == "msaf":
        return msaf_report

    audio = _safe_load_audio(audio_path, SAMPLE_RATE)
    smartmix = analyze_track_segmentation(track, source, audio=audio, sr=SAMPLE_RATE, stems=stems if options.use_stems else None)
    return _merge_hybrid_report(msaf_report, smartmix)


def analyze_with_msaf_package(
    track: dict[str, Any],
    source: str = "A",
    *,
    stems: dict[str, Any] | None = None,
    options: SegmentAnalysisOptions | None = None,
) -> dict[str, Any]:
    options = _normalize_options(options or SegmentAnalysisOptions(analyzer="msaf"))
    msaf = _import_msaf()
    audio_path = Path(str(track.get("path") or "")).resolve()
    warnings: list[str] = []
    boundaries, labels, used_boundaries_id, used_labels_id = _run_msaf_process(msaf, audio_path, options, warnings)
    boundaries = _clean_boundaries(boundaries, _track_duration(track, audio_path))
    labels = _clean_labels(labels, len(boundaries) - 1)
    audio = _safe_load_audio(audio_path, SAMPLE_RATE)
    stem_audio = _load_stems(stems, SAMPLE_RATE) if options.use_stems else {}
    bar_features = extract_bar_features(audio, SAMPLE_RATE, track, stem_audio or None)
    matrices = compute_similarity_matrices(bar_features)
    sections = _sections_from_msaf_boundaries(track, source, boundaries, labels, bar_features, matrices)
    sections = annotate_structural_groups(sections, matrices)
    sections = refine_section_labels(sections, bar_features)
    return {
        "method": f"msaf_package_{used_boundaries_id}",
        "analyzer": "msaf",
        "msaf": {
            "boundariesId": used_boundaries_id,
            "requestedBoundariesId": options.boundaries_id,
            "labelsId": used_labels_id,
            "requestedLabelsId": options.labels_id,
            "feature": options.feature,
            "rawBoundaries": [round(float(value), 3) for value in boundaries],
            "rawLabels": labels,
        },
        "barsDetected": len(bar_features) >= 2,
        "stemsUsed": bool(stem_audio),
        "boundaryCount": max(0, len(boundaries) - 2),
        "barFeatures": [asdict(item) for item in bar_features],
        "sections": sections,
        "minorSections": [],
        "vocalPhrases": [],
        "grooveBedCandidates": [],
        "safeCutPoints": [],
        "structuralGroups": summarize_structural_groups(sections),
        "warnings": warnings,
        "debug": {
            "msafBoundaries": [round(float(value), 3) for value in boundaries],
            "msafLabels": labels,
            "ssmShape": list(matrices["fused"].shape),
            "majorSectionCount": len(sections),
            "minorSectionCount": 0,
        },
    }


def _normalize_options(options: SegmentAnalysisOptions) -> SegmentAnalysisOptions:
    analyzer = (options.analyzer or "hybrid").lower()
    if analyzer not in {"smartmix", "msaf", "hybrid"}:
        raise ValueError("analyzer must be smartmix, msaf, or hybrid")
    boundaries_id = (options.boundaries_id or "scluster").lower()
    if boundaries_id not in MSAF_BOUNDARY_ALGORITHMS:
        raise ValueError("Unsupported MSAF boundary algorithm")
    labels_id = (options.labels_id or "").lower() or None
    if labels_id is not None and labels_id not in MSAF_LABEL_ALGORITHMS:
        raise ValueError("Unsupported MSAF label algorithm")
    feature = (options.feature or "pcp").lower()
    if feature not in MSAF_FEATURES:
        raise ValueError("feature must be pcp, mfcc, or tonnetz")
    return SegmentAnalysisOptions(
        analyzer=analyzer,
        boundaries_id=boundaries_id,
        labels_id=labels_id,
        feature=feature,
        use_stems=bool(options.use_stems),
        n_jobs=max(1, int(options.n_jobs or 1)),
    )


def _run_msaf_process(msaf: Any, audio_path: Path, options: SegmentAnalysisOptions, warnings: list[str]) -> tuple[Any, Any, str, str | None]:
    attempts: list[tuple[str, str | None]] = [(options.boundaries_id, options.labels_id)]
    if options.boundaries_id != "sf":
        attempts.append(("sf", None))
    last_error: Exception | None = None
    for boundaries_id, labels_id in attempts:
        try:
            with tempfile.TemporaryDirectory(prefix="smartmix_msaf_") as tmp:
                previous_cwd = os.getcwd()
                try:
                    os.chdir(tmp)
                    boundaries, labels = msaf.process(
                        str(audio_path),
                        boundaries_id=boundaries_id,
                        labels_id=labels_id,
                        feature=options.feature,
                        plot=False,
                        sonify_bounds=False,
                        n_jobs=max(1, int(options.n_jobs)),
                    )
                finally:
                    os.chdir(previous_cwd)
            if boundaries_id != options.boundaries_id or labels_id != options.labels_id:
                warnings.append(f"MSAF {options.boundaries_id}/{options.labels_id} failed; used {boundaries_id}/{labels_id} fallback.")
            return boundaries, labels, boundaries_id, labels_id
        except Exception as exc:
            last_error = exc
    raise RuntimeError(str(last_error or "MSAF process failed"))


def _import_msaf() -> Any:
    import scipy

    if not hasattr(scipy, "inf"):
        scipy.inf = np.inf
    import msaf

    return msaf


def _track_duration(track: dict[str, Any], audio_path: Path) -> float:
    value = track.get("duration")
    try:
        parsed = float(value)
        if np.isfinite(parsed) and parsed > 0:
            return parsed
    except (TypeError, ValueError):
        pass
    return float(librosa.get_duration(path=audio_path))


def _clean_boundaries(boundaries: Any, duration: float) -> list[float]:
    values = sorted(float(value) for value in np.asarray(boundaries).reshape(-1) if np.isfinite(float(value)))
    values = [value for value in values if 0 <= value <= duration + 0.25]
    if not values or values[0] > 0.05:
        values.insert(0, 0.0)
    if values[-1] < duration - 0.05:
        values.append(duration)
    cleaned = [max(0.0, min(float(values[0]), duration))]
    for value in values[1:]:
        clamped = max(0.0, min(float(value), duration))
        if clamped - cleaned[-1] >= 0.35:
            cleaned.append(clamped)
    if len(cleaned) < 2:
        cleaned = [0.0, duration]
    return cleaned


def _clean_labels(labels: Any, count: int) -> list[str]:
    raw = [str(value) for value in np.asarray(labels if labels is not None else []).reshape(-1)]
    if not raw:
        raw = [f"S{index + 1}" for index in range(max(0, count))]
    if len(raw) < count:
        raw.extend(raw[-1:] * (count - len(raw)) if raw else [f"S{index + 1}" for index in range(count)])
    return raw[:count]


def _sections_from_msaf_boundaries(
    track: dict[str, Any],
    source: str,
    boundaries: list[float],
    labels: list[str],
    bar_features: list[Any],
    matrices: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    if not bar_features:
        return []
    starts = [float(item.start) for item in bar_features]
    ends = [float(item.end) for item in bar_features]
    sections = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        start_idx = _nearest_index(starts, start)
        end_idx = _nearest_index(ends, end) + 1
        end_idx = max(start_idx + 1, min(len(bar_features), end_idx))
        section = _make_section(track, source, bar_features, matrices, start_idx, end_idx, len(sections) + 1, "msaf")
        section["msafLabel"] = labels[index] if index < len(labels) else None
        section["labelSource"] = "msaf+smartmix_metrics"
        sections.append(section)
    return sections


def _merge_hybrid_report(msaf_report: dict[str, Any], smartmix_report: dict[str, Any]) -> dict[str, Any]:
    report = dict(smartmix_report)
    report["method"] = "hybrid_msaf_package_smartmix_stem_aware"
    report["analyzer"] = "hybrid"
    report["msaf"] = msaf_report.get("msaf", {})
    report["sections"] = msaf_report.get("sections", []) or smartmix_report.get("sections", [])
    report["structuralGroups"] = msaf_report.get("structuralGroups", []) or smartmix_report.get("structuralGroups", [])
    report["boundaryCount"] = len(report["sections"])
    report["warnings"] = [
        *smartmix_report.get("warnings", []),
        *msaf_report.get("warnings", []),
    ]
    debug = dict(smartmix_report.get("debug") or {})
    debug["msafPackage"] = msaf_report.get("debug", {})
    report["debug"] = debug
    return report
