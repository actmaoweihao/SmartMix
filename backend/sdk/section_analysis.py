from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..analysis import analyze_audio
from ..arrangement import segments_from_segmentation_report
from ..services.segment_analysis import SegmentAnalysisOptions, analyze_track_segments, msaf_algorithms


@dataclass(frozen=True)
class SectionAnalysisConfig:
    """Stable public config for section analysis, labeling, and splitting."""

    analyzer: str = "hybrid"
    boundaries_id: str = "scluster"
    labels_id: str | None = "scluster"
    feature: str = "pcp"
    use_stems: bool = True
    n_jobs: int = 1
    source: str = "A"
    enrich_metadata: bool = True
    include_report: bool = True


def analyze_song_sections(
    input_path: str | Path,
    *,
    track_id: str | None = None,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
    stems: dict[str, str | Path] | None = None,
    config: SectionAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Analyze one song and return team-facing section labels and split ranges.

    This is the public SDK wrapper for the full SmartMix section pipeline:
    audio metadata -> MSAF/SmartMix segmentation -> section labeling -> timeline
    coverage repair -> normalized sections for callers.
    """

    cfg = config or SectionAnalysisConfig()
    audio_path = Path(input_path).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    warnings: list[str] = []
    track = _build_track_payload(audio_path, track_id=track_id, name=name, metadata=metadata, enrich_metadata=cfg.enrich_metadata, warnings=warnings)
    stem_paths = _normalize_stems(stems)
    options = SegmentAnalysisOptions(
        analyzer=cfg.analyzer,
        boundaries_id=cfg.boundaries_id,
        labels_id=cfg.labels_id,
        feature=cfg.feature,
        use_stems=cfg.use_stems,
        n_jobs=cfg.n_jobs,
    )
    report = analyze_track_segments(track, cfg.source, stems=stem_paths, options=options)
    sections = segments_from_segmentation_report(track, cfg.source, report.get("sections") or [], report)
    report_warnings = list(report.get("warnings") or [])

    result = {
        "track": _public_track(track),
        "config": asdict(cfg),
        "analyzer": report.get("analyzer") or cfg.analyzer,
        "method": report.get("method"),
        "sections": sections,
        "sectionCount": len(sections),
        "warnings": [*warnings, *report_warnings],
        "msaf": report.get("msaf", {}),
        "availableAlgorithms": msaf_algorithms(),
    }
    if cfg.include_report:
        result["report"] = report
    return result


def _build_track_payload(
    audio_path: Path,
    *,
    track_id: str | None,
    name: str | None,
    metadata: dict[str, Any] | None,
    enrich_metadata: bool,
    warnings: list[str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if enrich_metadata:
        try:
            payload.update(analyze_audio(audio_path))
        except Exception as exc:
            warnings.append(f"Audio metadata analysis failed; using segmentation fallback metadata. Detail: {exc}")
    payload.update(metadata or {})
    payload["id"] = str(track_id or payload.get("id") or audio_path.stem)
    payload["name"] = str(name or payload.get("name") or audio_path.name)
    payload["path"] = str(audio_path)
    return payload


def _normalize_stems(stems: dict[str, str | Path] | None) -> dict[str, Path] | None:
    if not stems:
        return None
    result = {}
    for name in ("vocals", "drums", "bass", "other"):
        value = stems.get(name)
        if value:
            result[name] = Path(value).expanduser().resolve()
    return result or None


def _public_track(track: dict[str, Any]) -> dict[str, Any]:
    keys = ("id", "name", "path", "duration", "bpm", "key", "camelot", "mode", "style", "style_label")
    return {key: track.get(key) for key in keys if key in track}
