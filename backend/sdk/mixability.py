from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..analysis import analyze_audio
from ..services.mixability_service import MixabilityOptions, mixability_service


@dataclass(frozen=True)
class MixabilityConfig:
    """Public SDK config for team integrations."""

    profile: str = "handoff"
    mode: str = "recommended"
    limit: int = 5
    settings: dict[str, Any] = field(default_factory=dict)
    enrich_paths: bool = True


def evaluate_song_pair(
    previous: dict[str, Any] | str | Path,
    incoming: dict[str, Any] | str | Path,
    *,
    config: MixabilityConfig | None = None,
) -> dict[str, Any]:
    """Return a full mixability report for two songs."""

    cfg = config or MixabilityConfig()
    return mixability_service.evaluate_pair(
        _track_payload(previous, enrich_paths=cfg.enrich_paths),
        _track_payload(incoming, enrich_paths=cfg.enrich_paths),
        options=_options(cfg),
    )


def recommend_next_song(
    current: dict[str, Any] | str | Path,
    candidates: list[dict[str, Any] | str | Path],
    *,
    config: MixabilityConfig | None = None,
) -> dict[str, Any]:
    """Rank candidate songs for the safest next handoff."""

    cfg = config or MixabilityConfig()
    return mixability_service.recommend_next(
        _track_payload(current, enrich_paths=cfg.enrich_paths),
        [_track_payload(candidate, enrich_paths=cfg.enrich_paths) for candidate in candidates],
        options=_options(cfg),
    )


def order_songs_for_mix(
    tracks: list[dict[str, Any] | str | Path],
    *,
    config: MixabilityConfig | None = None,
) -> dict[str, Any]:
    """Order a set of songs into a mixability-aware sequence."""

    cfg = config or MixabilityConfig()
    return mixability_service.order_tracks([_track_payload(track, enrich_paths=cfg.enrich_paths) for track in tracks], options=_options(cfg))


def evaluate_song_sequence(
    tracks: list[dict[str, Any] | str | Path],
    *,
    config: MixabilityConfig | None = None,
) -> dict[str, Any]:
    """Evaluate each adjacent transition in an existing sequence."""

    cfg = config or MixabilityConfig()
    return mixability_service.evaluate_sequence([_track_payload(track, enrich_paths=cfg.enrich_paths) for track in tracks], options=_options(cfg))


def _options(config: MixabilityConfig) -> MixabilityOptions:
    return MixabilityOptions(profile=config.profile, mode=config.mode, limit=config.limit, settings=config.settings)


def _track_payload(track: dict[str, Any] | str | Path, *, enrich_paths: bool) -> dict[str, Any]:
    if isinstance(track, dict):
        if not track.get("id"):
            payload = dict(track)
            payload["id"] = str(payload.get("path") or payload.get("name") or "track")
            return payload
        return track
    audio_path = Path(track).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    payload: dict[str, Any] = {"id": audio_path.stem, "name": audio_path.name, "path": str(audio_path)}
    if enrich_paths:
        payload.update(analyze_audio(audio_path))
        payload["id"] = audio_path.stem
        payload["name"] = audio_path.name
        payload["path"] = str(audio_path)
    return payload


def config_to_dict(config: MixabilityConfig) -> dict[str, Any]:
    return asdict(config)

