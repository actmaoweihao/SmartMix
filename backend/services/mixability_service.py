from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..mixability import evaluate_mixability, evaluate_transition_sequence, order_tracks_by_mixability, recommend_next_by_mixability
from ..storage import UPLOAD_DIR, read_json


MIXABILITY_PROFILES = {"pair_match", "handoff", "sort_recommended", "sort_harmonic"}
MIXABILITY_ORDER_MODES = {"recommended", "harmonic"}


@dataclass(frozen=True)
class MixabilityOptions:
    """Stable options for team-facing mixability calls."""

    profile: str = "handoff"
    mode: str = "recommended"
    limit: int = 5
    settings: dict[str, Any] = field(default_factory=dict)


class MixabilityService:
    """Team-facing facade for song-to-song mixability workflows.

    Callers should prefer this service over importing low-level scoring helpers
    directly. The scoring internals can change while this facade stays stable.
    """

    def evaluate_pair(
        self,
        prev_track: dict[str, Any],
        next_track: dict[str, Any],
        *,
        options: MixabilityOptions | None = None,
        profile: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved = options or MixabilityOptions()
        selected_profile = profile or resolved.profile
        if selected_profile not in MIXABILITY_PROFILES:
            raise ValueError(f"profile must be one of {sorted(MIXABILITY_PROFILES)}")
        return evaluate_mixability(prev_track, next_track, profile=selected_profile, settings=settings or resolved.settings)

    def recommend_next(
        self,
        current_track: dict[str, Any],
        candidates: list[dict[str, Any]],
        *,
        options: MixabilityOptions | None = None,
        settings: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        resolved = options or MixabilityOptions()
        return recommend_next_by_mixability(
            current_track,
            candidates,
            settings=settings or resolved.settings,
            limit=limit if limit is not None else resolved.limit,
        )

    def order_tracks(
        self,
        tracks: list[dict[str, Any]],
        *,
        options: MixabilityOptions | None = None,
        mode: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved = options or MixabilityOptions()
        selected_mode = mode or resolved.mode
        if selected_mode not in MIXABILITY_ORDER_MODES:
            raise ValueError(f"mode must be one of {sorted(MIXABILITY_ORDER_MODES)}")
        return order_tracks_by_mixability(tracks, mode=selected_mode, settings=settings or resolved.settings)

    def evaluate_sequence(
        self,
        tracks: list[dict[str, Any]],
        *,
        options: MixabilityOptions | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved = options or MixabilityOptions()
        return evaluate_transition_sequence(tracks, settings=settings or resolved.settings)

    def resolve_tracks(self, track_ids: list[str], track_overrides: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        by_id = {track["id"]: track for track in track_overrides or [] if track.get("id")}
        metas = []
        for track_id in track_ids:
            if not track_id:
                raise ValueError("trackIds cannot contain empty values")
            meta_path = UPLOAD_DIR / f"{track_id}.json"
            if meta_path.exists():
                meta = read_json(meta_path)
                meta.update(by_id.get(track_id, {}))
            elif track_id in by_id:
                meta = by_id[track_id]
            else:
                raise FileNotFoundError(f"Track not found: {track_id}")
            metas.append(meta)
        return metas

    def resolve_track_map(self, track_ids: list[str], track_overrides: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
        tracks = self.resolve_tracks(track_ids, track_overrides)
        return {str(track.get("id")): track for track in tracks if track.get("id")}


mixability_service = MixabilityService()

