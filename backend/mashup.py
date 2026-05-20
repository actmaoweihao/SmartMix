from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from .arrangement import build_track_segments, segment_compatibility
from .loudness import loudness_metrics, normalize_loudness
from .matching import camelot_key_distance
from .mixing import SAMPLE_RATE, _convert_to_mp3, _fade_curves
from .storage import EXPORT_DIR, STEM_DIR


STEM_NAMES = ("vocals", "drums", "bass", "other")
LAYER_MODES = {"full_mix", "vocals", "drums_bass_other", "instrumental"}
TRANSITIONS = {"none", "crossfade", "bass_swap", "filter_sweep", "hard_cut"}


def analyze_mashup_tracks(track_a: dict[str, Any], track_b: dict[str, Any], bars_per_segment: int = 16, use_stems: bool = True) -> dict[str, Any]:
    audio_a = _load_mono(Path(track_a["path"]))
    audio_b = _load_mono(Path(track_b["path"]))
    return {
        "trackA": {"segments": build_track_segments(track_a, "A", bars_per_segment, audio=audio_a, sr=SAMPLE_RATE)},
        "trackB": {"segments": build_track_segments(track_b, "B", bars_per_segment, audio=audio_b, sr=SAMPLE_RATE)},
        "useStems": bool(use_stems),
    }


def build_mashup_plan(
    track_a: dict[str, Any],
    track_b: dict[str, Any],
    *,
    mode: str = "auto",
    target_duration_sec: float = 180.0,
    bars_per_segment: int = 16,
    use_stems: bool = True,
) -> dict[str, Any]:
    analysis = analyze_mashup_tracks(track_a, track_b, bars_per_segment, use_stems)
    segments_a = analysis["trackA"]["segments"]
    segments_b = analysis["trackB"]["segments"]
    warnings = _base_warnings(track_a, track_b)
    if use_stems and (not _cached_stem_paths(str(track_a.get("id"))) or not _cached_stem_paths(str(track_b.get("id")))):
        warnings.append("Demucs stems are not cached for both tracks; stem layer modes will fall back to full mix where needed.")
    if not segments_a or not segments_b:
        return {"plan": [], "score": 0, "warnings": ["Not enough segment data to build a mashup plan."], "analysis": analysis}

    modes = ["smooth_join", "hook_swap", "a_vocal_b_instrumental", "b_vocal_a_instrumental", "energy_build"] if mode == "auto" else [mode]
    candidates = []
    for candidate_mode in modes:
        plan = _template_plan(candidate_mode, segments_a, segments_b, target_duration_sec, use_stems)
        score = score_plan(plan, warnings)
        candidates.append((score, candidate_mode, plan))
    score, selected_mode, plan = max(candidates, key=lambda item: item[0])
    return {
        "plan": plan,
        "score": int(round(score)),
        "warnings": warnings,
        "mode": selected_mode,
        "analysis": analysis,
    }


def render_mashup_plan(
    plan: list[dict[str, Any]],
    tracks_by_id: dict[str, dict[str, Any]],
    *,
    fmt: str = "wav",
    target_lufs: float = -14.0,
    use_stems: bool = True,
) -> dict[str, Any]:
    if not plan:
        raise ValueError("Mashup plan is empty")
    rendered = _render_plan_audio(plan, tracks_by_id, use_stems=use_stems)
    rendered = normalize_loudness(rendered, SAMPLE_RATE, float(target_lufs))
    rendered = np.nan_to_num(rendered, nan=0.0, posinf=0.0, neginf=0.0)
    rendered = _peak_limit(rendered)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = EXPORT_DIR / f"mashup_{uuid.uuid4().hex}.wav"
    sf.write(wav_path, rendered.T, SAMPLE_RATE, subtype="PCM_16")
    output_path = _convert_to_mp3(wav_path) if fmt == "mp3" else wav_path
    report = {
        "items": len(plan),
        "duration": round(rendered.shape[1] / SAMPLE_RATE, 3),
        "format": fmt,
        "targetLufs": target_lufs,
        "loudness": loudness_metrics(rendered, SAMPLE_RATE),
        "usedStems": bool(use_stems),
        "containsInvalid": bool(not np.all(np.isfinite(rendered))),
    }
    return {"ok": True, "downloadUrl": f"/api/exports/{output_path.name}", "report": report}


def score_plan(plan: list[dict[str, Any]], warnings: list[str] | None = None) -> float:
    if not plan:
        return 0.0
    sequential = sorted(plan, key=lambda item: (float(item.get("timelineStart", 0)), float(item.get("timelineEnd", 0))))
    transitions = []
    for left, right in zip(sequential, sequential[1:]):
        if abs(float(left.get("timelineEnd", 0)) - float(right.get("timelineStart", 0))) > 0.05:
            continue
        if left.get("_segment") and right.get("_segment"):
            transitions.append(segment_compatibility(left["_segment"], right["_segment"])["score"])
    base = float(np.mean(transitions)) if transitions else 0.68
    source_bonus = 0.08 if {"A", "B"}.issubset({item.get("source") for item in plan}) else -0.20
    layer_bonus = 0.08 if any(item.get("layerMode") != "full_mix" for item in plan) else 0.0
    warning_penalty = min(0.18, 0.04 * len(warnings or []))
    return max(0.0, min(100.0, (base + source_bonus + layer_bonus - warning_penalty) * 100.0))


def _template_plan(mode: str, a: list[dict[str, Any]], b: list[dict[str, Any]], target_duration: float, use_stems: bool) -> list[dict[str, Any]]:
    if mode == "hook_swap":
        sequence = [
            _pick(a, {"verse_like", "intro_like"}),
            _pick(b, {"chorus_like", "drop_like"}),
            _pick(a, {"breakdown_like", "verse_like"}),
            _pick(b, {"chorus_like", "outro_like", "drop_like"}),
        ]
        return _sequence_plan([item for item in sequence if item], target_duration)
    if mode == "a_vocal_b_instrumental":
        return _layer_plan(_pick(a, {"verse_like", "chorus_like"}), _pick(b, {"intro_like", "breakdown_like", "drop_like", "verse_like"}), "A", use_stems)
    if mode == "b_vocal_a_instrumental":
        return _layer_plan(_pick(b, {"verse_like", "chorus_like"}), _pick(a, {"intro_like", "breakdown_like", "drop_like", "verse_like"}), "B", use_stems)
    if mode == "energy_build":
        pool = sorted(a + b, key=lambda item: (float(item.get("energy", 0)), item.get("source")))
        selected = _ensure_both_sources(pool[:5] + pool[-3:], a, b)
        return _sequence_plan(selected[:6], target_duration, transition="filter_sweep")
    # smooth_join/default
    first = [_pick(a, {"intro_like", "verse_like"}), _pick(a, {"chorus_like", "drop_like"})]
    anchor = first[-1] or first[0] or a[0]
    second = [_best(anchor, b, {"chorus_like", "drop_like", "outro_like"}), _pick(b, {"outro_like", "breakdown_like"})]
    return _sequence_plan([item for item in first + second if item], target_duration)


def _sequence_plan(segments: list[dict[str, Any]], target_duration: float, transition: str = "crossfade") -> list[dict[str, Any]]:
    plan = []
    cursor = 0.0
    for index, segment in enumerate(segments):
        duration = max(0.5, float(segment["end"]) - float(segment["start"]))
        if plan and cursor + duration > target_duration * 1.12:
            break
        crossfade = 0.0 if index == 0 else min(4.0, duration * 0.2)
        timeline_start = max(0.0, cursor - crossfade)
        timeline_end = timeline_start + duration
        plan.append(_plan_item(segment, timeline_start, timeline_end, "full_mix", transition if index else "none", transition))
        cursor = timeline_end
    if plan:
        plan[-1]["transitionOut"] = "none"
    return _strip_internal(plan)


def _layer_plan(vocal: dict[str, Any] | None, bed: dict[str, Any] | None, vocal_source: str, use_stems: bool) -> list[dict[str, Any]]:
    if not vocal or not bed:
        return []
    duration = min(float(vocal["end"]) - float(vocal["start"]), float(bed["end"]) - float(bed["start"]))
    duration = max(0.5, duration)
    vocal_mode = "vocals" if use_stems else "full_mix"
    bed_mode = "instrumental" if use_stems else "full_mix"
    vocal_item = _plan_item(vocal, 0.0, duration, vocal_mode, "none", "crossfade", gain_db=1.0)
    bed_item = _plan_item(bed, 0.0, duration, bed_mode, "none", "crossfade", gain_db=-1.5)
    outro_source = _pick_source_outro(vocal_source, vocal, bed)
    outro_start = max(0.0, duration - 4.0)
    outro = _plan_item(outro_source, outro_start, outro_start + (float(outro_source["end"]) - float(outro_source["start"])), "full_mix", "crossfade", "none", gain_db=-1.0)
    return _strip_internal([bed_item, vocal_item, outro])


def _plan_item(segment: dict[str, Any], start: float, end: float, layer_mode: str, transition_in: str, transition_out: str, gain_db: float = 0.0) -> dict[str, Any]:
    return {
        "id": f"item_{uuid.uuid4().hex[:8]}",
        "source": segment["source"],
        "trackId": segment["trackId"],
        "segmentId": segment["id"],
        "sourceStart": float(segment["start"]),
        "sourceEnd": float(segment["end"]),
        "timelineStart": round(float(start), 3),
        "timelineEnd": round(float(end), 3),
        "layerMode": layer_mode,
        "gainDb": gain_db,
        "transitionIn": transition_in,
        "transitionOut": transition_out,
        "crossfadeSec": 4.0 if transition_in != "none" or transition_out != "none" else 0.0,
        "_segment": segment,
    }


def _render_plan_audio(plan: list[dict[str, Any]], tracks_by_id: dict[str, dict[str, Any]], *, use_stems: bool) -> np.ndarray:
    end_time = max(float(item.get("timelineEnd", 0)) for item in plan)
    output = np.zeros((2, max(1, int(np.ceil(end_time * SAMPLE_RATE))) + SAMPLE_RATE), dtype=np.float32)
    sorted_items = sorted(plan, key=lambda item: float(item.get("timelineStart", 0)))
    previous_by_lane: dict[str, dict[str, Any]] = {}
    for item in sorted_items:
        track = tracks_by_id.get(str(item.get("trackId")))
        if not track:
            raise ValueError(f"Unknown track in mashup plan: {item.get('trackId')}")
        clip = _load_plan_clip(item, track, use_stems=use_stems)
        clip = _tempo_align(clip, track, tracks_by_id)
        clip *= _db_to_gain(float(item.get("gainDb", 0.0)))
        start_sample = max(0, int(float(item.get("timelineStart", 0)) * SAMPLE_RATE))
        end_sample = min(output.shape[1], start_sample + clip.shape[1])
        clip = clip[:, : end_sample - start_sample]
        lane = "main" if item.get("layerMode") == "full_mix" else f"layer_{item.get('timelineStart')}"
        fade_in = float(item.get("crossfadeSec") or 0.0) if item.get("transitionIn") in {"crossfade", "bass_swap", "filter_sweep"} else 0.0
        if fade_in > 0 and previous_by_lane.get(lane):
            samples = min(int(fade_in * SAMPLE_RATE), clip.shape[1], end_sample - start_sample)
            if samples > 0:
                fade_out, fade_curve_in = _fade_curves(samples, equal_power=True)
                existing = output[:, start_sample : start_sample + samples].copy()
                incoming = clip[:, :samples].copy()
                if item.get("transitionIn") == "filter_sweep":
                    incoming = _highpass_sweep(incoming)
                output[:, start_sample : start_sample + samples] = existing * fade_out + incoming * fade_curve_in
                output[:, start_sample + samples : end_sample] += clip[:, samples:]
            else:
                output[:, start_sample:end_sample] += clip
        else:
            output[:, start_sample:end_sample] += clip
        previous_by_lane[lane] = item
    return _peak_limit(output)


def _load_plan_clip(item: dict[str, Any], track: dict[str, Any], *, use_stems: bool) -> np.ndarray:
    mode = item.get("layerMode") if item.get("layerMode") in LAYER_MODES else "full_mix"
    track_id = str(item.get("trackId"))
    start = float(item.get("sourceStart", 0))
    end = float(item.get("sourceEnd", start))
    if mode != "full_mix" and use_stems:
        paths = _cached_stem_paths(track_id)
        if paths:
            if mode == "vocals":
                audio = _load_stereo(paths["vocals"])
            else:
                parts = [_load_stereo(paths[name]) for name in ("drums", "bass", "other")]
                max_len = max(part.shape[1] for part in parts)
                audio = np.zeros((2, max_len), dtype=np.float32)
                for part in parts:
                    audio[:, : part.shape[1]] += part
                audio = _peak_limit(audio)
            return _slice_audio(audio, start, end)
    return _slice_audio(_load_stereo(Path(track["path"])), start, end)


def _tempo_align(clip: np.ndarray, track: dict[str, Any], tracks_by_id: dict[str, dict[str, Any]]) -> np.ndarray:
    bpms = [float(item.get("bpm") or 0) for item in tracks_by_id.values() if float(item.get("bpm") or 0) > 0]
    source_bpm = float(track.get("bpm") or 0)
    if not bpms or source_bpm <= 0:
        return clip
    target_bpm = float(np.median(bpms))
    rate = float(np.clip(target_bpm / source_bpm, 0.88, 1.12))
    if abs(rate - 1.0) < 0.015:
        return clip
    stretched = librosa.effects.time_stretch(clip, rate=rate)
    return np.ascontiguousarray(stretched, dtype=np.float32)


def _load_stereo(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=False)
    if y.ndim == 1:
        y = np.vstack([y, y])
    if y.shape[0] > 2:
        y = y[:2]
    return np.ascontiguousarray(y, dtype=np.float32)


def _load_mono(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return np.ascontiguousarray(y, dtype=np.float32)


def _slice_audio(audio: np.ndarray, start: float, end: float) -> np.ndarray:
    start_sample = max(0, int(start * SAMPLE_RATE))
    end_sample = max(start_sample, min(audio.shape[1], int(end * SAMPLE_RATE)))
    return np.ascontiguousarray(audio[:, start_sample:end_sample], dtype=np.float32)


def _cached_stem_paths(track_id: str) -> dict[str, Path] | None:
    root = STEM_DIR / track_id / "demucs_api"
    stems = {name: root / f"{name}.wav" for name in STEM_NAMES}
    return stems if all(path.exists() for path in stems.values()) else None


def _base_warnings(track_a: dict[str, Any], track_b: dict[str, Any]) -> list[str]:
    warnings = []
    key_eval = camelot_key_distance(track_a.get("camelot"), track_b.get("camelot"))
    if float(key_eval.get("score") or 0) < 65:
        warnings.append(f"Camelot relationship is {key_eval.get('relation')}; consider harmonic tuning before rendering.")
    bpm_a = float(track_a.get("bpm") or 0)
    bpm_b = float(track_b.get("bpm") or 0)
    if bpm_a > 0 and bpm_b > 0 and min(bpm_a, bpm_b) / max(bpm_a, bpm_b) < 0.88:
        warnings.append("BPM gap exceeds the transparent stretch range; render limits stretch to 0.88x-1.12x.")
    return warnings


def _pick(segments: list[dict[str, Any]], labels: set[str]) -> dict[str, Any] | None:
    pool = [item for item in segments if item.get("label") in labels]
    if not pool:
        pool = segments
    return max(pool, key=lambda item: float(item.get("energy", 0)) + float(item.get("mixInScore", 0)) * 0.2) if pool else None


def _best(anchor: dict[str, Any], candidates: list[dict[str, Any]], labels: set[str]) -> dict[str, Any] | None:
    pool = [item for item in candidates if item.get("label") in labels] or candidates
    return max(pool, key=lambda item: segment_compatibility(anchor, item)["score"]) if pool else None


def _ensure_both_sources(selected: list[dict[str, Any]], a: list[dict[str, Any]], b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = {item.get("source") for item in selected}
    if "A" not in sources and a:
        selected.insert(0, a[0])
    if "B" not in sources and b:
        selected.append(b[-1])
    return selected


def _pick_source_outro(vocal_source: str, vocal: dict[str, Any], bed: dict[str, Any]) -> dict[str, Any]:
    return vocal if vocal_source == "A" else bed


def _strip_internal(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in item.items() if not key.startswith("_")} for item in plan]


def _highpass_sweep(buffer: np.ndarray) -> np.ndarray:
    try:
        return signal.sosfilt(signal.butter(2, 180, "highpass", fs=SAMPLE_RATE, output="sos"), buffer, axis=1).astype(np.float32)
    except Exception:
        return buffer


def _peak_limit(buffer: np.ndarray) -> np.ndarray:
    out = np.nan_to_num(buffer, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0.98:
        out *= 0.98 / peak
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _db_to_gain(db: float) -> float:
    return float(10 ** (db / 20.0))
