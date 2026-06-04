from __future__ import annotations

from typing import Any

import numpy as np


SECTION_LABELS = {
    "intro": "Intro",
    "verse": "Verse",
    "build": "Build",
    "drop_chorus": "Drop / Chorus",
    "break": "Break",
    "transition": "Transition",
    "outro": "Outro",
}


def refine_section_labels(sections: list[dict[str, Any]], bar_features: list[dict[str, Any] | Any] | None = None) -> list[dict[str, Any]]:
    if not sections:
        return sections
    total = max(float(sections[-1].get("end", 0.0)), 1.0)
    profiles = _add_relative_context([_profile(section, bar_features) for section in sections])
    enriched = []
    for index, section in enumerate(sections):
        previous = sections[index - 1] if index > 0 else None
        nxt = sections[index + 1] if index + 1 < len(sections) else None
        profile = profiles[index]
        prev_profile = profiles[index - 1] if index > 0 else None
        next_profile = profiles[index + 1] if index + 1 < len(profiles) else None
        section_type, confidence, reasons = _classify(section, profile, previous, nxt, prev_profile, next_profile, index, len(sections), total)
        item = dict(section)
        item["rawLabel"] = section.get("rawLabel") or section.get("label")
        item["rawSectionType"] = section.get("sectionType")
        item["rawSectionLabel"] = section.get("sectionLabel")
        item["sectionType"] = section_type
        item["sectionLabel"] = SECTION_LABELS[section_type]
        item["label"] = item["sectionLabel"]
        item["sectionSubLabel"] = _sub_label(section_type, profile, section)
        item["arrangementLevel"] = _arrangement_level(profile)
        item["layerProfile"] = {
            "energy": round(profile["energy"], 4),
            "energyStart": round(profile["energy_start"], 4),
            "energyEnd": round(profile["energy_end"], 4),
            "energyDelta": round(profile["energy_delta"], 4),
            "vocal": round(profile["vocal"], 4),
            "drums": round(profile["drums"], 4),
            "bass": round(profile["bass"], 4),
            "brightness": round(profile["brightness"], 4),
            "density": round(profile["density"], 4),
            "tension": round(profile["tension"], 4),
            "repetition": round(float(section.get("repetitionScore", 0.0)), 4),
        }
        item["labelConfidence"] = round(confidence, 4)
        item["labelReasons"] = reasons[:5]
        enriched.append(item)
    return _merge_adjacent_same_labels(enriched)


def _classify(
    section: dict[str, Any],
    profile: dict[str, float],
    previous: dict[str, Any] | None,
    nxt: dict[str, Any] | None,
    prev_profile: dict[str, float] | None,
    next_profile: dict[str, float] | None,
    index: int,
    count: int,
    total_duration: float,
) -> tuple[str, float, list[str]]:
    pos = float(section.get("start", 0.0)) / total_duration
    old = str(section.get("sectionType") or "")
    label = str(section.get("label") or "")
    repetition = float(section.get("repetitionScore", 0.0))
    reasons: list[str] = []

    if index == 0 and pos < 0.12 and profile["vocal"] < 0.42:
        return "intro", 0.86, ["first low-vocal section", "early song position"]
    if index == count - 1 and pos > 0.72 and (profile["energy_delta"] < -0.08 or profile["energy"] < 0.58):
        return "outro", 0.82, ["last section", "late song position"]

    next_energy_jump = (next_profile["energy"] - profile["energy"]) if next_profile else 0.0
    prev_energy_jump = (profile["energy"] - prev_profile["energy"]) if prev_profile else 0.0
    next_bass_jump = (next_profile["bass"] - profile["bass"]) if next_profile else 0.0
    prev_bass_jump = (profile["bass"] - prev_profile["bass"]) if prev_profile else 0.0
    density_rank = profile.get("density_rank", 0.5)
    drum_rank = profile.get("drums_rank", 0.5)
    bass_rank = profile.get("bass_rank", 0.5)
    vocal_rank = profile.get("vocal_rank", 0.5)
    tension_rank = profile.get("tension_rank", 0.5)

    build_score = (
        max(0.0, profile["energy_delta"]) * 1.8
        + max(0.0, next_energy_jump) * 0.8
        + max(0.0, next_bass_jump) * 0.5
        + tension_rank * 0.28
        + (0.12 if label == "pre_chorus_like" or old == "build" else 0.0)
    )
    if build_score >= 0.34 and nxt is not None and (next_energy_jump > 0.08 or profile["energy_delta"] > 0.08 or tension_rank >= 0.72):
        reasons.extend(["energy/tension rises", "next section is stronger"])
        return "build", float(np.clip(0.58 + build_score * 0.28, 0, 0.93)), reasons

    break_score = (
        (1.0 - drum_rank) * 0.22
        + (1.0 - bass_rank) * 0.18
        + max(0.0, -profile["energy_delta"]) * 0.58
        + max(0.0, -prev_energy_jump) * 0.42
        + max(0.0, -prev_bass_jump) * 0.26
    )
    post_peak_break = previous is not None and prev_energy_jump < -0.16 and profile["energy"] < max(0.62, prev_profile["energy"] * 0.82 if prev_profile else 0.62)
    low_relative_layer = density_rank <= 0.22 and (drum_rank <= 0.35 or bass_rank <= 0.35)
    if post_peak_break or low_relative_layer or (profile["energy"] < 0.42 and profile["drums"] < 0.45) or break_score >= 0.58 or label == "breakdown_like":
        reasons = ["reduced rhythm/low-end layer"]
        if post_peak_break:
            reasons.append("energy drops after a fuller section")
        return "break", float(np.clip(0.55 + break_score * 0.34, 0, 0.92)), reasons

    drop_score = (
        drum_rank * 0.26
        + bass_rank * 0.24
        + density_rank * 0.20
        + max(0.0, prev_energy_jump) * 0.14
        + max(0.0, prev_bass_jump) * 0.12
        + repetition * 0.14
        + (0.16 if label in {"chorus_like", "drop_like"} or old == "drop_chorus" else 0.0)
    )
    repeated_hook = repetition >= 0.62 and (profile["vocal"] >= 0.34 or vocal_rank >= 0.55) and density_rank >= 0.42
    post_build_drop = previous is not None and str(previous.get("sectionType")) == "build" and profile["energy"] >= 0.50
    raw_chorus_peak = label == "chorus_like" and density_rank >= 0.55 and vocal_rank >= 0.45
    raw_drop_peak = label == "drop_like" and (bass_rank >= 0.45 or drum_rank >= 0.45)
    if drop_score >= 0.62 and (repeated_hook or post_build_drop or raw_drop_peak or raw_chorus_peak):
        if repeated_hook:
            reasons.append("repeated high-energy hook")
        if post_build_drop:
            reasons.append("arrives after build")
        if label == "drop_like":
            reasons.append("strong drum/bass section")
        if raw_chorus_peak:
            reasons.append("raw chorus section stays peak/hook")
        return "drop_chorus", float(np.clip(0.56 + drop_score * 0.28, 0, 0.95)), reasons or ["high-energy repeated section"]

    transition_score = abs(profile["energy_delta"]) + profile["tension"] * 0.4
    if transition_score >= 0.36 and vocal_rank < 0.55:
        return "transition", float(np.clip(0.54 + transition_score * 0.32, 0, 0.88)), ["short-term arrangement change"]

    if label == "chorus_like" and density_rank < 0.55:
        return "transition", 0.62, ["raw chorus label but not a relative peak"]

    if profile["vocal"] >= 0.26 or vocal_rank >= 0.45 or label == "verse_like" or old == "verse":
        return "verse", float(np.clip(0.54 + vocal_rank * 0.18 + (1 - repetition) * 0.08, 0, 0.84)), ["lead vocal or narrative section"]

    return "transition", 0.52, ["fallback structural bridge"]


def _merge_adjacent_same_labels(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(sections) < 2:
        return sections
    merged: list[dict[str, Any]] = []
    for section in sections:
        if merged and _can_merge(merged[-1], section):
            merged[-1] = _merge_pair(merged[-1], section)
        else:
            merged.append(dict(section))
    for index, section in enumerate(merged, start=1):
        section["id"] = section.get("id") or f"section_{index:03d}"
    return merged


def _can_merge(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("sectionType") != right.get("sectionType"):
        return False
    gap = float(right.get("start", 0.0) or 0.0) - float(left.get("end", 0.0) or 0.0)
    if gap > 0.2:
        return False
    return bool(left.get("sectionType"))


def _merge_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_duration = max(float(left.get("end", 0.0) or 0.0) - float(left.get("start", 0.0) or 0.0), 0.001)
    right_duration = max(float(right.get("end", 0.0) or 0.0) - float(right.get("start", 0.0) or 0.0), 0.001)
    total_duration = left_duration + right_duration
    item = dict(left)
    item["end"] = right.get("end", left.get("end"))
    item["barEnd"] = right.get("barEnd", left.get("barEnd"))
    item["bars"] = max(0, int(item.get("barEnd", 0) or 0) - int(item.get("barStart", 0) or 0))
    item["rawLabel"] = _join_unique([left.get("rawLabel"), right.get("rawLabel")], " + ")
    item["rawSectionType"] = _join_unique([left.get("rawSectionType"), right.get("rawSectionType")], " + ")
    item["rawSectionLabel"] = _join_unique([left.get("rawSectionLabel"), right.get("rawSectionLabel")], " + ")
    item["sectionSubLabel"] = _dominant_text(left, right, "sectionSubLabel", left_duration, right_duration)
    item["arrangementLevel"] = _dominant_text(left, right, "arrangementLevel", left_duration, right_duration)
    item["labelReasons"] = _dedupe([*(left.get("labelReasons") or []), *(right.get("labelReasons") or []), "merged adjacent same-label sections"])[:5]
    item["riskFlags"] = _dedupe([*(left.get("riskFlags") or []), *(right.get("riskFlags") or [])])
    item["similarSectionIds"] = _dedupe([*(left.get("similarSectionIds") or []), *(right.get("similarSectionIds") or [])])
    item["entryClean"] = bool(left.get("entryClean", True))
    item["exitClean"] = bool(right.get("exitClean", True))
    for key in ("confidence", "meanEnergy", "vocalDensity", "drumActivity", "bassEnergy", "brightness", "loopability", "grooveBedScore", "vocalPhraseScore", "groupConfidence", "repetitionScore", "labelConfidence"):
        if key in left or key in right:
            item[key] = round(_weighted(left.get(key), right.get(key), left_duration, right_duration), 4)
    if "mixInScore" in left:
        item["mixInScore"] = left.get("mixInScore")
    if "mixOutScore" in right:
        item["mixOutScore"] = right.get("mixOutScore")
    if isinstance(left.get("layerProfile"), dict) or isinstance(right.get("layerProfile"), dict):
        item["layerProfile"] = _merge_layer_profile(left.get("layerProfile") or {}, right.get("layerProfile") or {}, left_duration, right_duration)
    return item


def _merge_layer_profile(left: dict[str, Any], right: dict[str, Any], left_duration: float, right_duration: float) -> dict[str, float]:
    keys = sorted(set(left) | set(right))
    merged = {}
    for key in keys:
        merged[key] = round(_weighted(left.get(key), right.get(key), left_duration, right_duration), 4)
    return merged


def _weighted(left: Any, right: Any, left_duration: float, right_duration: float) -> float:
    left_value = float(left or 0.0)
    right_value = float(right if right is not None else left_value)
    return float((left_value * left_duration + right_value * right_duration) / max(left_duration + right_duration, 0.001))


def _dominant_text(left: dict[str, Any], right: dict[str, Any], key: str, left_duration: float, right_duration: float) -> Any:
    left_value = left.get(key)
    right_value = right.get(key)
    if left_value == right_value or not right_value:
        return left_value
    if not left_value:
        return right_value
    return left_value if left_duration >= right_duration else right_value


def _join_unique(values: list[Any], sep: str) -> str | None:
    cleaned = [str(value) for value in values if value]
    unique = _dedupe(cleaned)
    return sep.join(unique) if unique else None


def _dedupe(values: list[Any]) -> list[Any]:
    result = []
    seen = set()
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _profile(section: dict[str, Any] | None, bar_features: list[dict[str, Any] | Any] | None = None) -> dict[str, float]:
    if not section:
        return {key: 0.0 for key in ("energy", "energy_start", "energy_end", "energy_delta", "vocal", "drums", "bass", "brightness", "density", "tension")}
    start = int(section.get("barStart", 0))
    end = int(section.get("barEnd", start))
    bars = []
    if bar_features:
        bars = bar_features[max(0, start):max(start, end)]
    if bars:
        energy_values = [_get(bar, "energy") for bar in bars]
        vocal_values = [_get(bar, "vocalDensity") for bar in bars]
        drum_values = [_get(bar, "drumActivity") for bar in bars]
        bass_values = [_get(bar, "bassEnergy") for bar in bars]
        bright_values = [_get(bar, "spectralCentroid") / 8000.0 for bar in bars]
        flux_values = [_get(bar, "spectralFlux") for bar in bars]
        energy = float(np.mean(energy_values))
        energy_start = float(energy_values[0])
        energy_end = float(energy_values[-1])
        return {
            "energy": energy,
            "energy_start": energy_start,
            "energy_end": energy_end,
            "energy_delta": float(energy_end - energy_start),
            "vocal": float(np.mean(vocal_values)),
            "drums": float(np.mean(drum_values)),
            "bass": float(np.mean(bass_values)),
            "brightness": float(np.mean(bright_values)),
            "density": float(np.clip(np.mean([energy, np.mean(drum_values), np.mean(bass_values), np.mean(vocal_values)]), 0, 1)),
            "tension": float(np.clip(np.mean(flux_values) * 10.0 + max(0.0, energy_end - energy_start), 0, 1)),
        }
    energy = float(section.get("meanEnergy", section.get("energy", 0.0)) or 0.0)
    energy_start = float(section.get("energyStart", energy) or energy)
    energy_end = float(section.get("energyEnd", energy) or energy)
    vocal = float(section.get("vocalDensity", 0.0) or 0.0)
    drums = float(section.get("drumActivity", 0.0) or 0.0)
    bass = float(section.get("bassEnergy", 0.0) or 0.0)
    brightness = float(section.get("brightness", 0.0) or 0.0)
    return {
        "energy": energy,
        "energy_start": energy_start,
        "energy_end": energy_end,
        "energy_delta": float(section.get("energyDelta", energy_end - energy_start) or 0.0),
        "vocal": vocal,
        "drums": drums,
        "bass": bass,
        "brightness": brightness,
        "density": float(np.clip(np.mean([energy, drums, bass, vocal]), 0, 1)),
        "tension": float(np.clip(max(0.0, energy_end - energy_start) + brightness * 0.25, 0, 1)),
    }


def _add_relative_context(profiles: list[dict[str, float]]) -> list[dict[str, float]]:
    if not profiles:
        return profiles
    result = [dict(profile) for profile in profiles]
    for key in ("energy", "vocal", "drums", "bass", "brightness", "density", "tension"):
        values = np.asarray([profile.get(key, 0.0) for profile in profiles], dtype=np.float32)
        if values.size == 1 or float(np.max(values) - np.min(values)) < 1e-5:
            ranks = np.full(values.size, 0.5, dtype=np.float32)
        else:
            order = np.argsort(values, kind="stable")
            ranks = np.zeros(values.size, dtype=np.float32)
            unique_values = np.unique(values)
            for value in unique_values:
                indices = np.where(values == value)[0]
                positions = [int(np.where(order == index)[0][0]) for index in indices]
                rank = float(np.mean(positions) / max(values.size - 1, 1))
                ranks[indices] = rank
        low = float(np.percentile(values, 20)) if values.size else 0.0
        high = float(np.percentile(values, 80)) if values.size else 1.0
        spread = max(high - low, 1e-6)
        for index, item in enumerate(result):
            item[f"{key}_rank"] = float(ranks[index])
            item[f"{key}_relative"] = float(np.clip((float(values[index]) - low) / spread, 0, 1))
    return result


def _sub_label(section_type: str, profile: dict[str, float], section: dict[str, Any]) -> str:
    if section_type == "verse" and profile["density"] >= 0.58:
        return "Verse - Full"
    if section_type == "verse":
        return "Verse - Sparse" if profile["density"] < 0.42 else "Verse"
    if section_type == "build":
        return "Build - Rising"
    if section_type == "drop_chorus":
        return "Drop" if profile["vocal"] < 0.34 else "Chorus / Hook"
    if section_type == "break":
        return "Break - Low Layer"
    return SECTION_LABELS.get(section_type, "Transition")


def _arrangement_level(profile: dict[str, float]) -> str:
    density = profile["density"]
    if profile["energy_delta"] > 0.18:
        return "rising"
    if profile["energy_delta"] < -0.18:
        return "falling"
    if density >= 0.68:
        return "peak"
    if density >= 0.48:
        return "medium"
    return "sparse"


def _get(item: dict[str, Any] | Any, key: str) -> float:
    if isinstance(item, dict):
        return float(item.get(key, 0.0) or 0.0)
    return float(getattr(item, key, 0.0) or 0.0)
