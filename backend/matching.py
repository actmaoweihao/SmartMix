from __future__ import annotations

from typing import Any

from .tuning import recommend_pair_tuning
from .transition import plan_transition


KEY_TO_CAMELOT = {
    # Major keys, Camelot B.
    "C": "8B",
    "C#": "3B",
    "Db": "3B",
    "D": "10B",
    "D#": "5B",
    "Eb": "5B",
    "E": "12B",
    "F": "7B",
    "F#": "2B",
    "Gb": "2B",
    "G": "9B",
    "G#": "4B",
    "Ab": "4B",
    "A": "11B",
    "A#": "6B",
    "Bb": "6B",
    "B": "1B",
    # Minor keys, Camelot A.
    "Am": "8A",
    "A#m": "3A",
    "Bbm": "3A",
    "Bm": "10A",
    "Cm": "5A",
    "C#m": "12A",
    "Dbm": "12A",
    "Dm": "7A",
    "D#m": "2A",
    "Ebm": "2A",
    "Em": "9A",
    "Fm": "4A",
    "F#m": "11A",
    "Gbm": "11A",
    "Gm": "6A",
    "G#m": "1A",
    "Abm": "1A",
}

UNKNOWN_KEY_LABELS = {"unknown", "未知"}


def key_label_to_camelot(label: str | None, mode: str | None = None) -> str | None:
    if not label:
        return None
    raw = str(label).strip().replace("♯", "#").replace("♭", "b")
    if not raw or raw.lower() in UNKNOWN_KEY_LABELS:
        return None

    direct = parse_camelot(raw)
    if direct:
        return f"{direct[0]}{direct[1]}"

    parts = raw.replace("Major", "Maj").replace("Minor", "Min").split()
    if not parts:
        return None
    root = parts[0]
    inferred = (mode or (parts[1] if len(parts) > 1 else "")).lower()
    compact_minor = root.endswith("m") and root[:-1] in KEY_TO_CAMELOT
    key = root if compact_minor else f"{root}m" if inferred.startswith("min") or inferred == "minor" else root
    return KEY_TO_CAMELOT.get(key)


def parse_camelot(code: str | None) -> tuple[int, str] | None:
    if not code:
        return None
    raw = str(code).strip().upper()
    if len(raw) < 2 or raw[-1] not in {"A", "B"} or not raw[:-1].isdigit():
        return None
    number = int(raw[:-1])
    if number < 1 or number > 12:
        return None
    return number, raw[-1]


def camelot_num_dist(n1: int, n2: int) -> int:
    diff = abs(n1 - n2)
    return min(diff, 12 - diff)


def camelot_clockwise_delta(n1: int, n2: int) -> int:
    forward = (n2 - n1) % 12
    if forward == 0:
        return 0
    return forward if forward <= 7 else forward - 12


def camelot_key_distance(code1: str | None, code2: str | None) -> dict[str, Any]:
    parsed1 = parse_camelot(code1)
    parsed2 = parse_camelot(code2)
    warnings: list[str] = []
    if code1 and not parsed1:
        warnings.append(f"Invalid Camelot key: {code1}")
    if code2 and not parsed2:
        warnings.append(f"Invalid Camelot key: {code2}")

    if not parsed1 or not parsed2:
        return _key_result(
            code1,
            code2,
            None,
            None,
            None,
            40,
            "unknown",
            "At least one track has an unknown or invalid Camelot key; prefer short cuts, fades, or FX transitions.",
            warnings,
        )

    n1, m1 = parsed1
    n2, m2 = parsed2
    normalized1 = f"{n1}{m1}"
    normalized2 = f"{n2}{m2}"
    d_num = camelot_num_dist(n1, n2)
    clockwise = camelot_clockwise_delta(n1, n2)

    if n1 == n2 and m1 == m2:
        return _key_result(
            code1,
            code2,
            parsed1,
            parsed2,
            0,
            100,
            "same",
            f"{normalized1} -> {normalized2} is the same Camelot key.",
            warnings,
        )
    if n1 == n2 and m1 != m2:
        return _key_result(
            code1,
            code2,
            parsed1,
            parsed2,
            0,
            93,
            "relative_major_minor",
            f"{normalized1} -> {normalized2} is the relative major/minor pair on the Camelot Wheel.",
            warnings,
        )
    if m1 == m2 and d_num == 1:
        return _key_result(
            code1,
            code2,
            parsed1,
            parsed2,
            1,
            88,
            "adjacent",
            f"{normalized1} -> {normalized2} is an adjacent Camelot key with wrap-around support.",
            warnings,
        )
    if m1 == m2 and clockwise == 2:
        return _key_result(
            code1,
            code2,
            parsed1,
            parsed2,
            2,
            80,
            "energy_boost",
            f"{normalized1} -> {normalized2} is Energy Boost (+2 on the Camelot Wheel); use with controlled overlap.",
            warnings,
        )
    if is_diagonal_mix(m1, m2, clockwise):
        return _key_result(
            code1,
            code2,
            parsed1,
            parsed2,
            d_num,
            74,
            "diagonal_mix",
            f"{normalized1} -> {normalized2} is Diagonal Mix; treat it as a special-effect transition.",
            warnings,
        )
    if m1 == m2 and clockwise == 7:
        return _key_result(
            code1,
            code2,
            parsed1,
            parsed2,
            d_num,
            62,
            "jaws_mix",
            f"{normalized1} -> {normalized2} is Jaw's Mix (+7); use short cuts or dramatic sections.",
            warnings,
        )
    if is_mood_shifter(m1, m2, clockwise):
        return _key_result(
            code1,
            code2,
            parsed1,
            parsed2,
            d_num,
            66,
            "mood_shifter",
            f"{normalized1} -> {normalized2} is Mood Shifter (+4 across A/B); use short blends or section switches.",
            warnings,
        )

    return _key_result(
        code1,
        code2,
        parsed1,
        parsed2,
        d_num,
        24,
        "clash",
        f"{normalized1} -> {normalized2} is not same, adjacent, or relative major/minor; avoid long blends.",
        warnings,
    )


def _key_result(
    code1: str | None,
    code2: str | None,
    parsed1: tuple[int, str] | None,
    parsed2: tuple[int, str] | None,
    distance: int | None,
    score: float,
    relation: str,
    reason: str,
    warnings: list[str],
) -> dict[str, Any]:
    normalized1 = f"{parsed1[0]}{parsed1[1]}" if parsed1 else None
    normalized2 = f"{parsed2[0]}{parsed2[1]}" if parsed2 else None
    return {
        "distance": distance,
        "score": score,
        "rank": camelot_rank(relation),
        "relation": relation,
        "reason": reason,
        "debug": {
            "inputA": code1,
            "inputB": code2,
            "normalizedA": normalized1,
            "normalizedB": normalized2,
            "parsedA": {"number": parsed1[0], "letter": parsed1[1]} if parsed1 else None,
            "parsedB": {"number": parsed2[0], "letter": parsed2[1]} if parsed2 else None,
            "relation": relation,
            "score": score,
            "explanation": reason,
            "warnings": warnings,
        },
    }


def camelot_rank(relation: str) -> str:
    if relation == "same":
        return "perfect"
    if relation in {"relative_major_minor", "adjacent", "energy_boost"}:
        return "recommended"
    if relation in {"diagonal_mix", "jaws_mix", "mood_shifter"}:
        return "special_effect"
    if relation == "unknown":
        return "unknown"
    return "clash"


def is_diagonal_mix(mode1: str, mode2: str, clockwise: int) -> bool:
    if mode1 == mode2:
        return False
    return (mode1 == "A" and mode2 == "B" and clockwise == -1) or (mode1 == "B" and mode2 == "A" and clockwise == 1)


def is_mood_shifter(mode1: str, mode2: str, clockwise: int) -> bool:
    if mode1 == mode2:
        return False
    return (mode1 == "A" and mode2 == "B" and clockwise == 3) or (mode1 == "B" and mode2 == "A" and clockwise == -3)


def evaluate_track_match(track_a: dict[str, Any], track_b: dict[str, Any]) -> dict[str, Any]:
    forward = evaluate_direction(track_a, track_b)
    reverse = evaluate_direction(track_b, track_a)
    best = forward if forward["total_score"] >= reverse["total_score"] else reverse
    tuning_recommendations = recommend_pair_tuning(track_a, track_b)
    return {
        "track_a": _track_summary(track_a),
        "track_b": _track_summary(track_b),
        "overall_score": best["total_score"],
        "overall_level": best["level"],
        "recommended_direction": best["direction"],
        "tuning_recommendations": tuning_recommendations,
        "directions": {
            "a_to_b": forward,
            "b_to_a": reverse,
        },
    }


def evaluate_direction(prev_track: dict[str, Any], next_track: dict[str, Any]) -> dict[str, Any]:
    prev_camelot = _track_camelot(prev_track)
    next_camelot = _track_camelot(next_track)
    key_eval = camelot_key_distance(prev_camelot, next_camelot)
    bpm_eval = bpm_match_score(prev_track.get("bpm"), next_track.get("bpm"))
    energy_eval = energy_match_score(prev_track, next_track)
    structure_eval = structure_match_score(prev_track, next_track)

    raw_total = round(
        key_eval["score"] * 0.45
        + bpm_eval["score"] * 0.30
        + energy_eval["score"] * 0.15
        + structure_eval["score"] * 0.10,
        1,
    )
    total = adjusted_total_score(raw_total, key_eval, bpm_eval, energy_eval, structure_eval)
    return {
        "direction": f"{prev_track.get('name', 'A')} -> {next_track.get('name', 'B')}",
        "total_score": total,
        "raw_total_score": raw_total,
        "level": total_rank(total, key_eval, bpm_eval, energy_eval, structure_eval),
        "components": {
            "camelot": {
                **key_eval,
                "from": prev_camelot,
                "to": next_camelot,
                "weight": 0.45,
            },
            "bpm": {**bpm_eval, "weight": 0.30},
            "energy": {**energy_eval, "weight": 0.15},
            "structure": {**structure_eval, "weight": 0.10},
        },
        "transition": recommend_transition(prev_track, next_track),
    }


def bpm_match_score(bpm_a: Any, bpm_b: Any) -> dict[str, Any]:
    if not bpm_a or not bpm_b:
        return {"score": 50, "delta": None, "reason": "At least one track has unknown BPM; using neutral rhythm score."}
    a = float(bpm_a)
    candidates = [float(bpm_b), float(bpm_b) * 2, float(bpm_b) / 2]
    b = min(candidates, key=lambda item: abs(a - item))
    delta = abs(a - b)
    pct = delta / max(a, 1)
    score = max(0, min(100, 100 - pct * 420))
    return {
        "score": round(score, 1),
        "delta": round(delta, 2),
        "normalized_bpm_b": round(b, 2),
        "reason": "Compares BPM difference and automatically considers half/double tempo relationships.",
    }


def energy_match_score(track_a: dict[str, Any], track_b: dict[str, Any]) -> dict[str, Any]:
    profile_a = track_a.get("energy_profile") or {}
    profile_b = track_b.get("energy_profile") or {}
    if not profile_a or not profile_b:
        energy_a = track_a.get("energy")
        energy_b = track_b.get("energy")
        if energy_a is None or energy_b is None:
            return {"score": 55, "delta": None, "reason": "At least one track has unknown energy; using neutral score."}
        delta = abs(float(energy_a) - float(energy_b))
        score = max(0, min(100, 100 - delta * 120))
        return {
            "score": round(score, 1),
            "delta": round(delta, 3),
            "summary": f"energy diff {delta:.3f}",
            "reason": "Fallback score based on legacy single energy value.",
        }

    sub_scores = {
        "energy_index": _score_delta(profile_a.get("energy_index"), profile_b.get("energy_index"), 35),
        "lufs": _score_delta(profile_a.get("lufs"), profile_b.get("lufs"), 10),
        "rms_body": _score_delta(profile_a.get("rms_p85_db"), profile_b.get("rms_p85_db"), 12),
        "crest_factor": _score_delta(profile_a.get("crest_factor_db"), profile_b.get("crest_factor_db"), 10),
        "low_frequency": _score_delta(profile_a.get("low_frequency_ratio"), profile_b.get("low_frequency_ratio"), 0.35),
        "dynamic_range": _score_delta(profile_a.get("dynamic_range_db"), profile_b.get("dynamic_range_db"), 12),
        "transition_shape": _score_delta(
            profile_a.get("outro_relative_energy"),
            profile_b.get("intro_relative_energy"),
            0.75,
        ),
    }
    weights = {
        "energy_index": 0.18,
        "lufs": 0.18,
        "rms_body": 0.14,
        "crest_factor": 0.12,
        "low_frequency": 0.14,
        "dynamic_range": 0.10,
        "transition_shape": 0.14,
    }
    score = sum(sub_scores[key]["score"] * weights[key] for key in weights)
    return {
        "score": round(float(score), 1),
        "delta": {key: sub_scores[key]["delta"] for key in sub_scores},
        "sub_scores": {key: round(float(sub_scores[key]["score"]), 1) for key in sub_scores},
        "summary": _energy_summary(profile_a, profile_b),
        "reason": "Compares LUFS, RMS percentiles, crest factor, low-frequency ratio, dynamic range, and outro-to-intro energy shape.",
    }


def structure_match_score(prev_track: dict[str, Any], next_track: dict[str, Any]) -> dict[str, Any]:
    transition = recommend_transition(prev_track, next_track)
    seconds = transition["seconds"]
    if seconds >= 24:
        score = 96
    elif seconds >= 16:
        score = 88
    elif seconds >= 8:
        score = 76
    elif seconds >= 4:
        score = 58
    else:
        score = 35
    return {
        "score": score,
        "overlap_seconds": seconds,
        "phrase_bars": transition["phrase_bars"],
        "reason": "Scores whether the previous outro and next intro can carry a phrase-length overlap.",
    }


def recommend_transition(prev_track: dict[str, Any], next_track: dict[str, Any]) -> dict[str, Any]:
    return plan_transition(
        prev_track,
        next_track,
        {
            "crossfade": 64,
            "autoTransition": False,
            "aiPrecision": True,
            "phraseBars": 16,
        },
    ).to_dict()


def adjusted_total_score(
    raw_score: float,
    key_eval: dict[str, Any],
    bpm_eval: dict[str, Any],
    energy_eval: dict[str, Any],
    structure_eval: dict[str, Any],
) -> float:
    """Keep weighted scoring, but stop one weak pillar from being hidden by Camelot/BPM."""
    score = float(raw_score)
    if key_eval.get("relation") == "clash":
        score = min(score, 79.0)
    if key_eval.get("relation") == "unknown":
        score = min(score, 84.0)
    if float(energy_eval.get("score") or 0) < 60:
        score = min(score, 89.0)
    if float(bpm_eval.get("score") or 0) < 70:
        score = min(score, 84.0)
    if float(structure_eval.get("score") or 0) < 70:
        score = min(score, 84.0)
    return round(score, 1)


def total_rank(
    score: float,
    key_eval: dict[str, Any] | None = None,
    bpm_eval: dict[str, Any] | None = None,
    energy_eval: dict[str, Any] | None = None,
    structure_eval: dict[str, Any] | None = None,
) -> str:
    if key_eval and key_eval.get("relation") == "clash":
        return "usable" if score >= 60 else "avoid"
    if key_eval and key_eval.get("relation") == "unknown" and score >= 85:
        return "recommended"
    if energy_eval and float(energy_eval.get("score") or 0) < 60 and score >= 90:
        return "recommended"
    if bpm_eval and float(bpm_eval.get("score") or 0) < 70 and score >= 85:
        return "usable"
    if structure_eval and float(structure_eval.get("score") or 0) < 70 and score >= 85:
        return "usable"
    return _rank_from_score(score)


def _rank_from_score(score: float) -> str:
    if score >= 90:
        return "perfect"
    if score >= 75:
        return "recommended"
    if score >= 60:
        return "usable"
    return "avoid"


def _track_camelot(track: dict[str, Any]) -> str | None:
    camelot = track.get("camelot")
    if camelot is not None:
        parsed = parse_camelot(camelot)
        return f"{parsed[0]}{parsed[1]}" if parsed else None
    return key_label_to_camelot(track.get("key"), track.get("mode"))


def _track_summary(track: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": track.get("name"),
        "bpm": track.get("bpm"),
        "key": track.get("key"),
        "camelot": _track_camelot(track),
        "energy": track.get("energy"),
        "energy_profile": track.get("energy_profile"),
        "duration": track.get("duration"),
    }


def _score_delta(a: Any, b: Any, tolerance: float) -> dict[str, float | None]:
    if a is None or b is None:
        return {"score": 55.0, "delta": None}
    delta = abs(float(a) - float(b))
    score = max(0.0, min(100.0, 100.0 - (delta / max(tolerance, 1e-9)) * 100.0))
    return {"score": score, "delta": round(float(delta), 4)}


def _energy_summary(profile_a: dict[str, Any], profile_b: dict[str, Any]) -> str:
    index_delta = abs(float(profile_a.get("energy_index", 0)) - float(profile_b.get("energy_index", 0)))
    lufs_delta = abs(float(profile_a.get("lufs", 0)) - float(profile_b.get("lufs", 0)))
    low_delta = abs(float(profile_a.get("low_frequency_ratio", 0)) - float(profile_b.get("low_frequency_ratio", 0)))
    return f"index diff {index_delta:.1f}, LUFS diff {lufs_delta:.1f}, low diff {low_delta:.2f}"
