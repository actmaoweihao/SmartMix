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


def key_label_to_camelot(label: str | None, mode: str | None = None) -> str | None:
    if not label or label == "Unknown" or label == "未知":
        return None
    parts = label.replace("Major", "Maj").replace("Minor", "Min").split()
    if not parts:
        return None
    root = parts[0]
    inferred = (mode or (parts[1] if len(parts) > 1 else "")).lower()
    key = f"{root}m" if inferred.startswith("min") or inferred == "minor" else root
    return KEY_TO_CAMELOT.get(key)


def parse_camelot(code: str) -> tuple[int, str]:
    return int(code[:-1]), code[-1]


def camelot_num_dist(n1: int, n2: int) -> int:
    diff = abs(n1 - n2)
    return min(diff, 12 - diff)


def is_perfect_fifth(n1: int, n2: int) -> bool:
    return camelot_num_dist(n1, n2) == 5


def camelot_key_distance(code1: str | None, code2: str | None) -> dict[str, Any]:
    if not code1 or not code2:
        return {
            "distance": 4,
            "score": 50,
            "rank": "未知",
            "reason": "至少一首歌调性未知，使用中性调性评分。",
        }

    n1, m1 = parse_camelot(code1)
    n2, m2 = parse_camelot(code2)
    d_num = camelot_num_dist(n1, n2)

    if n1 == n2:
        distance = 0
        reason = "同号 A/B 或同 Camelot 数字，关系大小调兼容。"
    elif m1 == m2:
        distance = d_num
        reason = "同为大调或同为小调，按 Camelot 轮盘相邻距离评分。"
    else:
        distance = d_num + 2
        reason = "大小调不同且 Camelot 数字不同，增加模式切换惩罚。"

    if is_perfect_fifth(n1, n2):
        distance -= 1
        reason += " 纯五度关系，加分。"
    if d_num >= 6:
        distance += 1
        reason += " Camelot 距离较远，增加惩罚。"

    distance = max(0, distance)
    score = _camelot_distance_to_score(distance)
    return {
        "distance": distance,
        "score": score,
        "rank": camelot_rank(distance),
        "reason": reason,
    }


def camelot_rank(distance: int | float) -> str:
    if distance == 0:
        return "完美"
    if 1 <= distance <= 2:
        return "推荐"
    if 3 <= distance <= 4:
        return "可用"
    return "避坑"


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
    energy_eval = energy_match_score(prev_track.get("energy"), next_track.get("energy"))
    structure_eval = structure_match_score(prev_track, next_track)

    total = round(
        key_eval["score"] * 0.45
        + bpm_eval["score"] * 0.30
        + energy_eval["score"] * 0.15
        + structure_eval["score"] * 0.10,
        1,
    )
    return {
        "direction": f"{prev_track.get('name', 'A')} → {next_track.get('name', 'B')}",
        "total_score": total,
        "level": total_rank(total),
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
        return {"score": 50, "delta": None, "reason": "至少一首歌 BPM 未知，使用中性节奏评分。"}
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
        "reason": "比较 BPM 差，并自动考虑 half/double tempo 关系。",
    }


def energy_match_score(energy_a: Any, energy_b: Any) -> dict[str, Any]:
    if energy_a is None or energy_b is None:
        return {"score": 55, "delta": None, "reason": "至少一首歌能量未知，使用中性能量评分。"}
    delta = abs(float(energy_a) - float(energy_b))
    score = max(0, min(100, 100 - delta * 120))
    return {
        "score": round(score, 1),
        "delta": round(delta, 3),
        "reason": "能量越接近，现场听感越容易自然衔接。",
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
        "reason": "根据上一首可用尾段、下一首可用头段和 BPM 推算可承载几小节重叠。",
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


def total_rank(score: float) -> str:
    if score >= 90:
        return "完美"
    if score >= 75:
        return "推荐"
    if score >= 60:
        return "可用"
    return "避坑"


def _track_camelot(track: dict[str, Any]) -> str | None:
    return track.get("camelot") or key_label_to_camelot(track.get("key"), track.get("mode"))


def _track_summary(track: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": track.get("name"),
        "bpm": track.get("bpm"),
        "key": track.get("key"),
        "camelot": _track_camelot(track),
        "energy": track.get("energy"),
        "duration": track.get("duration"),
    }


def _camelot_distance_to_score(distance: int | float) -> float:
    if distance <= 0:
        return 100
    if distance <= 1:
        return 92
    if distance <= 2:
        return 84
    if distance <= 3:
        return 72
    if distance <= 4:
        return 62
    return max(0, 50 - (distance - 5) * 12)

