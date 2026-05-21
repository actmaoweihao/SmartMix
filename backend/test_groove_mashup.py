from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

import backend.groove_mashup as groove
from backend.groove_mashup import (
    build_groove_mashup_plan,
    build_vocal_handoff_arrangement,
    choose_groove_pitch_policy,
    extract_vocal_phrases,
    find_candidate_groove_beds,
    render_groove_vocal_mashup,
)
from backend.mashup import build_mashup_plan
from backend.mixing import SAMPLE_RATE


def make_track(root: Path, track_id: str, source: str, bpm: float = 120.0, freq: float = 220.0, seconds: int = 16) -> dict:
    t = np.linspace(0, seconds, SAMPLE_RATE * seconds, endpoint=False, dtype=np.float32)
    mono = 0.06 * np.sin(2 * np.pi * freq * t)
    mono += 0.05 * np.sin(2 * np.pi * 70.0 * t)
    for click in np.arange(0, seconds, 1.0):
        sample = int(click * SAMPLE_RATE)
        mono[sample : sample + 64] += np.hanning(64).astype(np.float32) * 0.09
    path = root / f"{track_id}.wav"
    sf.write(path, np.vstack([mono, mono]).T, SAMPLE_RATE, subtype="PCM_16")
    bars = [float(i) for i in range(seconds + 1)]
    vocal_curve = []
    for time in np.arange(0, seconds, 0.5):
        active = any(start <= time < end for start, end in ((1, 3), (4, 6), (8, 10), (12, 14)))
        vocal_curve.append({"time": float(time), "density": 0.78 if active else 0.05})
    return {
        "id": track_id,
        "source": source,
        "name": f"{track_id}.wav",
        "path": str(path),
        "content_type": "audio/wav",
        "duration": float(seconds),
        "bpm": bpm,
        "key": "A minor",
        "camelot": "8A" if source == "A" else "9A",
        "bars": bars,
        "phrases": [0.0, 4.0, 8.0, 12.0, 16.0],
        "energy_profile": [{"time": float(i), "energy": 0.35 + 0.03 * (i % 4)} for i in range(seconds)],
        "transition_candidates": {"vocal_density_curve": vocal_curve},
    }


def write_stems(stem_root: Path, track_id: str, vocal_freq: float = 440.0, seconds: int = 16) -> None:
    root = stem_root / track_id / "demucs_api"
    root.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, SAMPLE_RATE * seconds, endpoint=False, dtype=np.float32)
    gate = np.zeros_like(t)
    for start, end in ((1, 3), (4, 6), (8, 10), (12, 14)):
        gate[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)] = 1.0
    stems = {
        "vocals": 0.10 * np.sin(2 * np.pi * vocal_freq * t) * gate,
        "drums": 0.045 * np.sin(2 * np.pi * 8.0 * t),
        "bass": 0.08 * np.sin(2 * np.pi * 55.0 * t),
        "other": 0.04 * np.sin(2 * np.pi * 220.0 * t),
    }
    for name, mono in stems.items():
        sf.write(root / f"{name}.wav", np.vstack([mono, mono]).T.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


class GrooveMashupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.uploads = self.root / "uploads"
        self.stems = self.root / "stems"
        self.uploads.mkdir()
        self.stems.mkdir()
        self.track_a = make_track(self.uploads, "a", "A", bpm=120, freq=220)
        self.track_b = make_track(self.uploads, "b", "B", bpm=122, freq=330)
        write_stems(self.stems, "a", 440)
        write_stems(self.stems, "b", 550)
        self.original_stem_dir = groove.STEM_DIR
        groove.STEM_DIR = self.stems

    def tearDown(self) -> None:
        groove.STEM_DIR = self.original_stem_dir
        self.tmp.cleanup()

    def test_find_groove_bed_excludes_vocals(self) -> None:
        beds = find_candidate_groove_beds(self.track_a, self.track_b, groove._cached_stem_paths("a"), groove._cached_stem_paths("b"))

        self.assertTrue(beds)
        self.assertTrue(all(bed["drumsSource"] in {"A", "B"} for bed in beds))
        self.assertTrue(all(bed["bassSource"] in {"A", "B"} for bed in beds))
        self.assertTrue(all("vocals" not in bed for bed in beds))
        self.assertLessEqual(beds[0]["vocalLeakage"], 0.15)

    def test_extract_vocal_phrases_short_units(self) -> None:
        phrases = extract_vocal_phrases(self.track_a, self.stems / "a" / "demucs_api" / "vocals.wav", phrase_bars=[2, 4, 8])

        self.assertGreaterEqual(len(phrases), 3)
        self.assertTrue(all(phrase["bars"] in {2, 4, 8} for phrase in phrases))
        self.assertLess(max(phrase["bars"] for phrase in phrases), 16)

    def test_vocal_handoff_arrangement_contains_both_sources(self) -> None:
        bed = find_candidate_groove_beds(self.track_a, self.track_b)[0]
        phrases_a = extract_vocal_phrases(self.track_a, self.stems / "a" / "demucs_api" / "vocals.wav")
        phrases_b = extract_vocal_phrases(self.track_b, self.stems / "b" / "demucs_api" / "vocals.wav")
        arrangement = build_vocal_handoff_arrangement(bed, phrases_a, phrases_b, "groove_vocal_handoff", 24)

        self.assertGreaterEqual(len(arrangement["vocalEvents"]), 3)
        self.assertEqual({event["source"] for event in arrangement["vocalEvents"]}, {"A", "B"})

    def test_vocal_over_bed_not_full_mix_crossfade(self) -> None:
        plan = build_groove_mashup_plan(self.track_a, self.track_b, mode="groove_vocal_handoff", target_duration_sec=24, use_stems=True)

        self.assertEqual(plan["groovePlan"]["status"], "ok")
        self.assertIn("bed", plan["groovePlan"])
        self.assertTrue(plan["groovePlan"]["vocalEvents"])
        self.assertTrue(all(item["layerMode"] == "vocals" for item in plan["plan"]))
        self.assertFalse(any(item.get("layerMode") == "full_mix" for item in plan["plan"]))

    def test_auto_mode_prefers_groove_plan_when_stems_exist(self) -> None:
        plan = build_mashup_plan(self.track_a, self.track_b, mode="auto", target_duration_sec=24, use_stems=True)

        self.assertEqual(plan["mode"], "groove_vocal_handoff")
        self.assertEqual(plan["groovePlan"]["status"], "ok")
        self.assertTrue(plan["groovePlan"]["bed"])
        self.assertTrue(all(item["layerMode"] == "vocals" for item in plan["plan"]))

    def test_auto_mode_requires_stems_instead_of_legacy_crossfade(self) -> None:
        groove.STEM_DIR = self.root / "empty_stems"
        groove.STEM_DIR.mkdir()
        plan = build_mashup_plan(self.track_a, self.track_b, mode="auto", target_duration_sec=24, use_stems=True)

        self.assertEqual(plan["groovePlan"]["status"], "stems_required")
        self.assertFalse(plan["plan"])
        self.assertTrue(any("did not fall back" in warning for warning in plan["warnings"]))

    def test_a_vocal_on_b_groove_forces_b_bed(self) -> None:
        plan = build_groove_mashup_plan(self.track_a, self.track_b, mode="a_vocal_on_b_groove", target_duration_sec=16, use_stems=True)

        self.assertEqual(plan["groovePlan"]["bed"]["source"], "B")
        self.assertEqual(plan["groovePlan"]["targetBpm"], self.track_b["bpm"])

    def test_legacy_smooth_join_penalizes_unsafe_stretch(self) -> None:
        risky_b = {**self.track_b, "bpm": 95}
        plan = build_mashup_plan(self.track_a, risky_b, mode="smooth_join", target_duration_sec=24, use_stems=False)

        self.assertLessEqual(plan["score"], 42)
        self.assertTrue(any("Unsafe tempo mismatch" in warning for warning in plan["warnings"]))

    def test_bed_ducking_when_vocal_active(self) -> None:
        bed = np.ones((2, SAMPLE_RATE * 4), dtype=np.float32) * 0.12
        vocals = np.zeros_like(bed)
        vocals[:, SAMPLE_RATE : SAMPLE_RATE * 2] = 0.08
        ducked = groove._duck_bed_with_vocals(bed, vocals, SAMPLE_RATE, -3.0)

        active = rms(ducked[:, SAMPLE_RATE : SAMPLE_RATE * 2])
        inactive = rms(ducked[:, SAMPLE_RATE * 3 : SAMPLE_RATE * 4])
        self.assertLess(active, inactive * 0.9)

    def test_no_long_vocal_overlap(self) -> None:
        bed = find_candidate_groove_beds(self.track_a, self.track_b)[0]
        phrases_a = extract_vocal_phrases(self.track_a, self.stems / "a" / "demucs_api" / "vocals.wav")
        phrases_b = extract_vocal_phrases(self.track_b, self.stems / "b" / "demucs_api" / "vocals.wav")
        arrangement = build_vocal_handoff_arrangement(bed, phrases_a, phrases_b, "groove_vocal_handoff", 24)
        events = arrangement["vocalEvents"]
        max_overlap = 0.0
        for left_index, left in enumerate(events):
            for right in events[left_index + 1 :]:
                overlap = min(left["timelineEnd"], right["timelineEnd"]) - max(left["timelineStart"], right["timelineStart"])
                max_overlap = max(max_overlap, overlap)

        self.assertLessEqual(max_overlap, 0.5)

    def test_vocal_stretch_policy(self) -> None:
        bed = find_candidate_groove_beds(self.track_a, self.track_b)[0]
        phrase = extract_vocal_phrases({**self.track_b, "bpm": 90}, self.stems / "b" / "demucs_api" / "vocals.wav")[0]
        arrangement = build_vocal_handoff_arrangement(bed, [], [phrase], "b_vocal_on_a_groove", 8, max_vocal_stretch=1.06)

        self.assertFalse(arrangement["vocalEvents"])
        self.assertTrue(any("stretchRatio" in warning for warning in arrangement["globalWarnings"]))

    def test_pitch_policy_no_large_vocal_shift(self) -> None:
        bed = {"camelot": "1A"}
        phrase = {"camelot": "8B"}
        policy = choose_groove_pitch_policy(bed, phrase, allow_vocal_pitch_shift=False)

        self.assertEqual(policy["vocalPitchShiftSemitones"], 0)
        self.assertTrue(policy["warnings"])

    def test_loop_bed_microfade(self) -> None:
        bed = find_candidate_groove_beds(self.track_a, self.track_b)[0]
        audio, warnings = groove._render_bed_loop(bed, {"a": self.track_a, "b": self.track_b}, 36, SAMPLE_RATE)
        discontinuity = float(np.max(np.abs(np.diff(audio, axis=1))))

        self.assertFalse(warnings)
        self.assertTrue(np.all(np.isfinite(audio)))
        self.assertLess(discontinuity, 0.35)

    def test_render_groove_vocal_mashup_no_nan(self) -> None:
        plan = build_groove_mashup_plan(self.track_a, self.track_b, mode="groove_vocal_handoff", target_duration_sec=20, use_stems=True)
        audio, report = render_groove_vocal_mashup(plan["groovePlan"], {"a": self.track_a, "b": self.track_b}, SAMPLE_RATE, target_lufs=-14)

        self.assertEqual(audio.shape[0], 2)
        self.assertTrue(np.all(np.isfinite(audio)))
        self.assertLessEqual(float(np.max(np.abs(audio))), 1.0)
        self.assertGreater(len(report["vocalEvents"]), 2)
        self.assertIn("bed", report)


if __name__ == "__main__":
    unittest.main()
