from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.arrangement import build_track_segments, segment_compatibility
from backend.mashup import build_mashup_plan, render_mashup_plan
from backend.mixing import SAMPLE_RATE
from backend.storage import write_json


def make_test_track(root: Path, track_id: str, freq: float, noise: bool = False) -> dict:
    seconds = 8
    t = np.linspace(0, seconds, SAMPLE_RATE * seconds, endpoint=False, dtype=np.float32)
    mono = 0.12 * np.sin(2 * np.pi * freq * t)
    mono += 0.035 * np.sin(2 * np.pi * 90 * t)
    if noise:
        rng = np.random.default_rng(123)
        mono += rng.normal(0, 0.015, size=mono.shape).astype(np.float32)
    stereo = np.vstack([mono, mono]).astype(np.float32)
    path = root / f"{track_id}.wav"
    sf.write(path, stereo.T, SAMPLE_RATE, subtype="PCM_16")
    bars = [float(i) for i in range(0, seconds + 1)]
    track = {
        "id": track_id,
        "name": f"{track_id}.wav",
        "path": str(path),
        "content_type": "audio/wav",
        "duration": float(seconds),
        "bpm": 120 if track_id == "a" else 124,
        "key": "A Min",
        "camelot": "8A" if track_id == "a" else "9A",
        "bars": bars,
        "phrases": [0.0, 4.0],
        "sections": [
            {"type": "intro", "startTime": 0.0, "endTime": 2.0},
            {"type": "verse", "startTime": 2.0, "endTime": 4.0},
            {"type": "chorus", "startTime": 4.0, "endTime": 6.0},
            {"type": "outro", "startTime": 6.0, "endTime": 8.0},
        ],
        "transition_candidates": {
            "intro": 0.0,
            "outro": 6.0,
            "intro_vocal_density": 0.1,
            "outro_vocal_density": 0.1,
            "intro_energy": 0.3,
            "outro_energy": 0.3,
        },
    }
    write_json(root / f"{track_id}.json", track)
    return track


class MashupArrangementTests(unittest.TestCase):
    def test_segment_cutting_uses_bar_aligned_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            track = make_test_track(Path(tmp), "a", 220)
            segments = build_track_segments(track, "A", bars_per_segment=4)

        self.assertGreaterEqual(len(segments), 2)
        self.assertEqual(segments[0]["id"], "A_seg_001")
        self.assertEqual(segments[0]["barStart"], 0)
        self.assertEqual(segments[0]["barEnd"], 4)
        self.assertIn(segments[0]["label"], {"intro_like", "verse_like", "chorus_like", "outro_like"})

    def test_compatibility_score_is_bounded_and_componentized(self) -> None:
        a = {
            "bpm": 120,
            "camelot": "8A",
            "energy": 0.45,
            "vocalDensity": 0.2,
            "bassEnergy": 0.4,
            "drumActivity": 0.5,
            "brightness": 0.35,
        }
        b = {
            "bpm": 122,
            "camelot": "9A",
            "energy": 0.5,
            "vocalDensity": 0.25,
            "bassEnergy": 0.42,
            "drumActivity": 0.55,
            "brightness": 0.4,
        }

        result = segment_compatibility(a, b)

        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 1)
        self.assertEqual(
            set(result["components"]),
            {"bpmCompatibility", "camelotCompatibility", "energyFlow", "vocalConflictAvoidance", "timbreSimilarity"},
        )

    def test_plan_generation_includes_both_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            track_a = make_test_track(root, "a", 220)
            track_b = make_test_track(root, "b", 330, noise=True)
            result = build_mashup_plan(track_a, track_b, mode="hook_swap", target_duration_sec=12, bars_per_segment=4, use_stems=False)

        self.assertGreater(result["score"], 0)
        self.assertGreaterEqual(len(result["plan"]), 2)
        self.assertEqual({item["source"] for item in result["plan"]}, {"A", "B"})
        self.assertTrue(all(item["timelineEnd"] > item["timelineStart"] for item in result["plan"]))


class MashupRenderTests(unittest.TestCase):
    def test_render_short_sine_noise_plan_without_nan_or_inf(self) -> None:
        import backend.mashup as mashup

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uploads = root / "uploads"
            exports = root / "exports"
            stems = root / "stems"
            uploads.mkdir()
            exports.mkdir()
            stems.mkdir()
            track_a = make_test_track(uploads, "a", 220)
            track_b = make_test_track(uploads, "b", 330, noise=True)
            plan = build_mashup_plan(track_a, track_b, mode="smooth_join", target_duration_sec=10, bars_per_segment=4, use_stems=False)["plan"]

            original_export_dir = mashup.EXPORT_DIR
            original_stem_dir = mashup.STEM_DIR
            mashup.EXPORT_DIR = exports
            mashup.STEM_DIR = stems
            try:
                result = render_mashup_plan(plan, {"a": track_a, "b": track_b}, fmt="wav", target_lufs=-16, use_stems=False)
                output_path = exports / Path(result["downloadUrl"]).name
                output_exists = output_path.exists()
                output_size = output_path.stat().st_size if output_exists else 0
                audio, _sr = sf.read(output_path, dtype="float32", always_2d=True)
            finally:
                mashup.EXPORT_DIR = original_export_dir
                mashup.STEM_DIR = original_stem_dir

        self.assertTrue(output_exists)
        self.assertGreater(output_size, 1000)
        self.assertFalse(np.isnan(audio).any())
        self.assertFalse(np.isinf(audio).any())
        self.assertLessEqual(float(np.max(np.abs(audio))), 1.0)


if __name__ == "__main__":
    unittest.main()

