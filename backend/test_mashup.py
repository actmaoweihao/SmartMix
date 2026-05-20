from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.arrangement import build_music_segments, build_track_segments, score_segment_transition, segment_compatibility
from backend.mashup import build_mashup_plan, render_mashup_plan
from backend.mixing import SAMPLE_RATE
from backend.storage import write_json


def make_test_track(root: Path, track_id: str, freq: float, noise: bool = False, seconds: int = 8) -> dict:
    t = np.linspace(0, seconds, SAMPLE_RATE * seconds, endpoint=False, dtype=np.float32)
    mono = 0.12 * np.sin(2 * np.pi * freq * t)
    mono += 0.035 * np.sin(2 * np.pi * 90 * t)
    click_times = np.arange(0, seconds, 1.0)
    for click in click_times:
        sample = int(click * SAMPLE_RATE)
        mono[sample : sample + 64] += np.hanning(64).astype(np.float32) * 0.08
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
        "phrases": [0.0, 4.0, float(seconds)],
        "energy_profile": [
            {"time": 0.0, "energy": 0.18},
            {"time": 2.0, "energy": 0.35},
            {"time": 4.0, "energy": 0.72},
            {"time": 6.0, "energy": 0.48},
        ],
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


def base_segment(**updates: object) -> dict:
    segment = {
        "id": "seg",
        "trackId": "a",
        "source": "A",
        "start": 0.0,
        "end": 8.0,
        "duration": 8.0,
        "barStart": 0,
        "barEnd": 8,
        "phraseStart": 0,
        "phraseEnd": 1,
        "label": "verse_like",
        "energy": 0.45,
        "energyStart": 0.42,
        "energyEnd": 0.46,
        "energyDelta": 0.04,
        "vocalDensity": 0.25,
        "vocalStart": 0.15,
        "vocalEnd": 0.18,
        "bassEnergy": 0.35,
        "drumActivity": 0.45,
        "brightness": 0.42,
        "spectralChange": 0.2,
        "bpm": 120,
        "camelot": "8A",
        "key": "A minor",
        "mixInScore": 0.8,
        "mixOutScore": 0.82,
        "isCleanEntry": True,
        "isCleanExit": True,
        "riskFlags": [],
    }
    segment.update(updates)
    return segment


def write_test_stems(root: Path, track_id: str, freq: float, seconds: int = 8) -> None:
    stem_root = root / track_id / "demucs_api"
    stem_root.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, SAMPLE_RATE * seconds, endpoint=False, dtype=np.float32)
    stems = {
        "vocals": 0.08 * np.sin(2 * np.pi * (freq * 2.0) * t),
        "drums": 0.04 * np.sign(np.sin(2 * np.pi * 4.0 * t)),
        "bass": 0.07 * np.sin(2 * np.pi * 70.0 * t),
        "other": 0.05 * np.sin(2 * np.pi * freq * t),
    }
    for name, mono in stems.items():
        stereo = np.vstack([mono, mono]).astype(np.float32)
        sf.write(stem_root / f"{name}.wav", stereo.T, SAMPLE_RATE, subtype="PCM_16")


class MashupArrangementTests(unittest.TestCase):
    def test_build_music_segments_never_returns_negative_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            track = make_test_track(Path(tmp), "a", 220)
            segments = build_music_segments(track, "A", bars_per_segment=4)

        self.assertGreaterEqual(len(segments), 2)
        self.assertTrue(all(segment["end"] > segment["start"] for segment in segments))
        self.assertTrue(all(segment["duration"] > 0 for segment in segments))

    def test_segment_boundaries_snap_to_bars_or_phrases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            track = make_test_track(Path(tmp), "a", 220)
            segments = build_track_segments(track, "A", bars_per_segment=4)

        boundaries = set(track["bars"]) | set(track["phrases"])
        for segment in segments:
            self.assertTrue(any(abs(segment["start"] - boundary) <= 0.03 for boundary in boundaries))
            self.assertTrue(any(abs(segment["end"] - boundary) <= 0.03 for boundary in boundaries))

    def test_compatibility_score_is_bounded_and_componentized(self) -> None:
        result = segment_compatibility(base_segment(), base_segment(source="B", trackId="b", camelot="9A", bpm=122))

        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 1)
        self.assertEqual(
            set(result["components"]),
            {"bpmCompatibility", "camelotCompatibility", "energyFlow", "vocalConflictAvoidance", "timbreSimilarity"},
        )

    def test_score_segment_transition_returns_v2_components_and_transition(self) -> None:
        result = score_segment_transition(base_segment(), base_segment(source="B", trackId="b", camelot="9A", bpm=122))

        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 100)
        self.assertIn("recommendedTransition", result)
        self.assertEqual(
            set(result["components"]),
            {"bpm", "camelot", "phrase", "energyFlow", "vocalConflict", "bassConflict", "timbre", "entryExit"},
        )

    def test_high_vocal_conflict_lowers_score_and_adds_fix(self) -> None:
        clean = score_segment_transition(base_segment(), base_segment(source="B", trackId="b"))
        conflict = score_segment_transition(
            base_segment(vocalDensity=0.9, vocalEnd=0.95),
            base_segment(source="B", trackId="b", vocalDensity=0.9, vocalStart=0.95),
            context={"useStems": True},
        )

        self.assertLess(conflict["score"], clean["score"])
        self.assertTrue(conflict["warnings"])
        self.assertTrue(any("vocal" in fix.lower() for fix in conflict["fixes"]))

    def test_high_bass_overlap_recommends_bass_swap(self) -> None:
        result = score_segment_transition(
            base_segment(bassEnergy=0.92),
            base_segment(source="B", trackId="b", bassEnergy=0.9),
            context={"useStems": True},
        )

        self.assertEqual(result["recommendedTransition"], "bass_swap")
        self.assertTrue(any("bass" in fix.lower() for fix in result["fixes"]))

    def test_clean_entry_exit_scores_higher_than_dirty_boundaries(self) -> None:
        clean = score_segment_transition(base_segment(), base_segment(source="B", trackId="b"))
        dirty = score_segment_transition(
            base_segment(isCleanExit=False, mixOutScore=0.2),
            base_segment(source="B", trackId="b", isCleanEntry=False, mixInScore=0.2),
        )

        self.assertGreater(clean["components"]["entryExit"], dirty["components"]["entryExit"])
        self.assertGreater(clean["score"], dirty["score"])

    def test_plan_generation_returns_quality_report_and_alternatives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            track_a = make_test_track(root, "a", 220)
            track_b = make_test_track(root, "b", 330, noise=True)
            result = build_mashup_plan(
                track_a,
                track_b,
                mode="auto",
                target_duration_sec=12,
                bars_per_segment=4,
                use_stems=False,
                return_alternatives=True,
            )

        self.assertGreater(result["score"], 0)
        self.assertGreaterEqual(len(result["plan"]), 2)
        self.assertIn("qualityReport", result)
        self.assertIn("summary", result["qualityReport"])
        self.assertIn("alternativePlans", result)

    def test_plan_generation_includes_both_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            track_a = make_test_track(root, "a", 220)
            track_b = make_test_track(root, "b", 330, noise=True)
            result = build_mashup_plan(track_a, track_b, mode="hook_swap", target_duration_sec=12, bars_per_segment=4, use_stems=False)

        self.assertGreater(result["score"], 0)
        self.assertEqual({item["source"] for item in result["plan"]}, {"A", "B"})
        self.assertTrue(all(item["timelineEnd"] > item["timelineStart"] for item in result["plan"]))


class MashupRenderTests(unittest.TestCase):
    def render_in_temp(self, plan: list[dict], tracks: dict[str, dict], use_stems: bool, stems_writer: callable | None = None) -> tuple[dict, np.ndarray]:
        import backend.mashup as mashup

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exports = root / "exports"
            stems = root / "stems"
            exports.mkdir()
            stems.mkdir()
            if stems_writer:
                stems_writer(stems)
            original_export_dir = mashup.EXPORT_DIR
            original_stem_dir = mashup.STEM_DIR
            mashup.EXPORT_DIR = exports
            mashup.STEM_DIR = stems
            try:
                result = render_mashup_plan(plan, tracks, fmt="wav", target_lufs=-16, use_stems=use_stems)
                output_path = exports / Path(result["downloadUrl"]).name
                audio, _sr = sf.read(output_path, dtype="float32", always_2d=True)
            finally:
                mashup.EXPORT_DIR = original_export_dir
                mashup.STEM_DIR = original_stem_dir
        return result, audio

    def test_render_short_sine_noise_plan_without_nan_or_inf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            uploads = Path(tmp) / "uploads"
            uploads.mkdir()
            track_a = make_test_track(uploads, "a", 220)
            track_b = make_test_track(uploads, "b", 330, noise=True)
            plan = build_mashup_plan(track_a, track_b, mode="smooth_join", target_duration_sec=10, bars_per_segment=4, use_stems=False)["plan"]
            result, audio = self.render_in_temp(plan, {"a": track_a, "b": track_b}, use_stems=False)

        self.assertTrue(result["ok"])
        self.assertFalse(np.isnan(audio).any())
        self.assertFalse(np.isinf(audio).any())

    def test_crossfade_output_peak_is_reasonable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            uploads = Path(tmp) / "uploads"
            uploads.mkdir()
            track_a = make_test_track(uploads, "a", 220)
            track_b = make_test_track(uploads, "b", 330, noise=True)
            plan = [
                {
                    "id": "one",
                    "source": "A",
                    "trackId": "a",
                    "segmentId": "A_seg_001",
                    "sourceStart": 0.0,
                    "sourceEnd": 5.0,
                    "timelineStart": 0.0,
                    "timelineEnd": 5.0,
                    "layerMode": "full_mix",
                    "gainDb": 0.0,
                    "transitionIn": {"type": "none", "durationSec": 0.0},
                    "transitionOut": {"type": "crossfade", "durationSec": 1.0},
                },
                {
                    "id": "two",
                    "source": "B",
                    "trackId": "b",
                    "segmentId": "B_seg_001",
                    "sourceStart": 0.0,
                    "sourceEnd": 5.0,
                    "timelineStart": 4.0,
                    "timelineEnd": 9.0,
                    "layerMode": "full_mix",
                    "gainDb": 0.0,
                    "transitionIn": {"type": "crossfade", "durationSec": 1.0},
                    "transitionOut": {"type": "none", "durationSec": 0.0},
                },
            ]
            result, audio = self.render_in_temp(plan, {"a": track_a, "b": track_b}, use_stems=False)

        self.assertTrue(result["ok"])
        self.assertLessEqual(float(np.max(np.abs(audio))), 1.0)
        self.assertLess(result["report"]["peak"], 1.0)

    def test_vocal_over_instrumental_without_stems_warns_and_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            uploads = Path(tmp) / "uploads"
            uploads.mkdir()
            track_a = make_test_track(uploads, "a", 220)
            track_b = make_test_track(uploads, "b", 330)
            plan = build_mashup_plan(track_a, track_b, mode="a_vocal_b_instrumental", target_duration_sec=8, bars_per_segment=4, use_stems=False)["plan"]
            result, audio = self.render_in_temp(plan, {"a": track_a, "b": track_b}, use_stems=False)

        self.assertTrue(result["ok"])
        self.assertTrue(any("fell back" in warning for warning in result["report"]["warnings"]))
        self.assertFalse(np.isnan(audio).any())

    def test_stem_aware_vocals_and_instrumental_mix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            uploads = Path(tmp) / "uploads"
            uploads.mkdir()
            track_a = make_test_track(uploads, "a", 220)
            track_b = make_test_track(uploads, "b", 330)
            plan = build_mashup_plan(track_a, track_b, mode="a_vocal_b_instrumental", target_duration_sec=8, bars_per_segment=4, use_stems=True)["plan"]

            def stems_writer(stems_root: Path) -> None:
                write_test_stems(stems_root, "a", 220)
                write_test_stems(stems_root, "b", 330)

            result, audio = self.render_in_temp(plan, {"a": track_a, "b": track_b}, use_stems=True, stems_writer=stems_writer)

        self.assertTrue(result["ok"])
        self.assertFalse(any("fell back" in warning for warning in result["report"]["warnings"]))
        self.assertGreater(float(np.max(np.abs(audio))), 0.01)

    def test_final_limiter_keeps_peak_at_or_below_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            uploads = Path(tmp) / "uploads"
            uploads.mkdir()
            track_a = make_test_track(uploads, "a", 220)
            track_b = make_test_track(uploads, "b", 330)
            plan = build_mashup_plan(track_a, track_b, mode="smooth_join", target_duration_sec=10, bars_per_segment=4, use_stems=False)["plan"]
            for item in plan:
                item["gainDb"] = 12.0
            result, audio = self.render_in_temp(plan, {"a": track_a, "b": track_b}, use_stems=False)

        self.assertTrue(result["ok"])
        self.assertLessEqual(float(np.max(np.abs(audio))), 1.0)
        self.assertLessEqual(result["report"]["peak"], 1.0)


if __name__ == "__main__":
    unittest.main()
