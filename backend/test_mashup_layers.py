from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.arrangement import score_segment_transition
from backend.mashup import (
    apply_automation,
    apply_microfade,
    apply_simple_delay_tail,
    apply_simple_filter_sweep,
    build_layered_transition,
    choose_target_bpm,
    choose_transition_type_v2,
    compute_pitch_policy,
    compute_stretch_policy,
    make_equal_power_fade_in,
    make_equal_power_fade_out,
    render_layered_mashup,
    sidechain_like_duck,
)
from backend.mixing import SAMPLE_RATE
from backend.test_mashup import base_segment, make_test_track, write_test_stems


def item_segment(**updates: object) -> dict:
    data = base_segment(**updates)
    data.update(
        {
            "segmentId": data.get("id", "seg"),
            "sourceStart": data.get("start", 0.0),
            "sourceEnd": data.get("end", 8.0),
            "timelineStart": updates.get("timelineStart", data.get("start", 0.0)),
            "timelineEnd": updates.get("timelineEnd", data.get("end", 8.0)),
            "layerMode": updates.get("layerMode", "full_mix"),
            "gainDb": 0.0,
        }
    )
    return data


class MashupLayerRouterTests(unittest.TestCase):
    def decision(self, a: dict, b: dict, mode: str = "auto", use_stems: bool = True) -> str:
        compatibility = score_segment_transition(a, b, mode, {"useStems": use_stems})
        return choose_transition_type_v2(a, b, compatibility, mode, use_stems)["type"]

    def test_transition_router_variety(self) -> None:
        chorus = item_segment(label="chorus_like", energy=0.75, vocalEnd=0.1, isCleanExit=True)
        drop = item_segment(source="B", trackId="b", label="drop_like", energy=0.88, vocalStart=0.1, isCleanEntry=True)
        outro = item_segment(label="outro_like", energy=0.35, vocalEnd=0.1)
        intro = item_segment(source="B", trackId="b", label="intro_like", energy=0.28, vocalStart=0.1)
        vocal = item_segment(label="verse_like", vocalDensity=0.85, vocalEnd=0.85)
        bed = item_segment(source="B", trackId="b", label="breakdown_like", vocalDensity=0.1, vocalStart=0.1)
        bass_a = item_segment(bassEnergy=0.9, brightness=0.35)
        bass_b = item_segment(source="B", trackId="b", bassEnergy=0.88, brightness=0.37)
        bright = item_segment(brightness=0.95, bassEnergy=0.12)
        heavy = item_segment(source="B", trackId="b", brightness=0.12, bassEnergy=0.78)

        decisions = {
            self.decision(chorus, drop),
            self.decision(outro, intro),
            self.decision(vocal, bed, "a_vocal_b_instrumental"),
            self.decision(bass_a, bass_b),
            self.decision(bright, heavy),
        }

        self.assertIn("hard_cut", decisions)
        self.assertIn("equal_power_crossfade", decisions)
        self.assertIn("vocal_over_instrumental", decisions)
        self.assertIn("bass_swap", decisions)
        self.assertIn("filter_sweep", decisions)
        self.assertGreaterEqual(len(decisions), 5)

    def test_vocal_over_instrumental_layers(self) -> None:
        vocal = item_segment(trackId="a", source="A", layerMode="vocals", vocalDensity=0.9)
        bed = item_segment(trackId="b", source="B", layerMode="instrumental", vocalDensity=0.1)

        transition = build_layered_transition(vocal, bed, "a_vocal_b_instrumental", "vocal_over_instrumental", True, {"targetBpm": 120})
        stems = {(layer["source"], layer["stem"]) for layer in transition["layerEvents"]}

        self.assertIn(("A", "vocals"), stems)
        self.assertIn(("B", "drums"), stems)
        self.assertIn(("B", "bass"), stems)
        self.assertIn(("B", "other"), stems)
        self.assertNotIn(("B", "vocals"), stems)
        self.assertFalse(any(layer["stem"] == "full" for layer in transition["layerEvents"]))

    def test_bass_swap_automation(self) -> None:
        a = item_segment(trackId="a", source="A", bassEnergy=0.9)
        b = item_segment(trackId="b", source="B", bassEnergy=0.88)

        transition = build_layered_transition(a, b, "auto", "bass_swap", True, {"targetBpm": 120})
        a_bass = next(layer for layer in transition["layerEvents"] if layer["source"] == "A" and layer["stem"] == "bass")
        b_bass = next(layer for layer in transition["layerEvents"] if layer["source"] == "B" and layer["stem"] == "bass")

        self.assertGreater(a_bass["automation"]["gain"][0][1], a_bass["automation"]["gain"][-1][1])
        self.assertLess(b_bass["automation"]["gain"][0][1], b_bass["automation"]["gain"][-1][1])
        self.assertTrue(any("bass conflict fixed" in fix for fix in transition["fixes"]))

    def test_bass_swap_drops_vocals_before_incoming_vocals_open(self) -> None:
        a = item_segment(trackId="a", source="A", bassEnergy=0.9, vocalDensity=0.8, vocalEnd=0.85)
        b = item_segment(trackId="b", source="B", bassEnergy=0.88, vocalDensity=0.8, vocalStart=0.85)

        transition = build_layered_transition(a, b, "auto", "bass_swap", True, {"targetBpm": 120})
        a_vocal = next(layer for layer in transition["layerEvents"] if layer["source"] == "A" and layer["stem"] == "vocals")
        b_vocal = next(layer for layer in transition["layerEvents"] if layer["source"] == "B" and layer["stem"] == "vocals")

        self.assertLessEqual(a_vocal["automation"]["gain"][1][0], 0.34)
        self.assertLessEqual(a_vocal["automation"]["gain"][1][1], -50)
        self.assertGreaterEqual(b_vocal["automation"]["gain"][1][0], 0.66)
        self.assertLessEqual(b_vocal["automation"]["gain"][1][1], -50)
        self.assertTrue(any("vocal overlap prevented" in fix for fix in transition["fixes"]))

    def test_high_vocal_conflict_routes_to_vocal_drop_with_stems(self) -> None:
        a = item_segment(vocalDensity=0.8, vocalEnd=0.9, isCleanExit=False)
        b = item_segment(source="B", trackId="b", vocalDensity=0.85, vocalStart=0.9, isCleanEntry=False)
        compatibility = score_segment_transition(a, b, "smooth_join", {"useStems": True})
        decision = choose_transition_type_v2(a, b, compatibility, "smooth_join", True)

        self.assertEqual(decision["type"], "vocal_drop")
        self.assertIn("active vocals", decision["reason"])

    def test_high_vocal_conflict_without_stems_avoids_crossfade(self) -> None:
        a = item_segment(vocalDensity=0.8, vocalEnd=0.9, isCleanExit=False)
        b = item_segment(source="B", trackId="b", vocalDensity=0.85, vocalStart=0.9, isCleanEntry=False)
        compatibility = score_segment_transition(a, b, "smooth_join", {"useStems": False})
        decision = choose_transition_type_v2(a, b, compatibility, "smooth_join", False)

        self.assertNotEqual(decision["type"], "equal_power_crossfade")
        self.assertTrue(decision["warnings"])

    def test_equal_power_crossfade(self) -> None:
        out = make_equal_power_fade_out(101).reshape(-1)
        inn = make_equal_power_fade_in(101).reshape(-1)

        self.assertAlmostEqual(float(out[50]), 0.707, delta=0.02)
        self.assertAlmostEqual(float(inn[50]), 0.707, delta=0.02)
        self.assertNotAlmostEqual(float(out[50]), 0.5, delta=0.05)


class MashupLayerAudioTests(unittest.TestCase):
    def test_hard_cut_microfade(self) -> None:
        audio = np.ones((2, SAMPLE_RATE // 2), dtype=np.float32)
        faded = apply_microfade(audio, SAMPLE_RATE, fade_ms=10)

        self.assertAlmostEqual(float(faded[0, 0]), 0.0, delta=1e-6)
        self.assertAlmostEqual(float(faded[0, -1]), 0.0, delta=1e-6)
        self.assertLess(float(np.max(np.abs(np.diff(faded[0, :1000])))), 0.02)

    def test_echo_tail_audio_exists(self) -> None:
        t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False, dtype=np.float32)
        audio = np.vstack([0.1 * np.sin(2 * np.pi * 440 * t), 0.1 * np.sin(2 * np.pi * 440 * t)])

        tail = apply_simple_delay_tail(audio, SAMPLE_RATE, bpm=120, feedback=0.35, beats=0.5)
        first = float(np.sqrt(np.mean(tail[:, : SAMPLE_RATE // 2] ** 2)))
        last = float(np.sqrt(np.mean(tail[:, -SAMPLE_RATE // 2 :] ** 2)))

        self.assertGreater(first, 0.001)
        self.assertGreater(last, 0.0)
        self.assertLess(last, first)

    def test_sidechain_duck(self) -> None:
        instrumental = np.ones((2, SAMPLE_RATE), dtype=np.float32) * 0.1
        vocal = np.zeros((2, SAMPLE_RATE), dtype=np.float32)
        vocal[:, SAMPLE_RATE // 4 : SAMPLE_RATE // 2] = 0.2

        ducked = sidechain_like_duck(instrumental, vocal, SAMPLE_RATE, amount_db=-6)
        active = float(np.sqrt(np.mean(ducked[:, SAMPLE_RATE // 4 : SAMPLE_RATE // 2] ** 2)))
        inactive = float(np.sqrt(np.mean(ducked[:, : SAMPLE_RATE // 8] ** 2)))

        self.assertLess(active, inactive)

    def test_filter_sweep_changes_spectrum(self) -> None:
        t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False, dtype=np.float32)
        audio = np.vstack([
            0.05 * np.sin(2 * np.pi * 80 * t) + 0.05 * np.sin(2 * np.pi * 6000 * t),
            0.05 * np.sin(2 * np.pi * 80 * t) + 0.05 * np.sin(2 * np.pi * 6000 * t),
        ])

        swept = apply_simple_filter_sweep(audio, SAMPLE_RATE, lowpass_env=[[0, 12000], [1, 500]])
        freqs = np.fft.rfftfreq(audio.shape[1], 1 / SAMPLE_RATE)
        band = (freqs > 5000) & (freqs < 7000)
        high_before = float(np.mean(np.abs(np.fft.rfft(audio[0]))[band]))
        high_after = float(np.mean(np.abs(np.fft.rfft(swept[0]))[band]))

        self.assertLess(high_after, high_before * 0.75)

    def test_automation_gain_envelope_changes_rms(self) -> None:
        audio = np.ones((2, SAMPLE_RATE), dtype=np.float32) * 0.1
        rendered = apply_automation(audio, SAMPLE_RATE, {"gain": [[0, 0], [1, -20]]}, "bed")

        start = float(np.sqrt(np.mean(rendered[:, : SAMPLE_RATE // 4] ** 2)))
        end = float(np.sqrt(np.mean(rendered[:, -SAMPLE_RATE // 4 :] ** 2)))

        self.assertLess(end, start)

    def test_stretch_policy_vocal_protection(self) -> None:
        vocal = item_segment(bpm=100, layerMode="vocals")
        instrumental = item_segment(source="B", trackId="b", bpm=100, layerMode="instrumental")

        self.assertEqual(choose_target_bpm([vocal, instrumental], "a_vocal_b_instrumental"), 100)
        policy = compute_stretch_policy(vocal, 112, "vocals")
        self.assertTrue(policy["warnings"])
        self.assertTrue(policy["highRisk"])

    def test_pitch_policy_conservative(self) -> None:
        vocal = item_segment(camelot="1A")
        instrumental = item_segment(source="B", trackId="b", camelot="7B")

        policy = compute_pitch_policy(vocal, instrumental, "a_vocal_b_instrumental")

        self.assertEqual(policy["target"], "instrumental")
        self.assertTrue(policy["highRisk"])
        self.assertTrue(policy["warnings"])

    def test_render_layered_mashup_no_nan(self) -> None:
        import backend.mashup as mashup

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uploads = root / "uploads"
            stems = root / "stems"
            uploads.mkdir()
            stems.mkdir()
            track_a = make_test_track(uploads, "a", 220)
            track_b = make_test_track(uploads, "b", 330)
            write_test_stems(stems, "a", 220)
            write_test_stems(stems, "b", 330)
            original_stem_dir = mashup.STEM_DIR
            mashup.STEM_DIR = stems
            try:
                vocal = item_segment(trackId="a", source="A", layerMode="vocals", timelineStart=0.0, timelineEnd=4.0)
                bed = item_segment(trackId="b", source="B", layerMode="instrumental", timelineStart=0.0, timelineEnd=4.0)
                transition = build_layered_transition(vocal, bed, "a_vocal_b_instrumental", "vocal_over_instrumental", True, {"targetBpm": 120})
                plan = {"targetBpm": 120, "layers": transition["layerEvents"], "transitions": [transition]}
                audio, warnings = render_layered_mashup(plan, {"a": track_a, "b": track_b}, use_stems=True)
            finally:
                mashup.STEM_DIR = original_stem_dir

        self.assertEqual(audio.shape[0], 2)
        self.assertFalse(np.isnan(audio).any())
        self.assertFalse(np.isinf(audio).any())
        self.assertLessEqual(float(np.max(np.abs(audio))), 1.0)
        self.assertFalse(any("fell back" in warning for warning in warnings))

    def test_layered_main_vocal_is_not_truncated_to_short_timeline(self) -> None:
        import backend.mashup as mashup

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uploads = root / "uploads"
            stems = root / "stems"
            uploads.mkdir()
            stems.mkdir()
            track_a = make_test_track(uploads, "a", 220)
            write_test_stems(stems, "a", 220)
            original_stem_dir = mashup.STEM_DIR
            mashup.STEM_DIR = stems
            try:
                layer = {
                    "id": "vocal_layer",
                    "trackId": "a",
                    "source": "A",
                    "segmentId": "A_phrase",
                    "stem": "vocals",
                    "sourceStart": 0.0,
                    "sourceEnd": 3.0,
                    "timelineStart": 0.0,
                    "timelineEnd": 1.0,
                    "gainDb": 0.0,
                    "stretchRatio": 1.0,
                    "pitchShiftSemitones": 0,
                    "automation": {"gain": [[0, 0], [1, 0]]},
                    "role": "main_vocal",
                    "warnings": [],
                }
                audio, warnings = render_layered_mashup({"targetBpm": 120, "layers": [layer]}, {"a": track_a}, use_stems=True)
            finally:
                mashup.STEM_DIR = original_stem_dir

        self.assertFalse(warnings)
        self.assertGreater(audio.shape[1] / SAMPLE_RATE, 2.8)


if __name__ == "__main__":
    unittest.main()
