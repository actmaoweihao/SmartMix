from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from backend.loudness import loudness_metrics, normalize_loudness
from backend.analysis import _transition_candidates
from backend.mixing import SAMPLE_RATE, _apply_track_mixer, _beat_sync, _crossfade, _dynamic_eq_overlap, _resolve_mix_strategy
from backend.transition import plan_transition


def signal_like_clicks(t: np.ndarray, hz: float) -> np.ndarray:
    phase = np.mod(t * hz, 1.0)
    return np.exp(-phase * 42).astype(np.float32)


class BeatSyncTests(unittest.TestCase):
    def test_beat_sync_speeds_up_tracks_below_target_bpm(self) -> None:
        buffer_a = np.zeros((2, SAMPLE_RATE), dtype=np.float32)
        buffer_b = np.zeros((2, SAMPLE_RATE), dtype=np.float32)
        tracks = [{"bpm": 100}, {"bpm": 120}]

        with patch("backend.mixing.librosa.effects.time_stretch", side_effect=lambda y, rate: y) as stretch:
            _beat_sync([buffer_a, buffer_b], tracks)

        rates = [call.kwargs["rate"] for call in stretch.call_args_list]
        self.assertGreater(rates[0], 1.0)
        self.assertLess(rates[1], 1.0)
        self.assertAlmostEqual(rates[0], 1.1, places=3)
        self.assertAlmostEqual(rates[1], 110 / 120, places=3)


class LoudnessTests(unittest.TestCase):
    def test_normalize_loudness_moves_audio_toward_target(self) -> None:
        seconds = 3
        t = np.linspace(0, seconds, SAMPLE_RATE * seconds, endpoint=False)
        mono = 0.02 * np.sin(2 * np.pi * 440 * t)
        stereo = np.vstack([mono, mono]).astype(np.float32)

        normalized = normalize_loudness(stereo, SAMPLE_RATE, -18.0)
        metrics = loudness_metrics(normalized, SAMPLE_RATE)

        self.assertLess(abs(metrics["lufs"] - -18.0), 1.0)
        self.assertLessEqual(np.max(np.abs(normalized)), 0.98)


class TransitionPlanTests(unittest.TestCase):
    def test_plan_transition_reports_phrase_fit_and_candidates(self) -> None:
        prev_track = {
            "duration": 180,
            "bpm": 120,
            "outro_low": 16,
            "transition_candidates": {"outro": 148, "confidence": 0.8},
        }
        next_track = {
            "duration": 180,
            "bpm": 120,
            "intro_low": 16,
            "transition_candidates": {"intro": 32, "confidence": 0.8},
        }

        plan = plan_transition(
            prev_track,
            next_track,
            {"crossfade": 32, "aiPrecision": True, "phraseBars": 16, "autoTransition": False},
        )

        self.assertEqual(plan.seconds, 32)
        self.assertEqual(plan.phrase_bars, 16)
        self.assertEqual(plan.prev_outro, 148)
        self.assertEqual(plan.next_intro, 32)
        self.assertEqual(plan.prev_overlap_start, 148)
        self.assertEqual(plan.next_overlap_start, 0)
        self.assertGreaterEqual(plan.confidence, 0.7)

    def test_plan_transition_places_incoming_overlap_before_intro_anchor(self) -> None:
        plan = plan_transition(
            {"duration": 10, "bpm": 120, "transition_candidates": {"outro": 6, "confidence": 0.8}},
            {"duration": 10, "bpm": 120, "transition_candidates": {"intro": 4, "confidence": 0.8}},
            {"crossfade": 2, "aiPrecision": True, "phraseBars": 1, "autoTransition": False},
        )

        self.assertEqual(plan.prev_overlap_start, 6)
        self.assertEqual(plan.next_overlap_start, 2)


class AnalysisCandidateTests(unittest.TestCase):
    def test_transition_candidates_include_density_metadata(self) -> None:
        y = np.zeros(SAMPLE_RATE * 12, dtype=np.float32)
        bars = [0, 2, 4, 6, 8, 10]
        candidates = _transition_candidates(y, SAMPLE_RATE, 12, bars, {"intro_low": 2, "outro_low": 2})

        self.assertEqual(candidates["method"], "bar-vocal-energy")
        self.assertIn("intro_vocal_density", candidates)
        self.assertIn("outro_vocal_density", candidates)


class CrossfadePlanTests(unittest.TestCase):
    def test_crossfade_uses_planned_prev_and_next_anchors(self) -> None:
        prev = np.ones((2, SAMPLE_RATE * 10), dtype=np.float32)
        incoming = np.full((2, SAMPLE_RATE * 10), 2, dtype=np.float32)
        rendered = _crossfade(
            [prev, incoming],
            [
                {"duration": 10, "bpm": 120, "transition_candidates": {"outro": 6, "confidence": 0.8}},
                {"duration": 10, "bpm": 120, "transition_candidates": {"intro": 4, "confidence": 0.8}},
            ],
            {
                "crossfade": 2,
                "autoTransition": False,
                "aiPrecision": True,
                "phraseBars": 1,
                "filterMode": "none",
                "equalPowerFade": False,
            },
        )

        self.assertEqual(rendered.shape[1], SAMPLE_RATE * 14)

    def test_track_mixer_gain_affects_export_buffer(self) -> None:
        buffer = np.full((2, 128), 0.5, dtype=np.float32)
        mixed = _apply_track_mixer(buffer, {"mixer": {"gain": 0.5, "eq": {"low": 0, "mid": 0, "high": 0}}})

        self.assertTrue(np.allclose(mixed, 0.25))

    def test_auto_mix_strategy_prefers_vocal_safe_for_dense_vocals(self) -> None:
        strategy = _resolve_mix_strategy(
            {"mixStrategy": "auto"},
            {"bpm": 120, "energy": 0.5, "transition_candidates": {"outro_vocal_density": 0.8}},
            {"bpm": 121, "energy": 0.55, "transition_candidates": {"intro_vocal_density": 0.2}},
        )

        self.assertEqual(strategy, "vocalSafe")

    def test_auto_mix_strategy_prefers_vocal_handoff_for_incoming_vocal_over_drums(self) -> None:
        strategy = _resolve_mix_strategy(
            {"mixStrategy": "auto"},
            {
                "bpm": 129,
                "energy": 0.88,
                "transition_candidates": {"outro_vocal_density": 0.34, "outro_energy": 0.36},
            },
            {
                "bpm": 112,
                "energy": 0.84,
                "transition_candidates": {"intro_vocal_density": 0.35, "intro_energy": 0.38},
            },
        )

        self.assertEqual(strategy, "vocalHandoff")

    def test_vocal_handoff_overlap_returns_stable_audio(self) -> None:
        t = np.linspace(0, 4, SAMPLE_RATE * 4, endpoint=False, dtype=np.float32)
        prev = 0.3 * np.sin(2 * np.pi * 90 * t) + 0.12 * signal_like_clicks(t, 2)
        next_track = 0.22 * np.sin(2 * np.pi * 440 * t) + 0.08 * signal_like_clicks(t, 4)
        overlap = _dynamic_eq_overlap(
            np.vstack([prev, prev]).astype(np.float32),
            np.vstack([next_track, next_track]).astype(np.float32),
            "vocalHandoff",
        )

        self.assertEqual(overlap.shape, (2, SAMPLE_RATE * 4))
        self.assertFalse(np.isnan(overlap).any())
        self.assertLessEqual(np.max(np.abs(overlap)), 1.0)


if __name__ == "__main__":
    unittest.main()
