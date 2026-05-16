from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from backend.loudness import loudness_metrics, normalize_loudness
from backend.analysis import _transition_candidates
from backend.mixing import SAMPLE_RATE, _apply_track_mixer, _beat_sync, _crossfade, _dynamic_eq_overlap, _resolve_mix_strategy
from backend.seamless import (
    _automation_curves,
    _energy_handoff_profile,
    _estimate_transient_shift_samples,
    _render_transition_audio,
    _seconds_to_samples,
    _shift_audio,
    _vocal_handoff_timing,
    compute_tempo_adjustment,
    generate_seamless_transition,
)
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
        self.assertIn("sections", candidates)
        self.assertIn("vocal_density_curve", candidates)
        self.assertIn("energy_curve", candidates)


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


class SeamlessTransitionTests(unittest.TestCase):
    def test_compute_tempo_adjustment_allows_small_bpm_change(self) -> None:
        plan = compute_tempo_adjustment(120, 122, {"targetMode": "quality"})

        self.assertTrue(plan["shouldStretch"])
        self.assertAlmostEqual(plan["stretchRatio"], 120 / 122, places=3)
        self.assertEqual(plan["risk"], "low")

    def test_compute_tempo_adjustment_rejects_large_bpm_change(self) -> None:
        plan = compute_tempo_adjustment(128, 95, {"targetMode": "quality"})

        self.assertFalse(plan["shouldStretch"])
        self.assertEqual(plan["risk"], "high")

    def test_stem_render_keeps_rhythm_bed_before_handoff(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import soundfile as sf

        seconds = 4
        samples = SAMPLE_RATE * seconds
        outgoing = np.zeros((2, samples), dtype=np.float32)
        incoming = np.zeros((2, samples), dtype=np.float32)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            outgoing_paths = {}
            incoming_paths = {}
            for stem, value in {"vocals": 0.2, "drums": 0.3, "bass": 0.1, "other": 0.04}.items():
                path = root / f"out_{stem}.wav"
                sf.write(path, np.full((samples, 2), value, dtype=np.float32), SAMPLE_RATE)
                outgoing_paths[stem] = path
            for stem, value in {"vocals": -0.2, "drums": -0.3, "bass": -0.1, "other": -0.04}.items():
                path = root / f"in_{stem}.wav"
                sf.write(path, np.full((samples, 2), value, dtype=np.float32), SAMPLE_RATE)
                incoming_paths[stem] = path

            rendered = _render_transition_audio(
                outgoing,
                incoming,
                {"method": "bass_swap"},
                2,
                0.8,
                {"used": True, "paths": {"outgoing": outgoing_paths, "incoming": incoming_paths}},
            )

        start = _seconds_to_samples(2)
        early = rendered[:, start : start + _seconds_to_samples(0.25)]
        late = rendered[:, start + _seconds_to_samples(1.75) : start + _seconds_to_samples(2)]
        self.assertGreater(float(np.mean(early)), 0.1)
        self.assertLess(float(np.mean(late)), -0.1)
        self.assertLessEqual(float(np.max(np.abs(rendered))), 1.0)

    def test_full_mix_fallback_uses_layered_overlap_not_hard_switch(self) -> None:
        seconds = 4
        t = np.linspace(0, seconds, SAMPLE_RATE * seconds, endpoint=False, dtype=np.float32)
        outgoing = np.vstack([0.3 * signal_like_clicks(t, 2) + 0.14 * np.sin(2 * np.pi * 90 * t)] * 2).astype(np.float32)
        incoming = np.vstack([0.25 * signal_like_clicks(t, 4) - 0.12 * np.sin(2 * np.pi * 140 * t)] * 2).astype(np.float32)

        rendered = _render_transition_audio(
            outgoing,
            incoming,
            {"method": "beatmix"},
            2,
            0.7,
            {"used": False, "paths": None},
        )

        overlap_start = _seconds_to_samples(2)
        early = rendered[:, overlap_start : overlap_start + _seconds_to_samples(0.4)]
        middle = rendered[:, overlap_start + _seconds_to_samples(0.8) : overlap_start + _seconds_to_samples(1.2)]
        self.assertEqual(rendered.shape, (2, SAMPLE_RATE * 6))
        self.assertFalse(np.isnan(rendered).any())
        self.assertGreater(float(np.mean(np.abs(early))), 0.02)
        self.assertGreater(float(np.mean(np.abs(middle))), 0.02)
        self.assertLessEqual(float(np.max(np.abs(rendered))), 1.0)

    def test_transient_shift_estimator_aligns_delayed_incoming_drums(self) -> None:
        samples = SAMPLE_RATE
        reference = np.zeros((2, samples), dtype=np.float32)
        delayed = np.zeros((2, samples), dtype=np.float32)
        reference[:, 10_000] = 1.0
        delayed[:, 10_420] = 1.0

        shift = _estimate_transient_shift_samples(reference, delayed, max_shift_ms=20)
        aligned = _shift_audio(delayed, shift)

        self.assertAlmostEqual(shift, -420, delta=8)
        self.assertEqual(int(np.argmax(aligned[0])), 10_000)

    def test_vocal_handoff_waits_for_incoming_phrase(self) -> None:
        samples = SAMPLE_RATE * 4
        outgoing = np.zeros((2, samples), dtype=np.float32)
        incoming = np.zeros((2, samples), dtype=np.float32)
        outgoing[:, : SAMPLE_RATE] = 0.25
        incoming[:, int(samples * 0.62) :] = 0.25

        timing = _vocal_handoff_timing(outgoing, incoming, vocal_conflict=0.7, method="beatmix")

        self.assertLess(timing["outgoingEndFraction"], 0.35)
        self.assertGreaterEqual(timing["incomingStartFraction"], 0.58)

    def test_energy_handoff_ducks_hot_incoming_intro(self) -> None:
        samples = SAMPLE_RATE * 4
        outgoing = np.full((2, samples), 0.12, dtype=np.float32)
        incoming = np.full((2, samples), 0.42, dtype=np.float32)

        profile = _energy_handoff_profile(outgoing, incoming, "beatmix")
        curves = _automation_curves(samples, "beatmix", 0.1, None, profile)

        self.assertLess(profile["incomingTrimDb"], -3)
        self.assertLess(float(curves["in_drums"][0, SAMPLE_RATE // 2]), float(curves["in_drums"][0, -1]))

    @patch("backend.seamless._rubberband_command", return_value=None)
    @patch("backend.seamless._demucs_available", return_value=False)
    def test_generate_seamless_transition_falls_back_without_external_tools(self, _demucs, _rubberband) -> None:
        seconds = 16
        t = np.linspace(0, seconds, SAMPLE_RATE * seconds, endpoint=False, dtype=np.float32)
        outgoing = np.vstack([0.12 * np.sin(2 * np.pi * 220 * t), 0.12 * np.sin(2 * np.pi * 220 * t)]).astype(np.float32)
        incoming = np.vstack([0.12 * np.sin(2 * np.pi * 330 * t), 0.12 * np.sin(2 * np.pi * 330 * t)]).astype(np.float32)
        from tempfile import TemporaryDirectory
        from pathlib import Path
        import soundfile as sf

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "out.wav"
            in_path = tmp_path / "in.wav"
            sf.write(out_path, outgoing.T, SAMPLE_RATE)
            sf.write(in_path, incoming.T, SAMPLE_RATE)
            analysis_a = {
                "duration": seconds,
                "bpm": 120,
                "key": "8A",
                "camelot": "8A",
                "bars": [0, 2, 4, 6, 8, 10, 12, 14],
                "phrases": [0, 8],
                "transition_candidates": {"outro": 8, "outro_vocal_density": 0.05},
                "vocal_density_curve": [{"time": 8, "density": 0.05}],
            }
            analysis_b = {
                "duration": seconds,
                "bpm": 122,
                "key": "8A",
                "camelot": "8A",
                "bars": [0, 2, 4, 6, 8, 10, 12, 14],
                "phrases": [0, 8],
                "transition_candidates": {"intro": 0, "intro_vocal_density": 0.05},
                "vocal_density_curve": [{"time": 0, "density": 0.05}],
            }
            result = generate_seamless_transition(
                out_path,
                in_path,
                analysis_a,
                analysis_b,
                {
                    "method": "beatmix",
                    "overlapDuration": 4,
                    "outgoingCue": {"time": 8, "role": "outro", "sectionType": "outro"},
                    "incomingCue": {"time": 0, "role": "entry", "sectionType": "intro"},
                },
                {"useStemSeparation": True, "previewDurationBeforeTransition": 4, "previewDurationAfterTransition": 4},
            )

        self.assertTrue(Path(result["audioPath"]).exists())
        self.assertFalse(result["processingReport"]["usedStemSeparation"])
        self.assertGreaterEqual(len(result["warnings"]), 1)
        self.assertEqual(result["processingReport"]["crossfadeCurve"], "equal_power")
        self.assertIn("transientShiftMs", result["processingReport"])
        self.assertIn("incomingVocalDelayMs", result["processingReport"])
        self.assertIn("incomingEnergyTrimDb", result["processingReport"])


if __name__ == "__main__":
    unittest.main()
