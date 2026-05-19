from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from backend.loudness import loudness_metrics, normalize_loudness
from backend.analysis import _transition_candidates
from backend.mixing import SAMPLE_RATE, _apply_track_mixer, _beat_sync, _crossfade, _dynamic_eq_overlap, _render_stem_mixer, _resolve_mix_strategy
from backend.seamless import (
    _adapt_render_method,
    _refine_alignment_for_smoothness,
    _automation_curves,
    _energy_handoff_profile,
    _estimate_transient_shift_samples,
    _frequency_handoff_profile,
    _render_transition_audio,
    _rhythm_bridge_layer,
    _seconds_to_samples,
    _shift_audio,
    _stabilize_transition_loudness,
    _transition_glue_layer,
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


class StemDebuggerApiTests(unittest.TestCase):
    def test_stem_debugger_api_returns_cached_demucs_stems(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from fastapi.testclient import TestClient
        import backend.main as api

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            uploads = root / "uploads"
            stems = root / "stems"
            uploads.mkdir()
            stem_dir = stems / "track-1" / "demucs_api"
            stem_dir.mkdir(parents=True)
            source = uploads / "track-1.wav"
            source.write_bytes(b"audio")
            for stem in ("vocals", "drums", "bass", "other"):
                (stem_dir / f"{stem}.wav").write_bytes(f"{stem}-audio".encode("utf-8"))

            original_upload_dir = api.UPLOAD_DIR
            original_stem_dir = api.STEM_DIR
            api.UPLOAD_DIR = uploads
            api.STEM_DIR = stems
            try:
                api.write_json(
                    uploads / "track-1.json",
                    {"id": "track-1", "name": "Track 1.wav", "path": str(source), "content_type": "audio/wav"},
                )
                client = TestClient(api.app)
                response = client.post("/api/tracks/track-1/stems", json={"device": "auto"})
                payload = response.json()

                self.assertEqual(response.status_code, 200)
                self.assertTrue(payload["cached"])
                self.assertEqual(set(payload["stems"]), {"vocals", "drums", "bass", "other"})
                self.assertEqual(payload["stems"]["vocals"]["url"], "/api/tracks/track-1/stems/vocals/audio")

                audio = client.get(payload["stems"]["vocals"]["url"])
                self.assertEqual(audio.status_code, 200)
                self.assertIn(b"vocals-audio", audio.content)
            finally:
                api.UPLOAD_DIR = original_upload_dir
                api.STEM_DIR = original_stem_dir


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

    def test_diffmst_stem_mixer_renders_cached_stems(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import soundfile as sf
        import backend.mixing as mixing

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            stem_dir = root / "track-1" / "demucs_api"
            stem_dir.mkdir(parents=True)
            tone = np.ones((2, 256), dtype=np.float32) * 0.1
            for stem in ("vocals", "drums", "bass", "other"):
                sf.write(stem_dir / f"{stem}.wav", tone.T, SAMPLE_RATE)

            original_stem_dir = mixing.STEM_DIR
            mixing.STEM_DIR = root
            try:
                rendered = _render_stem_mixer(
                    {
                        "id": "track-1",
                        "duration": 256 / SAMPLE_RATE,
                        "stemMixer": {
                            "enabled": True,
                            "stems": {
                                "vocals": {"gain": 1.0, "pan": 0.5, "eqDb": {}, "compressor": {}},
                                "drums": {"gain": 0.0, "pan": 0.5, "eqDb": {}, "compressor": {}},
                                "bass": {"gain": 0.0, "pan": 0.5, "eqDb": {}, "compressor": {}},
                                "other": {"gain": 0.0, "pan": 0.5, "eqDb": {}, "compressor": {}},
                            },
                            "master": {"gainDb": 0.0, "eqDb": {}, "compressor": {}},
                        },
                    }
                )
            finally:
                mixing.STEM_DIR = original_stem_dir

        self.assertIsNotNone(rendered)
        assert rendered is not None
        self.assertEqual(rendered.shape, (2, 256))
        self.assertGreater(float(np.mean(np.abs(rendered))), 0.05)

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

    def test_crossfade_embeds_applied_transition_preview_audio(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import soundfile as sf
        import backend.mixing as mixing

        seconds = 12
        prev = np.full((2, SAMPLE_RATE * seconds), 0.1, dtype=np.float32)
        incoming = np.full((2, SAMPLE_RATE * seconds), -0.2, dtype=np.float32)
        preview = np.full((2, SAMPLE_RATE * 4), -0.42, dtype=np.float32)

        with TemporaryDirectory() as tmp:
            original_export_dir = mixing.EXPORT_DIR
            mixing.EXPORT_DIR = Path(tmp)
            try:
                preview_path = mixing.EXPORT_DIR / "preview.wav"
                sf.write(preview_path, preview.T, SAMPLE_RATE, subtype="PCM_16")
                rendered = _crossfade(
                    [prev, incoming],
                    [
                        {"id": "a", "duration": seconds, "transition_candidates": {"outro": 6}},
                        {
                            "id": "b",
                            "duration": seconds,
                            "transition_candidates": {"intro": 2},
                            "appliedTransitionPreview": {
                                "audioPath": str(preview_path),
                                "previewStartTime": 4,
                                "outgoingCue": {"time": 6},
                                "incomingCue": {"time": 2},
                                "outgoingTrackId": "a",
                                "incomingTrackId": "b",
                            },
                        },
                    ],
                    {"crossfade": 2, "autoTransition": False, "aiPrecision": True, "filterMode": "dynamicEq"},
                )
            finally:
                mixing.EXPORT_DIR = original_export_dir

        splice_start = SAMPLE_RATE * 4
        embedded = rendered[:, splice_start : splice_start + SAMPLE_RATE]
        self.assertLess(float(np.mean(embedded)), -0.08)
        self.assertEqual(rendered.shape[1], SAMPLE_RATE * 16)


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

            stem_report = {"used": True, "paths": {"outgoing": outgoing_paths, "incoming": incoming_paths}}
            rendered = _render_transition_audio(
                outgoing,
                incoming,
                {"method": "bass_swap"},
                2,
                0.8,
                stem_report,
            )

        start = rendered.shape[1] - samples
        active_samples = _seconds_to_samples(stem_report["renderOverlapDuration"])
        early = rendered[:, start : start + _seconds_to_samples(0.25)]
        late_start = start + max(0, active_samples - _seconds_to_samples(0.25))
        late = rendered[:, late_start : late_start + _seconds_to_samples(0.25)]
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

    def test_vocal_handoff_cuts_outgoing_before_next_lyric_reentry(self) -> None:
        samples = SAMPLE_RATE * 4
        outgoing = np.zeros((2, samples), dtype=np.float32)
        incoming = np.zeros((2, samples), dtype=np.float32)
        outgoing[:, : int(samples * 0.22)] = 0.24
        outgoing[:, int(samples * 0.32) : int(samples * 0.54)] = 0.24
        incoming[:, int(samples * 0.38) :] = 0.3

        timing = _vocal_handoff_timing(outgoing, incoming, vocal_conflict=0.75, method="beatmix")
        curves = _automation_curves(samples, "beatmix", 0.75, timing)

        self.assertTrue(timing["outgoingGuarded"])
        self.assertLess(timing["outgoingEndFraction"], 0.32)
        self.assertLess(float(curves["out_vocals"][0, int(samples * 0.34)]), 0.08)

    def test_vocal_guard_detects_later_outgoing_phrase_reentry(self) -> None:
        samples = SAMPLE_RATE * 4
        outgoing = np.zeros((2, samples), dtype=np.float32)
        incoming = np.zeros((2, samples), dtype=np.float32)
        outgoing[:, int(samples * 0.24) : int(samples * 0.36)] = 0.24
        outgoing[:, int(samples * 0.46) : int(samples * 0.62)] = 0.24
        incoming[:, int(samples * 0.54) :] = 0.3

        timing = _vocal_handoff_timing(outgoing, incoming, vocal_conflict=0.75, method="beatmix")

        self.assertTrue(timing["outgoingGuarded"])
        self.assertLess(timing["outgoingEndFraction"], 0.46)

    def test_adaptive_render_method_avoids_long_blend_for_high_risk_vocals(self) -> None:
        quick = _adapt_render_method(
            "beatmix",
            0.76,
            {"outgoingGuarded": True, "incomingStartFraction": 0.72},
            {"incomingTrimDb": -2},
            {"lowTrimDb": -1, "midTrimDb": 0, "highTrimDb": 0},
        )
        echo = _adapt_render_method(
            "bass_swap",
            0.6,
            {"outgoingGuarded": True, "incomingStartFraction": 0.5},
            {"incomingTrimDb": -5},
            {"lowTrimDb": -5, "midTrimDb": 0, "highTrimDb": 0},
        )

        self.assertEqual(quick, "echo_out")
        self.assertEqual(echo, "echo_out")

    def test_adapted_render_method_shortens_active_overlap(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import soundfile as sf

        seconds = 10
        samples = SAMPLE_RATE * seconds
        outgoing = np.zeros((2, samples), dtype=np.float32)
        incoming = np.zeros((2, samples), dtype=np.float32)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            outgoing_paths = {}
            incoming_paths = {}
            for stem in ("drums", "bass", "other", "vocals"):
                out = np.zeros((samples, 2), dtype=np.float32)
                inc = np.zeros((samples, 2), dtype=np.float32)
                if stem == "vocals":
                    out[SAMPLE_RATE * 2 : int(SAMPLE_RATE * 3.5)] = 0.25
                    out[int(SAMPLE_RATE * 5.6) : int(SAMPLE_RATE * 7.0)] = 0.25
                    inc[int(SAMPLE_RATE * 5.5) :] = 0.3
                else:
                    out[:] = 0.08
                    inc[:] = 0.22
                out_path = root / f"out_{stem}.wav"
                in_path = root / f"in_{stem}.wav"
                sf.write(out_path, out, SAMPLE_RATE)
                sf.write(in_path, inc, SAMPLE_RATE)
                outgoing_paths[stem] = out_path
                incoming_paths[stem] = in_path

            stem_report = {"used": True, "paths": {"outgoing": outgoing_paths, "incoming": incoming_paths}}
            rendered = _render_transition_audio(outgoing, incoming, {"method": "beatmix"}, 8, 0.8, stem_report)

        self.assertTrue(stem_report["renderMethod"] in {"quick_cut", "echo_out"})
        self.assertEqual(stem_report["renderMethod"], "echo_out")
        self.assertAlmostEqual(stem_report["renderOverlapDuration"], 8.0, places=1)
        self.assertGreater(rendered.shape[1], SAMPLE_RATE * 11)

    def test_cue_refinement_prefers_nearby_lower_vocal_bar(self) -> None:
        track_a = {
            "bars": [96, 100, 104, 108, 112],
            "phrases": [96, 112],
            "vocal_density_curve": [{"time": 100, "density": 0.8}, {"time": 104, "density": 0.08}],
            "energy_curve": [{"time": 100, "energy": 0.6}, {"time": 104, "energy": 0.42}],
        }
        track_b = {
            "bars": [0, 4, 8, 12],
            "phrases": [0, 8],
            "vocal_density_curve": [{"time": 0, "density": 0.1}],
            "energy_curve": [{"time": 0, "energy": 0.45}],
        }
        alignment = {
            "outgoingExitTime": 100,
            "incomingEntryTime": 0,
            "outgoingDownbeatTime": 100,
            "incomingDownbeatTime": 0,
            "overlapDuration": 8,
            "phraseAligned": True,
            "alignmentConfidence": 0.8,
        }

        refined = _refine_alignment_for_smoothness(track_a, track_b, alignment, {"method": "beatmix"})

        self.assertTrue(refined["cueRefinement"]["enabled"])
        self.assertEqual(refined["outgoingExitTime"], 104)

    def test_cue_refinement_avoids_next_outgoing_vocal_inside_overlap(self) -> None:
        track_a = {
            "bars": [96, 100, 104, 108, 112, 116],
            "phrases": [96, 112],
            "vocal_density_curve": [
                {"time": 100, "density": 0.55},
                {"time": 104, "density": 0.05},
                {"time": 108, "density": 0.86},
                {"time": 112, "density": 0.06},
                {"time": 116, "density": 0.04},
            ],
            "energy_curve": [
                {"time": 100, "energy": 0.58},
                {"time": 104, "energy": 0.5},
                {"time": 108, "energy": 0.63},
                {"time": 112, "energy": 0.42},
            ],
        }
        track_b = {
            "bars": [0, 4, 8, 12],
            "phrases": [0, 8],
            "vocal_density_curve": [{"time": 0, "density": 0.08}],
            "energy_curve": [{"time": 0, "energy": 0.48}],
        }
        alignment = {
            "outgoingExitTime": 100,
            "incomingEntryTime": 0,
            "outgoingDownbeatTime": 100,
            "incomingDownbeatTime": 0,
            "overlapDuration": 8,
            "phraseAligned": True,
            "alignmentConfidence": 0.8,
        }

        refined = _refine_alignment_for_smoothness(track_a, track_b, alignment, {"method": "beatmix"})

        self.assertTrue(refined["cueRefinement"]["enabled"])
        self.assertGreaterEqual(refined["outgoingExitTime"], 112)

    def test_energy_handoff_ducks_hot_incoming_intro(self) -> None:
        samples = SAMPLE_RATE * 4
        outgoing = np.full((2, samples), 0.12, dtype=np.float32)
        incoming = np.full((2, samples), 0.42, dtype=np.float32)

        profile = _energy_handoff_profile(outgoing, incoming, "beatmix")
        curves = _automation_curves(samples, "beatmix", 0.1, None, profile)

        self.assertLess(profile["incomingTrimDb"], -3)
        self.assertLess(float(curves["in_drums"][0, SAMPLE_RATE // 2]), float(curves["in_drums"][0, -1]))

    def test_transition_loudness_stabilizer_lifts_soft_overlap(self) -> None:
        before = np.full((2, SAMPLE_RATE * 2), 0.2, dtype=np.float32)
        overlap = np.full((2, SAMPLE_RATE * 2), 0.04, dtype=np.float32)
        after = np.full((2, SAMPLE_RATE * 2), 0.2, dtype=np.float32)
        rendered = np.concatenate([before, overlap, after], axis=1)

        stabilized, report = _stabilize_transition_loudness(rendered, SAMPLE_RATE * 2, SAMPLE_RATE * 2)

        lifted = stabilized[:, SAMPLE_RATE * 2 + SAMPLE_RATE // 2 : SAMPLE_RATE * 3]
        self.assertGreater(report["gainDb"], 3)
        self.assertGreater(float(np.mean(np.abs(lifted))), 0.09)
        self.assertLessEqual(float(np.max(np.abs(stabilized))), 1.0)

    def test_frequency_handoff_can_duck_only_hot_incoming_low_band(self) -> None:
        samples = SAMPLE_RATE * 4
        t = np.linspace(0, 4, samples, endpoint=False, dtype=np.float32)
        outgoing = np.vstack([0.08 * np.sin(2 * np.pi * 90 * t) + 0.08 * np.sin(2 * np.pi * 1200 * t)] * 2).astype(np.float32)
        incoming = np.vstack([0.45 * np.sin(2 * np.pi * 90 * t) + 0.08 * np.sin(2 * np.pi * 1200 * t)] * 2).astype(np.float32)

        profile = _frequency_handoff_profile(outgoing, incoming, "beatmix")
        curves = _automation_curves(samples, "beatmix", 0.1, None, None, profile)

        self.assertLess(profile["lowTrimDb"], -4)
        self.assertGreater(profile["midTrimDb"], -1)
        self.assertLess(float(curves["in_bass"][0, SAMPLE_RATE // 2]), float(curves["in_bass"][0, -1]))

    def test_transition_glue_layer_masks_middle_without_loud_edges(self) -> None:
        samples = SAMPLE_RATE * 2
        t = np.linspace(0, 2, samples, endpoint=False, dtype=np.float32)
        outgoing = np.vstack([0.18 * np.sin(2 * np.pi * 1200 * t)] * 2).astype(np.float32)
        incoming = np.vstack([0.18 * np.sin(2 * np.pi * 1800 * t)] * 2).astype(np.float32)

        glue = _transition_glue_layer(outgoing, incoming, "echo_out", 0.4)

        edge_energy = float(np.mean(np.abs(glue[:, : SAMPLE_RATE // 20]))) + float(np.mean(np.abs(glue[:, -SAMPLE_RATE // 20 :])))
        middle_energy = float(np.mean(np.abs(glue[:, SAMPLE_RATE // 2 : SAMPLE_RATE])))
        self.assertGreater(middle_energy, edge_energy)
        self.assertLessEqual(float(np.max(np.abs(glue))), 0.35)

    def test_rhythm_bridge_keeps_high_percussion_without_low_kick(self) -> None:
        samples = SAMPLE_RATE * 2
        t = np.linspace(0, 2, samples, endpoint=False, dtype=np.float32)
        drums = np.vstack([
            0.4 * np.sin(2 * np.pi * 90 * t) + 0.25 * np.sin(2 * np.pi * 6500 * t)
        ] * 2).astype(np.float32)

        bridge = _rhythm_bridge_layer(drums, "beatmix", 0.3)
        low = np.mean(np.abs(np.fft.rfft(bridge[0])[:80]))
        high = np.mean(np.abs(np.fft.rfft(bridge[0])[1200:4000]))

        self.assertGreater(float(high), float(low) * 2)
        self.assertGreater(float(np.mean(np.abs(bridge[:, : SAMPLE_RATE // 10]))), float(np.mean(np.abs(bridge[:, -SAMPLE_RATE // 10 :]))) * 5)

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
        self.assertIn("incomingBandTrimDb", result["processingReport"])
        self.assertIn("outgoingVocalGuarded", result["processingReport"])
        self.assertIn("renderMethod", result["processingReport"])
        self.assertIn("strategyAdapted", result["processingReport"])
        self.assertIn("renderOverlapDuration", result["processingReport"])
        self.assertIn("glueWet", result["processingReport"])
        self.assertIn("rhythmBridgeWet", result["processingReport"])
        self.assertEqual(result["outgoingCue"]["time"], result["alignment"]["outgoingExitTime"])
        self.assertEqual(result["incomingCue"]["time"], result["alignment"]["incomingEntryTime"])
        self.assertIn("requestedOutgoingCue", result)
        self.assertIn("requestedIncomingCue", result)


if __name__ == "__main__":
    unittest.main()
