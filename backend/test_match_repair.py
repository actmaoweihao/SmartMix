from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.repair import MatchRepairOptions, build_match_repair_plan, render_match_repair


class MatchRepairTests(unittest.TestCase):
    def test_builds_key_tempo_and_energy_repair_plan(self) -> None:
        source = {
            "id": "a",
            "name": "A",
            "path": "a.wav",
            "bpm": 120,
            "camelot": "8A",
            "energy_profile": {
                "lufs": -18.0,
                "low_frequency_ratio": 0.2,
            },
        }
        reference = {
            "id": "b",
            "name": "B",
            "path": "b.wav",
            "bpm": 126,
            "camelot": "2B",
            "energy_profile": {
                "lufs": -12.0,
                "low_frequency_ratio": 0.35,
            },
        }

        plan = build_match_repair_plan(source, reference, MatchRepairOptions(process_target="track_a", max_pitch_shift_semitones=6))

        self.assertEqual(plan["source"], "track_a")
        self.assertTrue(plan["pitch"]["enabled"])
        self.assertLessEqual(abs(plan["pitch"]["semitones"]), 6)
        self.assertTrue(plan["tempo"]["enabled"])
        self.assertAlmostEqual(plan["tempo"]["target_bpm"], 126)
        self.assertTrue(plan["energy"]["enabled"])
        self.assertGreaterEqual(len(plan["operations"]), 3)

    def test_render_energy_repair_writes_audio_file(self) -> None:
        sample_rate = 44100
        t = np.linspace(0, 1, sample_rate, endpoint=False, dtype=np.float32)
        mono = 0.05 * np.sin(2 * np.pi * 220 * t)
        stereo = np.vstack([mono, mono]).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            sf.write(source, stereo.T, sample_rate, subtype="PCM_16")
            plan = {
                "tempo": {"enabled": False},
                "pitch": {"enabled": False},
                "energy": {"enabled": True, "target_lufs": -16.0, "low_gain": 1.1},
            }

            output = render_match_repair(source, plan, MatchRepairOptions(format="wav"))

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
