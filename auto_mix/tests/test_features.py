import math
import unittest

import numpy as np

from auto_mix import (
    analyze_audio,
    calculate_band_energy,
    normalize_loudness,
    pan_stereo,
)


class FeatureTests(unittest.TestCase):
    def test_stereo_width_calculation_does_not_crash(self):
        sr = 44100
        t = np.linspace(0.0, 1.0, sr, endpoint=False)
        left = 0.1 * np.sin(2 * math.pi * 440 * t)
        right = 0.1 * np.sin(2 * math.pi * 660 * t)
        y = np.column_stack([left, right]).astype(np.float32)
        features = analyze_audio(y, sr)
        self.assertIn("stereo_width", features)
        self.assertGreaterEqual(features["stereo_width"], 0.0)

    def test_band_energy_outputs_expected_bands(self):
        sr = 44100
        t = np.linspace(0.0, 1.0, sr, endpoint=False)
        y = (0.1 * np.sin(2 * math.pi * 100 * t)).astype(np.float32)
        bands = calculate_band_energy(y[:, None], sr)
        for name in ("sub", "bass", "low_mid", "mid", "high_mid", "high"):
            self.assertIn(name, bands)
            self.assertGreaterEqual(bands[name], 0.0)

    def test_pan_function_outputs_stereo(self):
        sr = 44100
        y = np.ones((sr, 1), dtype=np.float32) * 0.1
        out = pan_stereo(y, 0.5)
        self.assertEqual(out.shape, (sr, 2))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_loudness_normalize_no_nan(self):
        sr = 44100
        rng = np.random.default_rng(123)
        y = (rng.normal(0.0, 0.01, size=(sr, 2))).astype(np.float32)
        out = normalize_loudness(y, sr, -14.0)
        self.assertFalse(np.isnan(out).any())
        self.assertTrue(np.all(np.isfinite(out)))


if __name__ == "__main__":
    unittest.main()
