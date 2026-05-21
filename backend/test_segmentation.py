from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.mixing import SAMPLE_RATE
from backend.segmentation import (
    analyze_track_segmentation,
    build_hierarchical_sections,
    build_minor_sections,
    compute_novelty_curve,
    compute_similarity_matrices,
    extract_bar_features,
    extract_groove_bed_candidates,
    extract_transition_safe_points,
    extract_vocal_phrases_from_sections,
    generate_boundary_candidates,
)


def synthetic_track(root: Path, seconds: int = 32, bpm: int = 120) -> tuple[dict, np.ndarray, dict[str, np.ndarray]]:
    t = np.linspace(0, seconds, SAMPLE_RATE * seconds, endpoint=False, dtype=np.float32)
    bars = [float(i) for i in range(seconds + 1)]
    low = 0.035 * np.sin(2 * np.pi * 180 * t)
    high = 0.10 * np.sin(2 * np.pi * 280 * t)
    audio = low.copy()
    audio[SAMPLE_RATE * 16 :] = high[SAMPLE_RATE * 16 :]
    audio += 0.05 * np.sin(2 * np.pi * 55 * t)
    drums = 0.04 * np.sin(2 * np.pi * 8 * t)
    bass = 0.08 * np.sin(2 * np.pi * 55 * t)
    other = 0.04 * np.sin(2 * np.pi * 220 * t)
    vocals = np.zeros_like(t)
    for start, end in ((4, 8), (8, 12), (16, 20), (20, 24)):
        vocals[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)] = 0.10 * np.sin(2 * np.pi * 440 * t[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)])
    full = audio + drums + bass + other + vocals
    path = root / "track.wav"
    sf.write(path, np.vstack([full, full]).T, SAMPLE_RATE, subtype="PCM_16")
    track = {
        "id": "track",
        "source": "A",
        "path": str(path),
        "duration": float(seconds),
        "bpm": bpm,
        "camelot": "8A",
        "key": "A minor",
        "bars": bars,
        "phrases": [0.0, 8.0, 16.0, 24.0, 32.0],
        "energy_profile": [{"time": float(i), "energy": 0.25 if i < 16 else 0.75} for i in range(seconds)],
        "transition_candidates": {"intro": 0.0, "outro": 24.0, "confidence": 0.7},
    }
    stems = {"vocals": vocals, "drums": drums, "bass": bass, "other": other}
    return track, full.astype(np.float32), stems


class SegmentationTests(unittest.TestCase):
    def test_bar_feature_extraction_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            track, audio, stems = synthetic_track(Path(tmp), seconds=16)
            features = extract_bar_features(audio, SAMPLE_RATE, track, stems)

        self.assertEqual(len(features), 16)
        self.assertEqual(len(features[0].chroma), 12)
        self.assertEqual(len(features[0].mfcc), 13)
        self.assertTrue(np.isfinite([features[0].energy, features[0].vocalDensity]).all())

    def test_similarity_matrices_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            track, audio, stems = synthetic_track(Path(tmp), seconds=16)
            features = extract_bar_features(audio, SAMPLE_RATE, track, stems)
            matrices = compute_similarity_matrices(features)

        for key in ("harmonic", "timbre", "rhythm", "energy", "fused"):
            self.assertEqual(matrices[key].shape, (16, 16))
            self.assertTrue(np.all(np.isfinite(matrices[key])))

    def test_novelty_curve_detects_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            track, audio, stems = synthetic_track(Path(tmp), seconds=32)
            features = extract_bar_features(audio, SAMPLE_RATE, track, stems)
            matrices = compute_similarity_matrices(features)
            novelty = compute_novelty_curve(matrices["fused"], [4, 8, 16])
        peak = int(np.argmax(novelty["fused_novelty"]))

        self.assertLessEqual(abs(peak - 16), 4)

    def test_boundary_does_not_cut_active_vocal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            track, audio, stems = synthetic_track(Path(tmp), seconds=16)
            features = extract_bar_features(audio, SAMPLE_RATE, track, stems)
            matrices = compute_similarity_matrices(features)
            novelty = compute_novelty_curve(matrices["fused"], [4])
            novelty["fused_novelty"][5] = 1.0
            boundaries = generate_boundary_candidates(features, novelty, track)

        risky = [item for item in boundaries if item["barIndex"] == 5]
        self.assertTrue(not risky or "cuts_vocal_phrase" in risky[0]["riskFlags"] or risky[0]["score"] < 0.5)

    def test_hierarchical_sections_not_fixed_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            track, audio, stems = synthetic_track(Path(tmp), seconds=32)
            features = extract_bar_features(audio, SAMPLE_RATE, track, stems)
            matrices = compute_similarity_matrices(features)
            novelty = compute_novelty_curve(matrices["fused"], [4, 8, 16])
            boundaries = generate_boundary_candidates(features, novelty, track)
            sections = build_hierarchical_sections(boundaries, features, matrices, track=track, source="A")

        starts = [section["barStart"] for section in sections]
        self.assertIn(16, starts)
        self.assertNotEqual(starts, [0, 16])

    def test_minor_sections_are_more_granular_than_major(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            track, audio, stems = synthetic_track(Path(tmp), seconds=32)
            features = extract_bar_features(audio, SAMPLE_RATE, track, stems)
            matrices = compute_similarity_matrices(features)
            novelty = compute_novelty_curve(matrices["fused"], [4, 8, 16])
            boundaries = generate_boundary_candidates(features, novelty, track)
            major = build_hierarchical_sections(boundaries, features, matrices, track=track, source="A")
            minor = build_minor_sections(boundaries, features, matrices, track=track, source="A")

        self.assertTrue(minor)
        self.assertLessEqual(max(section["bars"] for section in minor), 8)
        self.assertGreaterEqual(len(minor), len(major))

    def test_vocal_phrase_extraction_splits_long_vocal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            track, audio, stems = synthetic_track(Path(tmp), seconds=32)
            features = extract_bar_features(audio, SAMPLE_RATE, track, stems)
            matrices = compute_similarity_matrices(features)
            sections = build_hierarchical_sections([{"barIndex": 0}, {"barIndex": 16}], features, matrices, track=track, source="A")
            phrases = extract_vocal_phrases_from_sections(sections, features, stems["vocals"], track)

        self.assertGreaterEqual(len(phrases), 3)
        self.assertTrue(all(phrase["bars"] in {2, 4, 8} for phrase in phrases))
        self.assertLess(max(phrase["bars"] for phrase in phrases), 16)

    def test_vocal_phrase_extraction_uses_activity_islands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            track, audio, stems = synthetic_track(Path(tmp), seconds=16)
            vocals = np.zeros(SAMPLE_RATE * 16, dtype=np.float32)
            vocals[int(2.0 * SAMPLE_RATE) : int(4.0 * SAMPLE_RATE)] = 0.12
            vocals[int(10.0 * SAMPLE_RATE) : int(12.0 * SAMPLE_RATE)] = 0.12
            stems["vocals"] = vocals
            features = extract_bar_features(audio, SAMPLE_RATE, track, stems)
            matrices = compute_similarity_matrices(features)
            sections = [{"id": "A_sec_manual", "trackId": "track", "source": "A", "start": 0.0, "end": 16.0, "barStart": 0, "barEnd": 16, "bars": 16, "vocalDensity": 0.4}]
            phrases = extract_vocal_phrases_from_sections(sections, features, vocals, track)

        self.assertTrue(any(phrase["barStart"] == 2 for phrase in phrases))
        self.assertTrue(any(phrase["barStart"] == 10 for phrase in phrases))
        self.assertFalse(any(phrase["barStart"] == 4 and phrase["barEnd"] == 10 for phrase in phrases))

    def test_vocal_phrase_pickup_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            track, audio, stems = synthetic_track(Path(tmp), seconds=12)
            vocals = np.zeros(SAMPLE_RATE * 12, dtype=np.float32)
            vocals[int(3.4 * SAMPLE_RATE) : int(6.6 * SAMPLE_RATE)] = 0.11
            stems["vocals"] = vocals
            features = extract_bar_features(audio, SAMPLE_RATE, track, stems)
            matrices = compute_similarity_matrices(features)
            sections = build_hierarchical_sections([{"barIndex": 0}, {"barIndex": 8}], features, matrices, track=track, source="A")
            phrases = extract_vocal_phrases_from_sections(sections, features, vocals, track)

        self.assertTrue(any(phrase["hasPickup"] for phrase in phrases))
        self.assertTrue(any(phrase["hasTail"] for phrase in phrases))

    def test_groove_bed_excludes_vocals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            track, audio, stems = synthetic_track(Path(tmp), seconds=32)
            report = analyze_track_segmentation(track, "A", audio=audio, stems=stems)

        self.assertTrue(report["grooveBedCandidates"])
        self.assertTrue(all(candidate["usesStems"] == ["drums", "bass", "other"] for candidate in report["grooveBedCandidates"]))

    def test_groove_bed_loopability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            track, audio, stems = synthetic_track(Path(tmp), seconds=16)
            features = extract_bar_features(audio, SAMPLE_RATE, track, stems)
            sections = [{"id": "A_sec_manual", "trackId": "track", "source": "A", "start": 0.0, "end": 16.0, "barStart": 0, "barEnd": 16, "bars": 16, "vocalDensity": 0.0}]
            loopable_candidates = extract_groove_bed_candidates(sections, features, stems, track)
            loopable = next(candidate for candidate in loopable_candidates if candidate["bars"] == 16)
            features[-1].bassEnergy = 0.0
            features[-1].rms = 0.0
            features[-1].spectralCentroid = 8000.0
            not_loopable_candidates = extract_groove_bed_candidates(sections, features, stems, track)
            not_loopable = next(candidate for candidate in not_loopable_candidates if candidate["bars"] == 16)

        self.assertGreater(loopable["loopability"], not_loopable["loopability"])
        self.assertGreaterEqual(not_loopable_candidates[0]["loopability"], not_loopable["loopability"])

    def test_groove_bed_scans_subsections_away_from_vocals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            track, audio, stems = synthetic_track(Path(tmp), seconds=16)
            vocals = np.zeros(SAMPLE_RATE * 16, dtype=np.float32)
            vocals[int(4.0 * SAMPLE_RATE) : int(12.0 * SAMPLE_RATE)] = 0.12
            stems["vocals"] = vocals
            features = extract_bar_features(audio, SAMPLE_RATE, track, stems)
            sections = [{"id": "A_sec_manual", "trackId": "track", "source": "A", "start": 0.0, "end": 16.0, "barStart": 0, "barEnd": 16, "bars": 16, "vocalDensity": 0.5}]
            candidates = extract_groove_bed_candidates(sections, features, stems, track)

        self.assertTrue(candidates)
        self.assertLessEqual(candidates[0]["vocalLeakage"], 0.28)
        self.assertLess(candidates[0]["bars"], 16)

    def test_safe_cut_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            track, audio, stems = synthetic_track(Path(tmp), seconds=16)
            features = extract_bar_features(audio, SAMPLE_RATE, track, stems)
            matrices = compute_similarity_matrices(features)
            sections = build_hierarchical_sections([{"barIndex": 0}, {"barIndex": 8}], features, matrices, track=track, source="A")
            points = extract_transition_safe_points(sections, features, track)

        low_vocal = max((point for point in points if point["barIndex"] in {0, 1, 2}), key=lambda item: item["score"])
        mid_vocal = [point for point in points if point["barIndex"] in {5, 6}]
        self.assertGreater(low_vocal["score"], 0.5)
        self.assertTrue(all(point["score"] < low_vocal["score"] for point in mid_vocal))

    def test_segmentation_report_contains_debug(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            track, audio, stems = synthetic_track(Path(tmp), seconds=16)
            report = analyze_track_segmentation(track, "A", audio=audio, stems=stems)

        self.assertEqual(report["method"], "multi_scale_ssm_novelty_stem_corrected")
        self.assertTrue(report["sections"])
        self.assertTrue(report["minorSections"])
        self.assertIn("vocalPhrases", report)
        self.assertIn("grooveBedCandidates", report)
        self.assertIn("safeCutPoints", report)
        self.assertIn("debug", report)


if __name__ == "__main__":
    unittest.main()
