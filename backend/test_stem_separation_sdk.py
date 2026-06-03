from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


class StemSeparationSdkTests(unittest.TestCase):
    def test_separate_demucs_stems_returns_stable_result_shape(self) -> None:
        from backend.services import stem_separation

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            source = workspace / "song.mp3"
            prepared = workspace / "demucs_input.wav"
            source.write_bytes(b"audio")
            paths = {stem: workspace / "demucs_api" / f"{stem}.wav" for stem in stem_separation.STEM_NAMES}

            with (
                patch("backend.services.stem_separation.demucs_available", return_value=True),
                patch("backend.services.stem_separation.resolve_torch_device", return_value="cpu"),
                patch("backend.services.stem_separation.prepare_demucs_input", return_value=prepared),
                patch("backend.services.stem_separation.separate_prepared_demucs_input", return_value=paths),
            ):
                result = stem_separation.separate_demucs_stems(source, workspace, device="auto")

        self.assertEqual(result.engine, "demucs")
        self.assertEqual(result.device, "cpu")
        self.assertEqual(result.input_path, prepared)
        self.assertEqual(set(result.stems), {"vocals", "drums", "bass", "other"})

    def test_tracks_api_uses_stem_separation_sdk_for_uncached_request(self) -> None:
        from fastapi.testclient import TestClient

        import backend.main as api
        from backend.api import tracks as tracks_api
        from backend.services import tracks as tracks_service
        from backend.services.stem_separation import StemSeparationResult

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            uploads = root / "uploads"
            stems = root / "stems"
            uploads.mkdir()
            stems.mkdir()
            source = uploads / "track-1.wav"
            source.write_bytes(b"audio")
            out_dir = stems / "track-1" / "demucs_api"
            out_dir.mkdir(parents=True)
            stem_paths = {}
            for stem in ("vocals", "drums", "bass", "other"):
                stem_path = out_dir / f"{stem}.wav"
                stem_path.write_bytes(f"{stem}-audio".encode("utf-8"))
                stem_paths[stem] = stem_path

            original_upload_dir = tracks_service.UPLOAD_DIR
            original_service_stem_dir = tracks_service.STEM_DIR
            original_api_stem_dir = tracks_api.STEM_DIR
            tracks_service.UPLOAD_DIR = uploads
            tracks_service.STEM_DIR = stems
            tracks_api.STEM_DIR = stems
            try:
                api.write_json(
                    uploads / "track-1.json",
                    {"id": "track-1", "name": "Track 1.wav", "path": str(source), "content_type": "audio/wav"},
                )
                sdk_result = StemSeparationResult(
                    engine="demucs",
                    device="cpu",
                    input_path=stems / "track-1" / "demucs_input.wav",
                    workspace=stems / "track-1",
                    stems=stem_paths,
                )
                with (
                    patch("backend.api.tracks.demucs_available", return_value=True),
                    patch("backend.api.tracks.separate_demucs_stems", return_value=sdk_result) as separate,
                ):
                    response = TestClient(api.app).post(
                        "/api/tracks/track-1/stems",
                        json={"device": "cpu", "force": True},
                    )

                payload = response.json()
            finally:
                tracks_service.UPLOAD_DIR = original_upload_dir
                tracks_service.STEM_DIR = original_service_stem_dir
                tracks_api.STEM_DIR = original_api_stem_dir

        self.assertEqual(response.status_code, 200)
        separate.assert_called_once_with(source, stems / "track-1", "cpu")
        self.assertFalse(payload["cached"])
        self.assertEqual(payload["device"], "cpu")
        self.assertEqual(payload["stems"]["bass"]["url"], "/api/tracks/track-1/stems/bass/audio")


if __name__ == "__main__":
    unittest.main()
