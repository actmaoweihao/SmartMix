from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf


SAMPLE_RATE = 44100
STEM_NAMES = ("vocals", "drums", "bass", "other")


@dataclass(frozen=True)
class StemSeparationResult:
    engine: str
    device: str
    input_path: Path
    workspace: Path
    stems: dict[str, Path]


def demucs_available() -> bool:
    if shutil.which("demucs"):
        return True
    try:
        import demucs  # noqa: F401
    except Exception:
        return False
    return True


def resolve_torch_device(requested_device: str = "auto") -> str:
    import torch

    requested = (requested_device or "auto").lower()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be auto, cuda, or cpu")
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot see a CUDA GPU.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def prepare_demucs_input(input_path: str | Path, workspace: str | Path) -> Path:
    source = Path(input_path)
    target_workspace = Path(workspace)
    target_workspace.mkdir(parents=True, exist_ok=True)
    demucs_input = target_workspace / "demucs_input.wav"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "wav",
            str(demucs_input),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return demucs_input


def separate_demucs_stems(
    input_path: str | Path,
    workspace: str | Path,
    device: str = "auto",
) -> StemSeparationResult:
    if not demucs_available():
        raise RuntimeError("Demucs is not available. Install backend/requirements-tuning.txt first.")

    target_workspace = Path(workspace)
    resolved_device = resolve_torch_device(device)
    demucs_input = prepare_demucs_input(input_path, target_workspace)
    stems = separate_prepared_demucs_input(demucs_input, target_workspace, resolved_device)
    return StemSeparationResult(
        engine="demucs",
        device=resolved_device,
        input_path=demucs_input,
        workspace=target_workspace,
        stems=stems,
    )


def separate_prepared_demucs_input(input_path: str | Path, workspace: str | Path, device: str) -> dict[str, Path]:
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model("htdemucs")
    model.cpu()
    model.eval()

    audio = load_stereo(Path(input_path))
    wav = torch.from_numpy(audio)
    ref = wav.mean(0)
    ref_std = ref.std().clamp_min(1e-8)
    wav = (wav - ref.mean()) / ref_std

    with torch.no_grad():
        sources = apply_model(
            model,
            wav[None],
            device=device,
            shifts=1,
            split=True,
            overlap=0.25,
            progress=False,
            num_workers=0,
        )[0]
    if device == "cuda":
        torch.cuda.empty_cache()
    sources = sources * ref_std + ref.mean()

    out_dir = Path(workspace) / "demucs_api"
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = {}
    for source, name in zip(sources, model.sources):
        stem_path = out_dir / f"{name}.wav"
        stem_audio = source.detach().cpu().numpy().astype(np.float32)
        sf.write(stem_path, stem_audio.T, SAMPLE_RATE, subtype="PCM_16")
        stems[name] = stem_path

    if set(STEM_NAMES).issubset(stems):
        return stems
    raise RuntimeError("Demucs did not produce all four stems.")


def load_stereo(path: Path) -> np.ndarray:
    try:
        audio, source_sr = sf.read(path, always_2d=True, dtype="float32")
        if source_sr != SAMPLE_RATE:
            audio = librosa.resample(audio.T, orig_sr=source_sr, target_sr=SAMPLE_RATE).T
        y = audio.T
    except Exception:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=False)
    if y.ndim == 1:
        y = np.vstack([y, y])
    if y.shape[0] > 2:
        y = y[:2]
    return np.ascontiguousarray(y, dtype=np.float32)
