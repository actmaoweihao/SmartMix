from __future__ import annotations

import argparse
import sys
from pathlib import Path


SOURCE_CAMELOT = "9A"
TARGET_CAMELOT = "3A"

# 9A is E minor and 3A is B-flat/A-sharp minor. They are a tritone apart.
# Raising or lowering by 6 semitones reaches the same pitch class; lowering is
# often more comfortable for full songs because it avoids an overly bright sound.
SEMITONES_9A_TO_3A = -6


def choose_input_file() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    file_name = filedialog.askopenfilename(
        title="Choose a 9A song to convert to 3A",
        filetypes=[
            ("Audio files", "*.wav *.mp3 *.flac *.m4a *.ogg *.aiff *.aif"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return Path(file_name) if file_name else None


def build_output_path(input_path: Path, output_path: str | None) -> Path:
    if output_path:
        path = Path(output_path)
        if path.suffix:
            return path
        return path / f"{input_path.stem}_{SOURCE_CAMELOT}_to_{TARGET_CAMELOT}.wav"

    return input_path.with_name(f"{input_path.stem}_{SOURCE_CAMELOT}_to_{TARGET_CAMELOT}.wav")


def convert_9a_to_3a(input_path: Path, output_path: Path, semitones: float) -> None:
    try:
        import librosa
        import soundfile as sf
    except ImportError as exc:
        missing = exc.name or "audio dependencies"
        raise RuntimeError(
            f"Missing dependency: {missing}. Install project audio deps with:\n"
            "python -m pip install -r backend/requirements.txt"
        ) from exc

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio, sample_rate = librosa.load(input_path, sr=None, mono=False)
    shifted = librosa.effects.pitch_shift(
        y=audio,
        sr=sample_rate,
        n_steps=semitones,
        bins_per_octave=12,
    )

    # soundfile expects shape (frames, channels), while librosa loads stereo as
    # (channels, frames).
    if shifted.ndim == 2:
        shifted = shifted.T

    sf.write(output_path, shifted, sample_rate)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Camelot 9A song to Camelot 3A by pitch-shifting it 6 semitones.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to the source song. If omitted, a file picker opens.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output audio path or output folder. Defaults to '<input>_9A_to_3A.wav'.",
    )
    parser.add_argument(
        "--direction",
        choices=("down", "up"),
        default="down",
        help="Use 'down' for -6 semitones or 'up' for +6 semitones. Both land on 3A.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_path = Path(args.input) if args.input else choose_input_file()

    if input_path is None:
        print("No input file selected. Pass a file path or choose one in the dialog.", file=sys.stderr)
        return 2

    semitones = SEMITONES_9A_TO_3A if args.direction == "down" else abs(SEMITONES_9A_TO_3A)
    output_path = build_output_path(input_path, args.output)

    try:
        convert_9a_to_3a(input_path, output_path, semitones)
    except Exception as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 1

    print(f"Converted {SOURCE_CAMELOT} -> {TARGET_CAMELOT}: {output_path}")
    print(f"Pitch shift: {semitones:+g} semitones")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
