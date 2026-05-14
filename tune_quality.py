from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backend.tuning import render_harmonic_tune


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="High-quality Camelot pitch tuning with Demucs stems and Rubber Band when available.",
    )
    parser.add_argument("input", help="Path to the source song")
    parser.add_argument("--source", default="9A", help="Source Camelot key, for example 9A")
    parser.add_argument("--target", default="3A", help="Target Camelot key, for example 3A")
    parser.add_argument("-o", "--output", help="Output path. Defaults to '<input>_<source>_to_<target>.wav'.")
    parser.add_argument("--direction", choices=("nearest", "up", "down"), default="nearest")
    parser.add_argument("--format", choices=("wav", "mp3"), default="wav")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device for Demucs stem separation. Default uses CUDA when available.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file does not exist: {input_path}", file=sys.stderr)
        return 2

    suffix = ".mp3" if args.format == "mp3" else ".wav"
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_{args.source}_to_{args.target}{suffix}")

    try:
        result = render_harmonic_tune(
            input_path=input_path,
            source_camelot=args.source,
            target_camelot=args.target,
            output_path=output_path,
            prefer_direction=args.direction,
            fmt=args.format,
            device=args.device,
        )
    except Exception as exc:
        print(f"Tuning failed: {exc}", file=sys.stderr)
        return 1

    print(f"Output: {result.path}")
    print(f"Camelot: {result.source_camelot} -> {result.target_camelot}")
    print(f"Pitch shift: {result.semitones:+d} semitones")
    print(f"Method: {result.method}")
    print(f"Device: {result.device}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
