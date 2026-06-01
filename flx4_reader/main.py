"""Command line entry point for the DDJ-FLX4 reader."""

from __future__ import annotations

import argparse
import sys
import threading
from contextlib import ExitStack
from typing import Optional, Sequence, TextIO

from .event_handler import PrintEventHandler
from .event_parser import EventParser
from .light_control import LightCommandError, LightController, run_light_demo
from .midi_device import (
    MidiDependencyError,
    MidiDeviceNotFoundError,
    MidiDeviceOpenError,
    MidiOutputDeviceNotFoundError,
    format_device_list,
    list_input_devices,
    list_output_devices,
    open_input_device,
    open_output_device,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read DDJ-FLX4 MIDI input events.")
    parser.add_argument(
        "--device",
        help="Open a specific MIDI input device instead of auto-detecting DDJ-FLX4.",
    )
    parser.add_argument(
        "--out-device",
        help="Open a specific MIDI output device for light control.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print unmapped MIDI messages to stderr.",
    )
    parser.add_argument(
        "--lights",
        action="store_true",
        help="Read light commands from stdin and send them to DDJ-FLX4 MIDI OUT.",
    )
    parser.add_argument(
        "--light-command",
        action="append",
        default=[],
        metavar="COMMAND",
        help="Send one light command and exit. Can be repeated.",
    )
    parser.add_argument(
        "--light-demo",
        action="store_true",
        help="Blink supported FLX4 lights in sequence and exit.",
    )
    parser.add_argument(
        "--light-demo-delay",
        type=float,
        default=0.12,
        help="Seconds between demo light changes.",
    )
    parser.add_argument(
        "--light-demo-repeat",
        type=int,
        default=1,
        help="Demo repeat count. Use 0 to repeat forever.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List MIDI input/output devices and exit.",
    )
    return parser


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        if args.list_devices:
            print(_format_device_lists(list_input_devices(), list_output_devices()))
            return 0

        if args.light_command:
            with open_output_device(args.out_device) as midi_output:
                controller = LightController(midi_output)
                for command in args.light_command:
                    controller.handle_line(command)
            return 0

        if args.light_demo:
            with open_output_device(args.out_device) as midi_output:
                controller = LightController(midi_output)
                run_light_demo(
                    controller,
                    delay=args.light_demo_delay,
                    repeats=args.light_demo_repeat,
                )
            return 0

        parser = EventParser(debug=args.debug)
        handler = PrintEventHandler()
        lights_enabled = args.lights or bool(args.out_device)

        with ExitStack() as stack:
            midi_input = stack.enter_context(open_input_device(args.device))

            if lights_enabled:
                midi_output = stack.enter_context(open_output_device(args.out_device))
                controller = LightController(midi_output)
                _start_light_command_thread(controller, sys.stdin, sys.stderr)

            for message in midi_input:
                event = parser.parse(message)
                if event is not None:
                    handler.handle(event)

    except MidiDependencyError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except MidiDeviceNotFoundError as exc:
        print("DDJ-FLX4 MIDI input device was not found.", file=sys.stderr)
        print(_format_devices_or_empty(exc.devices, "No MIDI input devices detected."), file=sys.stderr)
        return 1
    except MidiOutputDeviceNotFoundError as exc:
        print("DDJ-FLX4 MIDI output device was not found.", file=sys.stderr)
        print(_format_devices_or_empty(exc.devices, "No MIDI output devices detected."), file=sys.stderr)
        return 1
    except MidiDeviceOpenError as exc:
        port_type = getattr(exc, "port_type", "input")
        print(
            f"Could not open DDJ-FLX4 MIDI {port_type} port. It may already be in use.",
            file=sys.stderr,
        )
        print(f"Device: {exc.device_name}", file=sys.stderr)
        print(f"Error: {exc.original_error}", file=sys.stderr)
        return 1
    except LightCommandError as exc:
        print(f"LIGHT ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0

    return 0


def _start_light_command_thread(
    controller: LightController,
    stream: TextIO,
    error_stream: TextIO,
) -> threading.Thread:
    thread = threading.Thread(
        target=_read_light_commands,
        args=(controller, stream, error_stream),
        name="flx4-light-commands",
        daemon=True,
    )
    thread.start()
    return thread


def _read_light_commands(
    controller: LightController,
    stream: TextIO,
    error_stream: TextIO,
) -> None:
    for line in stream:
        try:
            controller.handle_line(line)
        except Exception as exc:
            print(f"LIGHT ERROR: {exc}", file=error_stream, flush=True)


def _format_device_lists(input_devices: list[str], output_devices: list[str]) -> str:
    return "\n".join(
        [
            "Input devices:",
            _format_devices_or_empty(input_devices, "No MIDI input devices detected."),
            "",
            "Output devices:",
            _format_devices_or_empty(output_devices, "No MIDI output devices detected."),
        ]
    )


def _format_devices_or_empty(devices: list[str], empty_text: str) -> str:
    if not devices:
        return empty_text
    return format_device_list(devices)


if __name__ == "__main__":
    raise SystemExit(run())
