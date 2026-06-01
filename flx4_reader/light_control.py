"""DDJ-FLX4 MIDI OUT light control."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union


class LightCommandError(ValueError):
    pass


@dataclass(frozen=True)
class MidiOutMessage:
    status: int
    data1: int
    data2: int


@dataclass(frozen=True)
class ParsedLightCommand:
    kind: str
    target: int
    value: Union[int, str]


LIGHT_MAP = {
    1: (0x90, 0x0B),
    2: (0x90, 0x0C),
    3: (0x91, 0x0B),
    4: (0x91, 0x0C),
    5: (0x90, 0x10),
    6: (0x90, 0x11),
    7: (0x91, 0x10),
    8: (0x91, 0x11),
    9: (0x90, 0x58),
    10: (0x91, 0x58),
    11: (0x90, 0x54),
    12: (0x91, 0x54),
    13: (0x96, 0x00),
    14: (0x96, 0x01),
    15: (0x90, 0x1B),
    16: (0x90, 0x1E),
    17: (0x90, 0x20),
    18: (0x90, 0x22),
    19: (0x91, 0x1B),
    20: (0x91, 0x1E),
    21: (0x91, 0x20),
    22: (0x91, 0x22),
    41: (0x90, 0x17),
    42: (0x91, 0x17),
}

PAD_MODE_BASE = {
    "HOT_CUE": 0x00,
    "PAD_FX1": 0x10,
    "BEAT_JUMP": 0x20,
    "SAMPLER": 0x30,
    "KEYBOARD": 0x40,
    "PAD_FX2": 0x50,
    "BEAT_LOOP": 0x60,
    "KEY_SHIFT": 0x70,
}

MODE_LIGHT_MAP = {
    15: (1, "HOT_CUE"),
    16: (1, "PAD_FX1"),
    17: (1, "BEAT_JUMP"),
    18: (1, "SAMPLER"),
    19: (2, "HOT_CUE"),
    20: (2, "PAD_FX1"),
    21: (2, "BEAT_JUMP"),
    22: (2, "SAMPLER"),
}

LEVEL_VALUE = {
    0: 0x00,
    1: 0x40,
    2: 0x56,
    3: 0x64,
    4: 0x76,
    5: 0x7F,
}

DEFAULT_PAD_MODE = "HOT_CUE"
DEMO_LIGHT_IDS = tuple(range(1, 23)) + tuple(range(23, 39)) + (41, 42)
DEMO_METER_IDS = (1, 2)
_PAD_MODE_ALIASES = {
    "".join(ch for ch in mode if ch.isalnum()): mode for mode in PAD_MODE_BASE
}


class LightController:
    def __init__(
        self,
        output: Any,
        pad_modes: Optional[Mapping[int, str]] = None,
        mido_module: Optional[Any] = None,
    ) -> None:
        self.output = output
        self._mido = mido_module
        self._lock = threading.Lock()
        self.pad_modes: Dict[int, str] = {1: DEFAULT_PAD_MODE, 2: DEFAULT_PAD_MODE}

        if pad_modes is not None:
            for deck, mode in pad_modes.items():
                self.set_pad_mode(deck, mode)

    def handle_line(self, line: str) -> Optional[MidiOutMessage]:
        command = parse_light_command(line)
        if command is None:
            return None

        if command.kind == "light":
            return self.send_light(command.target, int(command.value))
        if command.kind == "meter":
            return self.send_level_meter(command.target, int(command.value))
        if command.kind == "pad_mode":
            self.set_pad_mode(command.target, str(command.value))
            return None

        raise LightCommandError(f"Unsupported light command: {command.kind}")

    def send_light(self, light_id: int, state: int) -> MidiOutMessage:
        with self._lock:
            raw_message = build_light_message(light_id, state, self.pad_modes)
            self._send_raw(raw_message)
            if state == 1 and light_id in MODE_LIGHT_MAP:
                deck, mode = MODE_LIGHT_MAP[light_id]
                self.pad_modes[deck] = mode
            return raw_message

    def send_level_meter(self, meter_id: int, level: int) -> MidiOutMessage:
        with self._lock:
            raw_message = build_level_meter_message(meter_id, level)
            self._send_raw(raw_message)
            return raw_message

    def set_pad_mode(self, deck: int, mode: str) -> None:
        normalized_deck = _validate_deck(deck)
        normalized_mode = normalize_pad_mode(mode)
        with self._lock:
            self.pad_modes[normalized_deck] = normalized_mode

    def _send_raw(self, raw_message: MidiOutMessage) -> None:
        if self._mido is None:
            self._mido = _load_mido()
        self.output.send(to_mido_message(raw_message, self._mido))


def parse_light_command(line: str) -> Optional[ParsedLightCommand]:
    parts = _split_command_parts(line)
    if not parts:
        return None

    command = parts[0].upper()
    if command in ("L", "LIGHT"):
        if len(parts) != 3:
            raise LightCommandError("Use `L <id> <0|1>` or `L1 1`.")
        return ParsedLightCommand(
            "light",
            _parse_int(parts[1], "light id"),
            _parse_state(parts[2]),
        )

    if command in ("M", "METER", "LEVEL"):
        if len(parts) != 3:
            raise LightCommandError("Use `M <meter> <0..5>` or `M1 5`.")
        meter_id = _validate_meter(_parse_int(parts[1], "meter id"))
        level = _validate_level(_parse_int(parts[2], "meter level"))
        return ParsedLightCommand("meter", meter_id, level)

    if command in ("P", "PAD", "MODE", "PAD_MODE"):
        if len(parts) < 3:
            raise LightCommandError("Use `P <deck> <mode>` or `P1 HOT_CUE`.")
        deck = _validate_deck(_parse_int(parts[1], "deck"))
        mode = normalize_pad_mode("_".join(parts[2:]))
        return ParsedLightCommand("pad_mode", deck, mode)

    raise LightCommandError(f"Unknown light command: {parts[0]}")


def build_light_message(
    light_id: int,
    state: int,
    pad_modes: Optional[Mapping[int, str]] = None,
) -> MidiOutMessage:
    light_id = _parse_int(light_id, "light id")
    state = _parse_state(state)
    data2 = 0x7F if state else 0x00

    if light_id in LIGHT_MAP:
        status, data1 = LIGHT_MAP[light_id]
        return MidiOutMessage(status, data1, data2)

    if 23 <= light_id <= 38:
        deck = 1 if light_id <= 30 else 2
        pad = light_id - 22 if deck == 1 else light_id - 30
        mode = _pad_mode_for_deck(deck, pad_modes)
        status = 0x97 if deck == 1 else 0x99
        data1 = PAD_MODE_BASE[mode] + (pad - 1)
        return MidiOutMessage(status, data1, data2)

    raise LightCommandError(f"Unsupported light id: {light_id}")


def build_level_meter_message(meter_id: int, level: int) -> MidiOutMessage:
    meter_id = _validate_meter(_parse_int(meter_id, "meter id"))
    level = _validate_level(_parse_int(level, "meter level"))
    status = 0xB0 if meter_id == 1 else 0xB1
    return MidiOutMessage(status, 0x02, LEVEL_VALUE[level])


def to_mido_message(raw_message: MidiOutMessage, mido_module: Optional[Any] = None):
    mido = mido_module or _load_mido()
    status_type = raw_message.status & 0xF0
    channel = raw_message.status & 0x0F

    if status_type == 0x90:
        return mido.Message(
            "note_on",
            channel=channel,
            note=raw_message.data1,
            velocity=raw_message.data2,
        )

    if status_type == 0xB0:
        return mido.Message(
            "control_change",
            channel=channel,
            control=raw_message.data1,
            value=raw_message.data2,
        )

    raise LightCommandError(f"Unsupported MIDI status: 0x{raw_message.status:02X}")


def format_midi_bytes(raw_message: MidiOutMessage) -> str:
    return f"{raw_message.status:02X} {raw_message.data1:02X} {raw_message.data2:02X}"


def run_light_demo(
    controller: LightController,
    delay: float = 0.12,
    repeats: int = 1,
) -> None:
    if delay < 0:
        raise LightCommandError("Demo delay must be 0 or greater.")
    if repeats < 0:
        raise LightCommandError("Demo repeats must be 0 or greater.")

    controller.set_pad_mode(1, DEFAULT_PAD_MODE)
    controller.set_pad_mode(2, DEFAULT_PAD_MODE)

    remaining = repeats
    while repeats == 0 or remaining > 0:
        for light_id in DEMO_LIGHT_IDS:
            controller.send_light(light_id, 1)
            time.sleep(delay)
            controller.send_light(light_id, 0)
            time.sleep(delay)

        for meter_id in DEMO_METER_IDS:
            for level in range(1, 6):
                controller.send_level_meter(meter_id, level)
                time.sleep(delay)
            controller.send_level_meter(meter_id, 0)
            time.sleep(delay)

        if repeats != 0:
            remaining -= 1


def normalize_pad_mode(mode: str) -> str:
    key = "".join(ch for ch in str(mode).upper() if ch.isalnum())
    try:
        return _PAD_MODE_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(PAD_MODE_BASE)
        raise LightCommandError(f"Unsupported pad mode `{mode}`. Use one of: {choices}.") from exc


def _split_command_parts(line: str) -> list[str]:
    content = line.split("#", 1)[0].strip()
    if not content:
        return []

    parts = content.replace(",", " ").split()
    first = parts[0]
    prefix = first[:1].upper()
    if prefix in ("L", "M", "P") and len(first) > 1:
        return [prefix, first[1:]] + parts[1:]
    return parts


def _parse_int(value: object, name: str) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    try:
        if text.lower().startswith("0x"):
            return int(text, 16)
        return int(text, 10)
    except ValueError as exc:
        raise LightCommandError(f"Invalid {name}: {value}") from exc


def _parse_state(value: object) -> int:
    if isinstance(value, int):
        state = value
    else:
        text = str(value).strip().upper()
        if text in ("ON", "TRUE"):
            return 1
        if text in ("OFF", "FALSE"):
            return 0
        state = _parse_int(value, "light state")

    if state not in (0, 1):
        raise LightCommandError("Light state must be 0 or 1.")
    return state


def _validate_deck(deck: int) -> int:
    if deck not in (1, 2):
        raise LightCommandError("Deck must be 1 or 2.")
    return deck


def _validate_meter(meter_id: int) -> int:
    if meter_id not in (1, 2):
        raise LightCommandError("Meter id must be 1 or 2.")
    return meter_id


def _validate_level(level: int) -> int:
    if level not in LEVEL_VALUE:
        raise LightCommandError("Meter level must be 0 through 5.")
    return level


def _pad_mode_for_deck(deck: int, pad_modes: Optional[Mapping[int, str]]) -> str:
    if pad_modes is None:
        return DEFAULT_PAD_MODE
    return normalize_pad_mode(pad_modes.get(deck, DEFAULT_PAD_MODE))


def _load_mido():
    import mido

    return mido
