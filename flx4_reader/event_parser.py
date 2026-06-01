"""Convert raw mido messages into normalized FLX4 events."""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional, TextIO

from .flx4_mapping import CC_MAP, IGNORED_CC_MAP, JOG_MAP, NOTE_MAP

Event = Dict[str, object]


class EventParser:
    def __init__(self, debug: bool = False, debug_stream: Optional[TextIO] = None) -> None:
        self.debug = debug
        self.debug_stream = debug_stream or sys.stderr

    def parse(self, message: Any) -> Optional[Event]:
        message_type = getattr(message, "type", None)

        if message_type in ("note_on", "note_off"):
            return self._parse_note(message)

        if message_type == "control_change":
            return self._parse_control_change(message)

        if self.debug:
            print(f"UNMAPPED type={message_type}", file=self.debug_stream)
        return None

    def _parse_note(self, message: Any) -> Optional[Event]:
        channel = getattr(message, "channel", None)
        note = getattr(message, "note", None)
        velocity = getattr(message, "velocity", 0)

        target = NOTE_MAP.get((channel, note))
        if target is None:
            if self.debug:
                print(
                    f"UNMAPPED note channel={_display_channel(channel)} note={note} velocity={velocity}",
                    file=self.debug_stream,
                )
            return None

        event_type, event_id = target
        state = 0 if message.type == "note_off" or velocity == 0 else 1
        return {"type": event_type, "id": event_id, "data": state}

    def _parse_control_change(self, message: Any) -> Optional[Event]:
        channel = getattr(message, "channel", None)
        control = getattr(message, "control", None)
        value = getattr(message, "value", 0)
        key = (channel, control)

        target = JOG_MAP.get(key)
        if target is not None:
            direction = _jog_direction(value)
            if direction is None:
                return None
            event_type, event_id = target
            return {"type": event_type, "id": event_id, "data": direction}

        target = CC_MAP.get(key)
        if target is not None:
            event_type, event_id = target
            percent = round((value / 127) * 100, 6)
            return {"type": event_type, "id": event_id, "data": percent}

        if key in IGNORED_CC_MAP:
            return None

        if self.debug:
            print(
                f"UNMAPPED control channel={_display_channel(channel)} control={control} value={value}",
                file=self.debug_stream,
            )
        return None


def _jog_direction(value: int) -> Optional[int]:
    if value > 64:
        return 1
    if value < 64:
        return -1
    return None


def _display_channel(channel: object) -> object:
    if isinstance(channel, int):
        return channel + 1
    return channel

