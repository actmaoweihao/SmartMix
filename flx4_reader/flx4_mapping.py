"""DDJ-FLX4 MIDI mapping.

Keys use mido's zero-based channel numbers:
official MIDI channel 1 -> 0, channel 2 -> 1, and so on.
The output values are intentionally only normalized ids.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

MidiKey = Tuple[int, int]
EventTarget = Tuple[str, int]


NOTE_MAP: Dict[MidiKey, EventTarget] = {}
CC_MAP: Dict[MidiKey, EventTarget] = {}
JOG_MAP: Dict[MidiKey, EventTarget] = {}
IGNORED_CC_MAP: Dict[MidiKey, str] = {}


def _add_button(channel: int, note: int) -> None:
    key = (channel, note)
    if key not in NOTE_MAP:
        NOTE_MAP[key] = ("B", len(NOTE_MAP) + 1)


def _add_buttons(channel: int, notes: List[int]) -> None:
    for note in notes:
        _add_button(channel, note)


def _add_control(channel: int, control: int) -> None:
    key = (channel, control)
    if key not in CC_MAP:
        CC_MAP[key] = ("C", len(CC_MAP) + 1)


def _ignore_lsb(channel: int, control: int) -> None:
    IGNORED_CC_MAP[(channel, control)] = "LSB"


def _build_note_map() -> None:
    # 1. DECK
    _add_buttons(0, [11, 14, 12, 72, 63, 54, 103, 16, 76, 17, 78])
    _add_buttons(1, [11, 14, 12, 72, 63, 54, 103, 16, 76, 17, 78])
    _add_buttons(0, [77, 80, 81, 62, 83, 61, 88, 92, 96])
    _add_buttons(1, [77, 80, 81, 62, 83, 61, 88, 92, 96])

    # 2. EFFECT
    _add_buttons(4, [16, 17])
    _add_buttons(5, [16, 17])
    _add_buttons(4, [99, 100, 74, 102, 75, 107, 71])
    _add_button(5, 71)
    _add_button(4, 67)
    _add_button(5, 67)

    # 3. MIXER
    _add_buttons(6, [99, 120])
    _add_buttons(0, [84, 104, 102, 82])
    _add_buttons(1, [84, 104, 102, 82])
    _add_buttons(6, [0, 8, 1, 9, 109])

    # 4. BROWSE
    _add_buttons(6, [65, 66, 70, 104, 71, 122])

    # 5. PERFORMANCE PAD mode buttons
    _add_buttons(0, [27, 105, 30, 107, 32, 109, 34, 111])
    _add_buttons(1, [27, 105, 30, 107, 32, 109, 34, 111])

    # 5. PERFORMANCE PAD 1-8.
    # Each pad has 8 modes. Channels 7/8 are deck 1 normal/shift,
    # channels 9/10 are deck 2 normal/shift.
    mode_note_bases = [0, 16, 32, 48, 64, 80, 96, 112]
    for pad_offset in range(8):
        for normal_channel, shift_channel in [(7, 8), (9, 10)]:
            for base in mode_note_bases:
                note = base + pad_offset
                _add_button(normal_channel, note)
                _add_button(shift_channel, note)


def _build_cc_map() -> None:
    # Jog wheels: top/side and shifted variants, left deck -> J 1, right deck -> J 2.
    for control in [34, 35, 41, 33]:
        JOG_MAP[(0, control)] = ("J", 1)
        JOG_MAP[(1, control)] = ("J", 2)

    # Absolute controls. For 14-bit controls the FLX4 also sends an LSB CC;
    # this v1 reader emits the MSB only, matching the required 0-127 percent rule.
    _add_control(0, 0)   # TEMPO deck 1
    _add_control(1, 0)   # TEMPO deck 2
    _ignore_lsb(0, 32)
    _ignore_lsb(1, 32)

    _add_control(4, 2)   # LEVEL/DEPTH
    _ignore_lsb(4, 34)

    _add_control(6, 8)   # MASTER LEVEL
    _ignore_lsb(6, 40)

    _add_control(0, 4)   # TRIM deck 1
    _add_control(1, 4)   # TRIM deck 2
    _ignore_lsb(0, 36)
    _ignore_lsb(1, 36)

    _add_control(0, 7)   # EQ HI deck 1
    _add_control(1, 7)   # EQ HI deck 2
    _ignore_lsb(0, 39)
    _ignore_lsb(1, 39)

    _add_control(0, 11)  # EQ MID deck 1
    _add_control(1, 11)  # EQ MID deck 2
    _ignore_lsb(0, 43)
    _ignore_lsb(1, 43)

    _add_control(0, 15)  # EQ LOW deck 1
    _add_control(1, 15)  # EQ LOW deck 2
    _ignore_lsb(0, 47)
    _ignore_lsb(1, 47)

    _add_control(6, 23)  # CFX deck 1
    _add_control(6, 24)  # CFX deck 2
    _ignore_lsb(6, 55)
    _ignore_lsb(6, 56)

    _add_control(0, 19)  # CH FADER deck 1
    _add_control(1, 19)  # CH FADER deck 2
    _ignore_lsb(0, 51)
    _ignore_lsb(1, 51)

    _add_control(6, 31)  # CROSSFADER
    _ignore_lsb(6, 63)

    _add_control(6, 5)   # MIC LEVEL
    _ignore_lsb(6, 37)

    _add_control(6, 12)  # HEADPHONE MIX
    _ignore_lsb(6, 44)

    _add_control(6, 13)  # HEADPHONE LEVEL
    _ignore_lsb(6, 45)

    _add_control(6, 64)   # BROWSE
    _add_control(6, 100)  # BROWSE + SHIFT


_build_note_map()
_build_cc_map()
