"""MIDI device discovery and opening for DDJ-FLX4."""

from __future__ import annotations

from typing import Iterable, List, Optional


FLX4_NAME_MARKERS = ("ddj-flx4", "ddj flx4", "flx4")


class MidiDependencyError(RuntimeError):
    pass


class MidiDeviceNotFoundError(RuntimeError):
    def __init__(self, devices: List[str]) -> None:
        super().__init__("DDJ-FLX4 MIDI input device was not found.")
        self.devices = devices


class MidiOutputDeviceNotFoundError(RuntimeError):
    def __init__(self, devices: List[str]) -> None:
        super().__init__("DDJ-FLX4 MIDI output device was not found.")
        self.devices = devices


class MidiDeviceOpenError(RuntimeError):
    def __init__(
        self,
        device_name: str,
        original_error: BaseException,
        port_type: str = "input",
    ) -> None:
        super().__init__(f"Could not open MIDI {port_type} device: {device_name}")
        self.device_name = device_name
        self.original_error = original_error
        self.port_type = port_type


def list_input_devices() -> List[str]:
    mido = _load_mido()
    try:
        return list(mido.get_input_names())
    except (ImportError, ModuleNotFoundError) as exc:
        raise MidiDependencyError(
            "Missing dependency: install with `pip install mido python-rtmidi`."
        ) from exc


def list_output_devices() -> List[str]:
    mido = _load_mido()
    try:
        return list(mido.get_output_names())
    except (ImportError, ModuleNotFoundError) as exc:
        raise MidiDependencyError(
            "Missing dependency: install with `pip install mido python-rtmidi`."
        ) from exc


def find_flx4_input_device(devices: Optional[Iterable[str]] = None) -> Optional[str]:
    names = list(devices if devices is not None else list_input_devices())
    for name in names:
        normalized = _normalize_device_name(name)
        if any(marker in normalized for marker in FLX4_NAME_MARKERS):
            return name
    return None


def find_flx4_output_device(devices: Optional[Iterable[str]] = None) -> Optional[str]:
    names = list(devices if devices is not None else list_output_devices())
    for name in names:
        normalized = _normalize_device_name(name)
        if any(marker in normalized for marker in FLX4_NAME_MARKERS):
            return name
    return None


def open_input_device(device_name: Optional[str] = None):
    mido = _load_mido()
    try:
        devices = list(mido.get_input_names())
    except (ImportError, ModuleNotFoundError) as exc:
        raise MidiDependencyError(
            "Missing dependency: install with `pip install mido python-rtmidi`."
        ) from exc
    target_name = device_name or find_flx4_input_device(devices)

    if target_name is None:
        raise MidiDeviceNotFoundError(devices)

    try:
        return mido.open_input(target_name)
    except (OSError, IOError, RuntimeError, ValueError) as exc:
        raise MidiDeviceOpenError(target_name, exc, "input") from exc


def open_output_device(device_name: Optional[str] = None):
    mido = _load_mido()
    try:
        devices = list(mido.get_output_names())
    except (ImportError, ModuleNotFoundError) as exc:
        raise MidiDependencyError(
            "Missing dependency: install with `pip install mido python-rtmidi`."
        ) from exc
    target_name = device_name or find_flx4_output_device(devices)

    if target_name is None:
        raise MidiOutputDeviceNotFoundError(devices)

    try:
        return mido.open_output(target_name)
    except (OSError, IOError, RuntimeError, ValueError) as exc:
        raise MidiDeviceOpenError(target_name, exc, "output") from exc


def _load_mido():
    try:
        import mido
    except ImportError as exc:
        raise MidiDependencyError(
            "Missing dependency: install with `pip install mido python-rtmidi`."
        ) from exc
    return mido


def _normalize_device_name(name: str) -> str:
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


def format_device_list(devices: List[str]) -> str:
    if not devices:
        return "No MIDI input devices detected."
    return "\n".join(f"- {device}" for device in devices)
