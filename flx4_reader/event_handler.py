"""Handle normalized FLX4 events."""

from __future__ import annotations

import sys
from typing import Mapping, Optional, TextIO


class PrintEventHandler:
    def __init__(self, stream: Optional[TextIO] = None, flush: bool = True) -> None:
        self.stream = stream or sys.stdout
        self.flush = flush

    def handle(self, event: Mapping[str, object]) -> None:
        event_type = event["type"]
        event_id = event["id"]
        data = event["data"]

        if event_type == "C":
            data_text = f"{float(data):.6f}"
        else:
            data_text = str(data)

        print(f"{event_type} {event_id} {data_text}", file=self.stream, flush=self.flush)
