from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunLog:
    def __init__(self, path: Path, *, reset: bool = True) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            path.write_text("", encoding="utf-8")

    def write(self, event: str, **fields: Any) -> None:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
