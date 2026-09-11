from __future__ import annotations

import os
from datetime import datetime


def vlog(message: str) -> None:
    if os.getenv("VERBOSE", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")