"""Make direct ``python scripts/<name>.py`` invocation work before installation."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
