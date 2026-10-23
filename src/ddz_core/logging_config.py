"""Central Loguru setup. Game events are persisted separately, not via text logs."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger


def configure_logging(log_directory: str | Path | None = None, level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=sys.stderr.isatty(),
               format="{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level:<8} | "
                      "{name}:{function}:{line} | {extra} | {message}")
    if log_directory is not None:
        root = Path(log_directory)
        root.mkdir(parents=True, exist_ok=True)
        logger.add(root / "app-{time:YYYY-MM-DD}.jsonl", level=level,
                   serialize=True, rotation="25 MB", retention="14 days",
                   compression="gz", enqueue=True, backtrace=False, diagnose=False)


def default_log_directory() -> str | None:
    return os.environ.get("DDZ_LOG_DIR")
