"""Minimal `.env` loader so a local GEMINI_API_KEY file is actually used.

Kept dependency-free on purpose: only `KEY=VALUE` lines are needed, and
python-dotenv would be an extra install for that alone. Real environment
variables always win over the file.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_COMMENT_PREFIX = "#"
_EXPORT_PREFIX = "export "
_QUOTE_CHARS = "\"'"


def parse_env_text(text: str) -> dict[str, str]:
    """Parse `KEY=VALUE` lines, skipping blanks and comments."""
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(_COMMENT_PREFIX):
            continue
        if line.startswith(_EXPORT_PREFIX):
            line = line[len(_EXPORT_PREFIX):].strip()

        key, separator, value = line.partition("=")
        if not separator:
            continue

        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in _QUOTE_CHARS:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_env_file(path: str | Path) -> dict[str, str]:
    """Load `path` into the process environment without clobbering set variables.

    Returns the variables that were actually applied.
    """
    path = Path(path)
    if not path.exists():
        return {}

    try:
        values = parse_env_text(path.read_text(encoding="utf-8"))
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return {}

    applied: dict[str, str] = {}
    for key, value in values.items():
        if value and key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    if applied:
        logger.debug("Loaded %s from %s", ", ".join(sorted(applied)), path)
    return applied
