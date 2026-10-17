"""Locate the Tesseract binary and report the installed language data.

On Windows Tesseract is frequently installed without being added to PATH, so
`pytesseract` fails with TesseractNotFoundError even though the engine is
present. This module finds it once and configures `pytesseract` to use it.
"""
from __future__ import annotations

import logging
import os
import shutil
from functools import lru_cache
from pathlib import Path

import pytesseract

logger = logging.getLogger(__name__)

REQUIRED_LANGUAGES: tuple[str, ...] = ("eng", "tam")
_TESSERACT_EXE = "tesseract.exe" if os.name == "nt" else "tesseract"


def _candidate_dirs() -> list[Path]:
    """Directories that commonly hold a Tesseract install."""
    dirs: list[Path] = []
    program_files = os.environ.get("ProgramFiles")
    if program_files:
        dirs.append(Path(program_files) / "Tesseract-OCR")
    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    if program_files_x86:
        dirs.append(Path(program_files_x86) / "Tesseract-OCR")
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        dirs.append(Path(local_app_data) / "Programs" / "Tesseract-OCR")
    dirs.extend((Path("/usr/bin"), Path("/usr/local/bin"), Path("/opt/homebrew/bin")))
    return dirs


def find_tesseract() -> str | None:
    """Return the path to the Tesseract executable, or None if it is absent."""
    override = os.environ.get("TESSERACT_CMD")
    if override and Path(override).exists():
        return override

    on_path = shutil.which("tesseract")
    if on_path:
        return on_path

    for directory in _candidate_dirs():
        executable = directory / _TESSERACT_EXE
        if executable.exists():
            return str(executable)
    return None


@lru_cache(maxsize=1)
def configure_tesseract() -> str | None:
    """Point `pytesseract` at the located binary and return its path."""
    located = find_tesseract()
    if located:
        pytesseract.pytesseract.tesseract_cmd = located
        logger.debug("Using Tesseract at %s", located)
    else:
        logger.warning(
            "Tesseract was not found on PATH or in the usual install locations. "
            "Set the TESSERACT_CMD environment variable to its full path."
        )
    return located


@lru_cache(maxsize=1)
def available_languages() -> list[str]:
    """Return the language codes Tesseract can currently load."""
    configure_tesseract()
    try:
        return sorted(pytesseract.get_languages(config=""))
    except Exception as exc:  # Tesseract missing or not runnable
        logger.warning("Could not query Tesseract languages: %s", exc)
        return []


def missing_languages(required: tuple[str, ...] = REQUIRED_LANGUAGES) -> list[str]:
    """Return the required language codes that are not installed."""
    present = set(available_languages())
    return [language for language in required if language not in present]


def describe() -> str:
    """One-line summary of the OCR engine state, for logs and diagnostics."""
    located = configure_tesseract()
    if not located:
        return "Tesseract: not found"
    languages = available_languages() or ["<none>"]
    return f"Tesseract: {located} (languages: {', '.join(languages)})"
