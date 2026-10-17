"""Verify that every runtime dependency of e-Sevai Saathi is present.

Usage:
    python scripts/check_env.py

Exits with status 1 when a required dependency is missing, so it can be used
as a pre-flight step before `scripts/build_index.py`.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REQUIRED_PACKAGES: tuple[tuple[str, str], ...] = (
    ("pymupdf", "PyMuPDF"),
    ("pytesseract", "pytesseract"),
    ("PIL", "Pillow"),
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("sentence_transformers", "sentence-transformers"),
    ("faiss", "faiss-cpu"),
    ("rank_bm25", "rank-bm25"),
    ("google.genai", "google-genai"),
    ("streamlit", "streamlit"),
    ("yaml", "PyYAML"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("tqdm", "tqdm"),
)


def report(status: str, message: str) -> None:
    print(f"[{status:<7}] {message}")


def check_packages() -> list[str]:
    """Import every required package and report its version."""
    missing: list[str] = []
    for module_name, display_name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            report("MISSING", f"{display_name} ({module_name}): {exc.__class__.__name__}: {exc}")
            missing.append(display_name)
            continue
        version = getattr(module, "__version__", "")
        report("ok", f"{display_name} ({module_name}) {version}".rstrip())
    return missing


def check_torch_numpy() -> list[str]:
    """torch < 2.3 cannot use numpy >= 2, so prove the pair actually works."""
    import numpy as np
    import torch

    try:
        bridged = torch.from_numpy(np.arange(3)).tolist()
    except Exception as exc:
        report("FAIL", f"torch {torch.__version__} cannot read numpy {np.__version__}: {exc}")
        report("FAIL", "install numpy<2 with: python -m pip install \"numpy<2.0\"")
        return ["torch+numpy bridge"]

    report("ok", f"torch {torch.__version__} + numpy {np.__version__} ({bridged})")
    return []


def check_ocr() -> list[str]:
    """Confirm the Tesseract binary and the eng/tam language data are usable."""
    from tesseract import REQUIRED_LANGUAGES, available_languages, configure_tesseract

    located = configure_tesseract()
    if not located:
        report("FAIL", "Tesseract executable not found - screenshot OCR would be skipped")
        report("INFO", "install Tesseract, or set TESSERACT_CMD to its full path")
        return ["tesseract"]

    report("ok", f"Tesseract executable: {located}")

    languages = available_languages()
    if not languages:
        report("FAIL", "Tesseract ran but reported no language data")
        return ["tesseract languages"]

    report("ok", f"Tesseract languages: {', '.join(languages)}")
    absent = [language for language in REQUIRED_LANGUAGES if language not in set(languages)]
    if absent:
        report("FAIL", f"Tesseract lacks language data for: {', '.join(absent)}")
        report("INFO", "copy tam.traineddata into the Tesseract tessdata directory")
        return ["tesseract languages"]
    return []


def check_project_files() -> list[str]:
    """Load config.yaml and report manuals and index state."""
    from config import load_config

    try:
        config = load_config(ROOT / "config.yaml")
    except Exception as exc:
        report("FAIL", f"config.yaml could not be loaded: {exc.__class__.__name__}: {exc}")
        return ["config.yaml"]

    report("ok", f"config.yaml loaded (manuals directory: {config.manuals_dir})")

    manuals = sorted(config.manuals_dir.glob("*.pdf"))
    if manuals:
        report("ok", f"{len(manuals)} manual PDF(s) found")
    else:
        report("WARN", f"no PDFs in {config.manuals_dir}")

    index_files = (config.faiss_file, config.chunks_file, config.bm25_file)
    if all(path.exists() for path in index_files):
        report("ok", f"index present in {config.index_dir}")
    else:
        report("WARN", "index not built yet - run: python scripts/build_index.py")
    return []


def check_api_key() -> list[str]:
    """Report where the Gemini key comes from. Runs before config.yaml is loaded."""
    from env_file import load_env_file

    already_set = bool(os.getenv("GEMINI_API_KEY"))
    applied = load_env_file(ROOT / ".env")

    if already_set:
        report("ok", "GEMINI_API_KEY is set in the environment")
    elif "GEMINI_API_KEY" in applied:
        report("ok", "GEMINI_API_KEY loaded from .env")
    else:
        report("WARN", "GEMINI_API_KEY is not set - indexing works, answering does not")
    return []


def main() -> int:
    print("e-Sevai Saathi environment check")
    print(f"python {sys.version.split()[0]} at {sys.executable}")
    print()

    failures: list[str] = []
    failures += check_packages()
    failures += check_torch_numpy()
    failures += check_ocr()
    failures += check_api_key()
    failures += check_project_files()

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        print("Install everything with: python -m pip install -r requirements.txt")
        return 1

    print("All required dependencies are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
