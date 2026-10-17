"""Configuration loading for e-Sevai Saathi."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import yaml

from env_file import load_env_file


@dataclass
class AppConfig:
    raw: dict[str, Any]
    root: Path

    def path(self, key: str) -> Path:
        value = self.raw["paths"][key]
        return self.root / value

    @property
    def manuals_dir(self) -> Path:
        return self.path("manuals_dir")

    @property
    def crops_dir(self) -> Path:
        return self.path("crops_dir")

    @property
    def index_dir(self) -> Path:
        return self.path("index_dir")

    @property
    def cache_dir(self) -> Path:
        return self.path("cache_dir")

    @property
    def chunks_file(self) -> Path:
        return self.path("chunks_file")

    @property
    def faiss_file(self) -> Path:
        return self.path("faiss_file")

    @property
    def bm25_file(self) -> Path:
        return self.path("bm25_file")

    @property
    def services_file(self) -> Path:
        return self.root / self.raw["paths"]["services_file"]

    @property
    def glossary_file(self) -> Path:
        return self.root / self.raw["paths"]["glossary_file"]

    @property
    def gemini_api_key(self) -> str | None:
        return os.getenv("GEMINI_API_KEY")

    @property
    def llm_cache_file(self) -> Path:
        return self.root / self.raw["llm"]["cache_file"]


def load_config(config_path: str | Path = "config.yaml") -> AppConfig:
    """Load YAML configuration and ensure required directories exist."""
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    root = path.parent
    # A .env next to config.yaml supplies local keys; real environment wins.
    load_env_file(root / ".env")
    config = AppConfig(raw=raw, root=root)
    for directory in (config.manuals_dir, config.crops_dir, config.index_dir, config.cache_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return config
