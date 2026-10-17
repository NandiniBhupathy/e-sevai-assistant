"""Small provider-swappable Gemini wrapper with caching and retry."""
from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover
    genai = None
    types = None


class LLMError(RuntimeError):
    """Raised when an LLM request cannot be completed."""


class GeminiLLM:
    """Gemini client with disk cache, throttling, JSON mode and retry."""

    def __init__(self, config):
        self.config = config
        self.model = config.raw["llm"]["model"]
        self.temperature = float(config.raw["llm"].get("temperature", 0.1))
        self.max_tokens = int(config.raw["llm"].get("max_output_tokens", 1200))
        self.min_delay = float(config.raw["llm"].get("min_delay_seconds", 2.0))
        self.max_retries = int(config.raw["llm"].get("max_retries", 5))
        self.cache_file = config.llm_cache_file
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._last_call = 0.0
        self._client = None

    def _client_or_raise(self):
        if not self.config.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set.")
        if genai is None:
            raise LLMError("google-genai is not installed. Run pip install -r requirements.txt.")
        if self._client is None:
            self._client = genai.Client(api_key=self.config.gemini_api_key)
        return self._client

    def _load_cache(self) -> dict[str, str]:
        if not self.cache_file.exists():
            return {}
        try:
            return json.loads(self.cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("LLM cache is unreadable; starting with an empty cache.")
            return {}

    def _save_cache(self, cache: dict[str, str]) -> None:
        tmp = self.cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.cache_file)

    @staticmethod
    def _key(prompt: str, json_mode: bool) -> str:
        payload = ("json\n" if json_mode else "text\n") + prompt
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _looks_like_rate_limit(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(token in text for token in ("429", "resource_exhausted", "rate limit", "quota"))

    def generate(self, prompt: str, *, json_mode: bool = False) -> str:
        """Generate text, using cache and exponential backoff for rate limits."""
        key = self._key(prompt, json_mode)
        cache = self._load_cache()
        if key in cache:
            return cache[key]

        client = self._client_or_raise()
        for attempt in range(self.max_retries):
            wait = self.min_delay - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                config_kwargs: dict[str, Any] = {
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                }
                if json_mode:
                    config_kwargs["response_mime_type"] = "application/json"
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                self._last_call = time.monotonic()
                text = (response.text or "").strip()
                if not text:
                    raise LLMError("Gemini returned an empty response.")
                cache[key] = text
                self._save_cache(cache)
                return text
            except Exception as exc:
                self._last_call = time.monotonic()
                if attempt >= self.max_retries - 1:
                    raise LLMError(f"LLM request failed after retries: {exc}") from exc
                if self._looks_like_rate_limit(exc) or "timeout" in str(exc).lower() or "connection" in str(exc).lower():
                    delay = min(60.0, (2 ** attempt) + random.random())
                    logger.warning("LLM request retry %d/%d in %.1fs: %s", attempt + 1, self.max_retries, delay, exc)
                    time.sleep(delay)
                else:
                    raise LLMError(f"LLM request failed: {exc}") from exc
        raise LLMError("LLM request failed unexpectedly.")

    def generate_json(self, prompt: str, repair_prompt: str | None = None) -> dict[str, Any]:
        """Generate and parse JSON; make one repair call for malformed JSON."""
        raw = self.generate(prompt, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if not repair_prompt:
                raise LLMError("LLM returned malformed JSON.")
            repaired = self.generate(repair_prompt + "\n\nMalformed response:\n" + raw, json_mode=True)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError as exc:
                raise LLMError("LLM returned malformed JSON after repair.") from exc


def extract_image_text(image_path: str | Path) -> str:
    """Swap-in hook for future vision models; currently OCR is performed elsewhere."""
    return ""
