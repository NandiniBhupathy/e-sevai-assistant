"""End-to-end query understanding, retrieval, answering and citation assembly."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from config import load_config
from index import SearchIndex
from llm import GeminiLLM, LLMError
from prompts import answer_prompt, abstain_message, clarification_message, understand_prompt
from retrieve import retrieve

logger = logging.getLogger(__name__)


class Pipeline:
    """Reusable pipeline for Streamlit and evaluation."""

    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.services = json.loads(self.config.services_file.read_text(encoding="utf-8"))
        self.glossary = json.loads(self.config.glossary_file.read_text(encoding="utf-8"))
        self.llm = GeminiLLM(self.config)
        self.index = SearchIndex(self.config)
        self.index.load()

    def _understand(self, question: str, user_type: str):
        prompt = understand_prompt(question, user_type, self.services)
        repair = "Return ONLY valid JSON matching the requested schema. Do not use markdown."
        result = self.llm.generate_json(prompt, repair)
        result.setdefault("user_type", "operator" if user_type.lower() == "operator" else "citizen")
        return result

    def answer(self, question: str, user_type="auto", service_code=None):
        """Answer a user question and return a serializable result."""
        if not question.strip():
            return {"answer": "Please enter a question.", "found": False, "sources": [], "images": [], "chunks": []}

        understanding = self._understand(question, user_type)
        if service_code:
            understanding["service_code"] = service_code
            understanding["needs_service_clarification"] = False

        if understanding.get("needs_service_clarification"):
            return {
                "answer": clarification_message(understanding.get("language", "en"), self.services),
                "found": False, "sources": [], "images": [], "chunks": [], "understanding": understanding
            }

        results, best_dense = retrieve(
            self.index,
            understanding.get("english_query") or question,
            question,
            understanding.get("service_code"),
            bool(understanding.get("is_generic")),
            self.config,
        )
        if not results:
            return {
                "answer": abstain_message(understanding.get("language", "en")),
                "found": False, "sources": [], "images": [], "chunks": [],
                "understanding": understanding, "best_similarity": best_dense,
            }

        prompt = answer_prompt(question, understanding, results, self.glossary)
        repair = "Return ONLY valid JSON. used_chunks must be labels such as C1, C2."
        response = self.llm.generate_json(prompt, repair)
        labels = {f"C{i}": chunk for i, chunk in enumerate(results, 1)}
        used = [labels[label] for label in response.get("used_chunks", []) if label in labels]
        if response.get("found") and not used:
            response["found"] = False

        if not response.get("found"):
            answer = abstain_message(understanding.get("language", "en"))
            return {
                "answer": answer, "found": False, "sources": [], "images": [],
                "chunks": results, "understanding": understanding, "best_similarity": best_dense,
            }

        answer = str(response.get("answer", "")).strip()
        if response.get("needs_date_note"):
            year = used[0].get("manual_year", "unknown") if used else "unknown"
            if year != "unknown":
                note = f"Manual date: {year}; procedures may have changed."
                if understanding.get("language") == "ta":
                    note = f"இந்த கையேடு {year} காலத்தைச் சேர்ந்தது; நடைமுறைகள் மாறியிருக்கலாம்."
                answer += "\n\n" + note

        sources = []
        for chunk in used:
            source = {"manual": chunk["manual_file"], "page": chunk["page"], "chunk_id": chunk["chunk_id"]}
            if source not in sources:
                sources.append(source)

        images = []
        for chunk in sorted(used, key=lambda x: x["fusion_rank"]):
            if chunk.get("image_path") and chunk["image_path"] not in [x["path"] for x in images]:
                images.append({
                    "path": chunk["image_path"],
                    "caption": f"{chunk['service_code']} manual, page {chunk['page']}",
                    "chunk_id": chunk["chunk_id"],
                })
            if len(images) >= int(self.config.raw["ui"]["max_images"]):
                break

        return {
            "answer": answer,
            "found": True,
            "sources": sources,
            "images": images,
            "chunks": results,
            "understanding": understanding,
            "best_similarity": best_dense,
            "conflict_noted": bool(response.get("conflict_noted")),
        }
