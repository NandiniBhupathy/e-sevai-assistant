"""Streamlit UI for e-Sevai Saathi."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import Pipeline
from llm import LLMError

st.set_page_config(page_title="e-Sevai Saathi", page_icon="🧾", layout="wide")
st.title("e-Sevai Saathi")
st.caption("Tamil-English assistant for official Tamil Nadu e-Sevai training manuals")

@st.cache_resource
def get_pipeline():
    return Pipeline(str(ROOT / "config.yaml"))

try:
    pipeline = get_pipeline()
except FileNotFoundError:
    st.error("The search index is missing. Put PDFs in data/manuals and run scripts/build_index.py.")
    st.stop()
except LLMError as exc:
    st.warning(str(exc))
    pipeline = None
except Exception as exc:
    st.error(f"Could not initialize the application: {exc}")
    st.stop()

with st.sidebar:
    st.header("Query settings")
    user_type_label = st.selectbox("User type", ["Auto", "Citizen", "Operator"])
    service_codes = sorted(pipeline.services) if pipeline else []
    service_label = st.selectbox("Service (optional)", ["Auto"] + service_codes)
    show_debug = st.checkbox("Show retrieved chunks for debugging", value=False)
    if pipeline is not None and not pipeline.config.gemini_api_key:
        st.warning(
            "GEMINI_API_KEY is not set. Put it in a .env file next to config.yaml, "
            "or set it in the environment. Answers will fail until then."
        )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Sources: " + "; ".join(f"{s['manual']} p. {s['page']}" for s in message["sources"]))
        for image in message.get("images", []):
            image_path = ROOT / image["path"]
            if image_path.exists():
                st.image(str(image_path), caption=image["caption"])

question = st.chat_input("Ask in Tamil, English, or Tanglish…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the manuals…"):
            try:
                result = pipeline.answer(
                    question,
                    user_type=user_type_label.lower(),
                    service_code=None if service_label == "Auto" else service_label,
                )
            except LLMError as exc:
                st.error(str(exc))
                result = None
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                result = None

        if result:
            st.markdown(result["answer"])
            if result["sources"]:
                st.caption("Sources: " + "; ".join(f"{s['manual']} p. {s['page']}" for s in result["sources"]))
            for image in result["images"]:
                image_path = ROOT / image["path"]
                if image_path.exists():
                    st.image(str(image_path), caption=image["caption"])
            if show_debug:
                with st.expander("Retrieved chunks"):
                    for chunk in result["chunks"]:
                        st.markdown(
                            f"**{chunk['chunk_id']}** — {chunk['service_code']} p.{chunk['page']} "
                            f"({chunk['chunk_type']}, fusion={chunk.get('fusion_score', 0):.4f})"
                        )
                        st.write(chunk["text"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "images": result["images"],
            })
