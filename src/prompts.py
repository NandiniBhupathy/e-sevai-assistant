"""Prompts used by the two-call query pipeline."""
from __future__ import annotations

import json


def understand_prompt(question: str, user_type: str, services: dict) -> str:
    service_text = "\n".join(f"{code}: {meta['name']}" for code, meta in services.items())
    return f"""You classify questions for e-Sevai Saathi, a Tamil Nadu e-Sevai training-manual assistant.

Question:
{question}

UI user_type:
{user_type}

Known services:
{service_text}

Rules:
- language must be ta for Tamil script, en for primarily English, mixed for Tanglish/code-mixed Tamil-English.
- Identify a service code only when supported by the question or an unambiguous service name.
- english_query should be a concise English semantic representation preserving service and portal terms.
- is_generic=true for questions not tied to one service, such as generic login/navigation.
- needs_service_clarification=true when the question clearly needs a specific service but no service can be identified.
- user_type must be citizen, operator, or auto. If UI value is Citizen/Operator, preserve it as citizen/operator. If Auto, infer from wording; exact menu/button questions are usually operator questions.
Return ONLY valid JSON with:
{{"language":"ta|en|mixed","user_type":"citizen|operator","service_code":"CODE or null","english_query":"...","is_generic":true,"needs_service_clarification":false}}
"""


def answer_prompt(question: str, understanding: dict, chunks: list[dict], glossary: dict) -> str:
    labelled = []
    for n, chunk in enumerate(chunks, 1):
        labelled.append(
            f"C{n} | service={chunk['service_code']} | page={chunk['page']} | "
            f"type={chunk['chunk_type']} | manual={chunk['manual_file']}\n{chunk['text']}"
        )
    return f"""You answer questions using ONLY the retrieved manual chunks below.

Question: {question}
Language: {understanding.get('language')}
User type: {understanding.get('user_type')}
Service: {understanding.get('service_code')}
Glossary:
{json.dumps(glossary, ensure_ascii=False, indent=2)}

Retrieved chunks:
{chr(10).join(labelled)}

Rules:
1. Never add facts not present in chunks.
2. If the chunks do not answer the question, found=false.
3. Reply in the question's language. Tamil should be simple and natural. Mixed should remain mostly Tamil with the user's English portal terms.
4. Citizen answers are short and plain. Give documents, fee, and where/how to apply only when supported.
5. Operator answers should be ordered and include exact English menu/button labels from the chunks, plus a short Tamil explanation if the question is Tamil.
6. Keep glossary portal terms exactly as supplied.
7. If sources conflict, state both values and where each appears. conflict_noted=true.
8. Ignore dummy/sample applicant names, IDs, Aadhaar-like numbers, addresses, phone numbers and certificate values.
9. needs_date_note=true for fee, document or eligibility answers.
10. used_chunks must contain ONLY labels that support the answer.
11. Return ONLY valid JSON:
{{"answer":"...","used_chunks":["C1"],"found":true,"conflict_noted":false,"needs_date_note":false}}
"""


def clarification_message(language: str, services: dict) -> str:
    names = list(services.items())[:8]
    if language == "ta":
        return "இந்த சேவையை தெளிவாகத் தேர்வு செய்யவும்:\n" + "\n".join(f"- {c}: {m['name']}" for c, m in names)
    return "Please specify the service. Likely options include:\n" + "\n".join(f"- {c}: {m['name']}" for c, m in names)


def abstain_message(language: str) -> str:
    if language == "ta":
        return "இந்த தகவல் வழங்கப்பட்ட e-Sevai பயிற்சி கையேட்டில் கிடைக்கவில்லை. அருகிலுள்ள e-Sevai மையத்தில் உறுதிப்படுத்தவும்."
    return "That information is not covered in the provided e-Sevai training manual. Please confirm it at the nearest e-Sevai centre."
