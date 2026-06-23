"""
Quality & Regulatory Compliance Intelligence agent.

Checks a document's content against a configurable list of regulatory
requirements and flags whether each is addressed, partially addressed, or
missing - with a short justification. The checklist below is illustrative
(swap in real OISD/Factory Act/DGMS clause text for production use).
"""
import json
from app.agents.groq_client import get_groq_client, MODEL
from app.db import sqlite_store

DEFAULT_CHECKLIST = [
    "Permit-to-work procedure is documented and references hazard identification",
    "Gas detection / hazardous atmosphere monitoring procedure is specified",
    "Emergency evacuation and incident response steps are defined",
    "Maintenance and inspection schedule is documented with frequency",
    "Personal protective equipment (PPE) requirements are specified",
    "Confined space entry procedure is documented (if applicable)",
]

COMPLIANCE_PROMPT = """You are a regulatory compliance auditor for Indian industrial
operations (OISD / Factory Act / DGMS style standards). Given the document excerpt
below and a checklist of requirements, assess each requirement.

Return ONLY a JSON array, no preamble, no markdown fences. Each item:
{{"regulation": "<the checklist item>", "status": "<met|partial|missing>", "note": "<one sentence justification>"}}

CHECKLIST:
{checklist}

DOCUMENT EXCERPT:
{text}
"""


def run_compliance_check(doc_id: int, checklist: list[str] = None) -> list[dict]:
    checklist = checklist or DEFAULT_CHECKLIST
    chunks = sqlite_store.get_chunks_for_doc(doc_id)
    full_text = " ".join(c["text"] for c in chunks)[:6000]

    client = get_groq_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": COMPLIANCE_PROMPT.format(
                    checklist="\n".join(f"- {c}" for c in checklist), text=full_text
                ),
            }
        ],
        temperature=0,
        max_tokens=1200,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        flags = json.loads(raw)
        if not isinstance(flags, list):
            flags = []
    except json.JSONDecodeError:
        flags = []

    sqlite_store.insert_compliance_flags(doc_id, flags)
    return flags
