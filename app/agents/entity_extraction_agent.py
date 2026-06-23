"""
Universal Document Ingestion & Knowledge Graph Agent (lightweight version).

Pulls structured entities (equipment tags, regulatory references, personnel,
dates, process parameters) out of raw document text using Groq, and returns
them as rows ready for storage. This is what we present as the "knowledge
graph" layer - in a production system this would feed a real graph DB
(Neo4j / Supabase + pgvector + a relationship table); for the hackathon,
the relational `entities` table keyed by doc_id IS the graph (doc -> entity
edges), which is honest and demonstrable.
"""
import json
from app.agents.groq_client import get_groq_client, MODEL

ENTITY_EXTRACTION_PROMPT = """You are an industrial document entity extractor.
Given the document text below, extract entities relevant to industrial operations.

Return ONLY a JSON array, no preamble, no markdown fences. Each item must be:
{{"type": "<equipment_tag|regulation|date|personnel|process_parameter|location>", "value": "<the entity text>", "context": "<short surrounding context, max 15 words>"}}

If nothing relevant is found, return an empty array [].

DOCUMENT TEXT:
{text}
"""


def extract_entities(text: str) -> list[dict]:
    client = get_groq_client()
    # cap input to keep extraction fast/cheap for the demo
    snippet = text[:4000]

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": ENTITY_EXTRACTION_PROMPT.format(text=snippet)}
        ],
        temperature=0,
        max_tokens=1500,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        entities = json.loads(raw)
        if isinstance(entities, list):
            return entities
        return []
    except json.JSONDecodeError:
        return []
