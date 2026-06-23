"""
Expert Knowledge Copilot - RAG-powered Q&A over the ingested document corpus.

Flow: embed query -> retrieve top-k chunks from vector store -> build a
grounded prompt with source attribution -> Groq generates an answer that
cites which document/chunk it pulled from.
"""
from app.rag.embeddings import embed_query
from app.db import vector_store, sqlite_store
from app.agents.groq_client import get_groq_client, MODEL

ANSWER_PROMPT = """You are an industrial operations knowledge copilot. Answer the
question using ONLY the context provided below. If the context doesn't contain
the answer, say so clearly - do not make things up.

For every claim, cite the source using the format [Source: <filename>].

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (with inline [Source: filename] citations):
"""


def ask(question: str, top_k: int = 5) -> dict:
    query_vec = embed_query(question)
    hits = vector_store.search(query_vec, top_k=top_k)

    if not hits:
        return {
            "answer": "No documents have been ingested yet, so I have nothing to ground an answer on. Upload some documents first.",
            "sources": [],
        }

    context_blocks = []
    sources = []
    for chunk_id, score in hits:
        row = sqlite_store.get_chunk_by_id(chunk_id)
        if row is None:
            continue
        context_blocks.append(f"[Source: {row['filename']}]\n{row['text']}")
        sources.append(
            {"filename": row["filename"], "chunk_id": chunk_id, "relevance": round(score, 3)}
        )

    context = "\n\n---\n\n".join(context_blocks)

    client = get_groq_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": ANSWER_PROMPT.format(context=context, question=question)}
        ],
        temperature=0.2,
        max_tokens=800,
    )

    return {
        "answer": response.choices[0].message.content.strip(),
        "sources": sources,
    }
