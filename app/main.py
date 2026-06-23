from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.db.sqlite_store import init_db, list_documents, get_entities_for_doc
from app.rag.ingest import ingest_document
from app.agents.copilot_agent import ask
from app.agents.compliance_agent import run_compliance_check, DEFAULT_CHECKLIST

app = FastAPI(title="Industrial Knowledge Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()


@app.get("/")
def root():
    return {"status": "ok", "service": "industrial-knowledge-intelligence"}


@app.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        result = ingest_document(file.filename, file_bytes)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    try:
        return ask(req.question, top_k=req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def documents_endpoint():
    return list_documents()


@app.get("/entities/{doc_id}")
def entities_endpoint(doc_id: int):
    return get_entities_for_doc(doc_id)


class ComplianceRequest(BaseModel):
    doc_id: int
    checklist: list[str] | None = None


@app.post("/compliance-check")
def compliance_endpoint(req: ComplianceRequest):
    try:
        flags = run_compliance_check(req.doc_id, req.checklist)
        return {"doc_id": req.doc_id, "flags": flags}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compliance-checklist")
def default_checklist():
    return {"checklist": DEFAULT_CHECKLIST}
