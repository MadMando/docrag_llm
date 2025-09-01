# server.py
"""
DocRAG LLM — FastAPI server (all-in-one container edition).
Endpoints:
  - GET  /api/health
  - GET  /api/models
  - POST /api/ingest
  - POST /api/ask
Environment:
  DOCRAG_LLM (default: llama3.2:1b)
  DOCRAG_EMBED (default: nomic-embed-text)
  OLLAMA_HOST (default: 0.0.0.0:11434 set by Dockerfile)
"""

import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from docrag import DocragSettings, RAGPipeline
except ImportError:
    raise SystemExit("docrag-llm must be installed: pip install docrag-llm")

app = FastAPI(title="DocRAG LLM API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def pipeline_for(persist: str, collection: str) -> RAGPipeline:
    cfg = DocragSettings(
        persist_path=persist,
        collection=collection,
        embed_model=os.getenv("DOCRAG_EMBED", "nomic-embed-text"),
        llm_model=os.getenv("DOCRAG_LLM", "llama3.2:1b"),
    )
    return RAGPipeline(cfg)

class IngestBody(BaseModel):
    uri_or_path: str
    collection: str = Field(default="demo")
    persist: str = Field(default="./.chroma")

class AskBody(BaseModel):
    question: str
    collection: str = Field(default="demo")
    persist: str = Field(default="./.chroma")
    top_k: int = Field(default=5, ge=1, le=50)
    require_citations: bool = Field(default=False)

@app.get("/api/health")
def health():
    return {"ok": True, "service": "docrag-llm"}

@app.get("/api/models")
def list_models():
    try:
        import ollama  # type: ignore
        data = ollama.list()
        names = [m.get("name") for m in data.get("models", []) if m.get("name")]
        return {"ok": True, "models": names}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e), "models": []}, status_code=200)

@app.post("/api/ingest")
def ingest(body: IngestBody):
    pl = pipeline_for(body.persist, body.collection)
    n = pl.ingest(body.uri_or_path)
    return {"ok": True, "chunks": n, "collection": body.collection, "persist": body.persist}

@app.post("/api/ask")
def ask(body: AskBody):
    pl = pipeline_for(body.persist, body.collection)
    answer = pl.ask(body.question, top_k=body.top_k, require_citations=body.require_citations)
    return {"ok": True, "answer": answer}
