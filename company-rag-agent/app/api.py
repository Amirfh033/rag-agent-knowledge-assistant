"""HTTP API + web chat UI.   Run: uvicorn app.api:app --port 8000"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import config
from .service import AssistantService

app = FastAPI(title="Nimbus Flow Assistant", version="1.0")
_service: AssistantService | None = None
WEB = config.ROOT / "web" / "index.html"


def get_service() -> AssistantService:
    global _service
    if _service is None:
        try:
            _service = AssistantService()
        except FileNotFoundError as e:
            raise HTTPException(503, str(e))
    return _service


class AskIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(None, max_length=64)


@app.get("/")
def index():
    return FileResponse(WEB)


@app.get("/health")
def health():
    try:
        s = get_service()
        return {"status": "ok", "chat_model": s.llm.chat_model, "embed_model": s.llm.embed_model,
                "chunks": len(s.index.chunks), "documents": len(s.index.meta.get("files", {}))}
    except HTTPException as e:
        return {"status": "no_index", "detail": e.detail}


@app.post("/ask")
def ask(body: AskIn):
    session_id, r = get_service().ask(body.question, body.session_id)
    return {"session_id": session_id, **r.to_dict()}


@app.post("/ingest")
def ingest():
    s = get_service()
    return {"status": "reindexed", **s.reindex()}


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    get_service().sessions.pop(session_id, None)
    return {"status": "cleared"}
