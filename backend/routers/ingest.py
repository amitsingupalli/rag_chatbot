from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.models.schemas import DocumentUploadResponse
from backend.routers.auth import get_current_user

router = APIRouter(prefix="/documents", tags=["Ingestion"])


def get_rag_engine():
    from backend.main import rag_engine
    return rag_engine


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    conversation_id: str | None = None,
    current_user: dict = Depends(get_current_user),
):
    """Upload document tagged with current_user['user_id'] and optional conversation_id."""
    rag_engine = get_rag_engine()
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not ready")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        chunks = rag_engine.ingestion.ingest_bytes(
            data, file.filename or "upload", current_user["user_id"], conversation_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DocumentUploadResponse(
        filename=file.filename or "upload",
        chunks_indexed=chunks,
        message=f"Indexed {chunks} chunks from {file.filename}",
    )
