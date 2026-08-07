from __future__ import annotations

import base64
import json
import uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.config import settings
from backend.models.schemas import ChatRequest, ChatResponse
from backend.routers.auth import get_current_user

router = APIRouter(prefix="", tags=["Query & RAG"])


def get_db():
    from backend.main import db
    return db


def get_rag_engine():
    from backend.main import rag_engine
    return rag_engine


@router.post("/conversations/{conversation_id}/messages", response_model=ChatResponse)
def send_message(
    conversation_id: str,
    payload: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """Post message in a thread, execute RAG pipeline, append response."""
    db = get_db()
    rag_engine = get_rag_engine()

    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not ready")

    conv = db.get_conversation(conversation_id)
    if not conv or conv["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied")

    image_path = None
    if payload.image_base64:
        img_data = base64.b64decode(
            payload.image_base64.split(",")[-1] if "," in payload.image_base64 else payload.image_base64
        )
        image_path = str(settings.uploads_path / f"{uuid.uuid4().hex}_chat_image.png")
        with open(image_path, "wb") as f:
            f.write(img_data)

    db.add_message(
        conversation_id,
        "user",
        payload.message,
        image_path,
    )

    msgs = db.get_messages(conversation_id)
    if len(msgs) == 1:
        title = payload.message[:44] + ("..." if len(payload.message) > 44 else "")
        db.update_conversation_title(conversation_id, title)

    result = rag_engine.chat(
        user_id=current_user["user_id"],
        conversation_id=conversation_id,
        message=payload.message,
        image_base64=payload.image_base64,
        use_web_search=payload.use_web_search,
    )

    sources = result.get("sources", [])
    web_sources = result.get("web_sources", [])
    all_sources = sources + web_sources

    saved = db.add_message(
        conversation_id,
        "assistant",
        result["reply"],
        sources=all_sources,
    )

    return ChatResponse(
        reply=result["reply"],
        sources=sources,
        web_sources=web_sources,
        used_web_search=result.get("used_web_search", False),
        message_id=saved["message_id"],
    )


@router.post("/conversations/{conversation_id}/messages/stream")
async def send_message_stream(
    conversation_id: str,
    payload: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """Post message in a thread and stream assistant response token-by-token (SSE / text-event-stream)."""
    db = get_db()
    rag_engine = get_rag_engine()

    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not ready")

    conv = db.get_conversation(conversation_id)
    if not conv or conv["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied")

    image_path = None
    if payload.image_base64:
        img_data = base64.b64decode(
            payload.image_base64.split(",")[-1] if "," in payload.image_base64 else payload.image_base64
        )
        image_path = str(settings.uploads_path / f"{uuid.uuid4().hex}_chat_image.png")
        with open(image_path, "wb") as f:
            f.write(img_data)

    db.add_message(
        conversation_id,
        "user",
        payload.message,
        image_path,
    )

    async def event_generator():
        stream_gen = rag_engine.chat_stream(
            user_id=current_user["user_id"],
            conversation_id=conversation_id,
            message=payload.message,
            image_base64=payload.image_base64,
            use_web_search=payload.use_web_search,
        )
        for delta, sources, web_sources, used_web in stream_gen:
            chunk_data = json.dumps({"delta": delta, "sources": sources, "web_sources": web_sources})
            yield f"data: {chunk_data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
