from __future__ import annotations

import json
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException

from backend.models.schemas import ConversationCreate, ConversationResponse, MessageResponse
from backend.routers.auth import get_current_user

router = APIRouter(prefix="/conversations", tags=["Conversations"])


def get_db():
    from backend.main import db
    return db


@router.get("", response_model=list[ConversationResponse])
def list_user_conversations(current_user: dict = Depends(get_current_user)):
    """List ONLY the current authenticated user's conversations."""
    db = get_db()
    convs = db.list_conversations(current_user["user_id"])
    return [
        ConversationResponse(
            conversation_id=c["conversation_id"],
            user_id=c["user_id"],
            title=c["title"],
            created_at=datetime.fromisoformat(c["created_at"]),
            updated_at=datetime.fromisoformat(c["updated_at"]),
        )
        for c in convs
    ]


@router.post("", response_model=ConversationResponse)
def create_conversation(payload: ConversationCreate, current_user: dict = Depends(get_current_user)):
    """Create a new conversation permanently tagged with current_user['user_id']."""
    db = get_db()
    conv = db.create_conversation(current_user["user_id"], payload.title)
    return ConversationResponse(
        conversation_id=conv["conversation_id"],
        user_id=conv["user_id"],
        title=conv["title"],
        created_at=datetime.fromisoformat(conv["created_at"]),
        updated_at=datetime.fromisoformat(conv["updated_at"]),
    )


@router.get("/{conversation_id}/messages", response_model=list[MessageResponse])
def get_conversation_messages(conversation_id: str, current_user: dict = Depends(get_current_user)):
    """Get messages for a conversation strictly checking ownership."""
    db = get_db()
    conv = db.get_conversation(conversation_id)
    if not conv or conv["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied")

    msgs = db.get_messages(conversation_id)
    result = []
    for m in msgs:
        srcs = None
        wsrcs = None
        if m.get("sources"):
            try:
                srcs = json.loads(m["sources"]) if isinstance(m["sources"], str) else m["sources"]
            except Exception:
                srcs = [str(m["sources"])]
        if m.get("web_sources"):
            try:
                wsrcs = json.loads(m["web_sources"]) if isinstance(m["web_sources"], str) else m["web_sources"]
            except Exception:
                wsrcs = [str(m["web_sources"])]

        result.append(
            MessageResponse(
                message_id=m["message_id"],
                conversation_id=m["conversation_id"],
                role=m["role"],
                content=m["content"],
                image_path=m.get("image_path"),
                sources=srcs,
                web_sources=wsrcs,
                created_at=datetime.fromisoformat(m["created_at"]),
            )
        )
    return result


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a conversation strictly verifying ownership."""
    db = get_db()
    conv = db.get_conversation(conversation_id)
    if not conv or conv["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied")
    db.delete_conversation(conversation_id)
    return {"status": "deleted"}
