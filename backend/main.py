from __future__ import annotations

import base64
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.db.database import Database
from backend.models.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationCreate,
    ConversationResponse,
    DocumentUploadResponse,
    HealthResponse,
    MessageResponse,
    UserCreate,
    UserResponse,
)
from backend.rag.engine import AdvancedRAGEngine

db = Database(settings.db_path)
rag_engine: AdvancedRAGEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine
    settings.data_path.mkdir(parents=True, exist_ok=True)
    settings.storage_path.mkdir(parents=True, exist_ok=True)
    settings.documents_path.mkdir(parents=True, exist_ok=True)
    settings.uploads_path.mkdir(parents=True, exist_ok=True)
    rag_engine = AdvancedRAGEngine(db)
    yield


app = FastAPI(
    title="RAG Chatbot API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model=settings.groq_model,
        ollama_url="",
    )


@app.post("/users", response_model=UserResponse)
def create_user(payload: UserCreate):
    existing = db.get_user_by_username(payload.username)
    if existing:
        return UserResponse(
            user_id=existing["user_id"],
            username=existing["username"],
            created_at=datetime.fromisoformat(existing["created_at"]),
        )
    user = db.create_user(payload.username)
    return UserResponse(
        user_id=user["user_id"],
        username=user["username"],
        created_at=datetime.fromisoformat(user["created_at"]),
    )


@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: str):
    user = db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(
        user_id=user["user_id"],
        username=user["username"],
        created_at=datetime.fromisoformat(user["created_at"]),
    )


@app.post("/conversations", response_model=ConversationResponse)
def create_conversation(payload: ConversationCreate):
    if not db.get_user(payload.user_id):
        raise HTTPException(status_code=404, detail="User not found")
    conv = db.create_conversation(payload.user_id, payload.title)
    return ConversationResponse(
        conversation_id=conv["conversation_id"],
        user_id=conv["user_id"],
        title=conv["title"],
        created_at=datetime.fromisoformat(conv["created_at"]),
        updated_at=datetime.fromisoformat(conv["updated_at"]),
    )


@app.get("/conversations/{user_id}", response_model=list[ConversationResponse])
def list_conversations(user_id: str):
    convs = db.list_conversations(user_id)
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


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    conv = db.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete_conversation(conversation_id)
    return {"status": "deleted"}


@app.get("/messages/{conversation_id}", response_model=list[MessageResponse])
def get_messages(conversation_id: str):
    conv = db.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    msgs = db.get_messages(conversation_id)
    return [
        MessageResponse(
            message_id=m["message_id"],
            conversation_id=m["conversation_id"],
            role=m["role"],
            content=m["content"],
            image_path=m.get("image_path"),
            created_at=datetime.fromisoformat(m["created_at"]),
        )
        for m in msgs
    ]


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not ready")

    conv = db.get_conversation(payload.conversation_id)
    if not conv or conv["user_id"] != payload.user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    image_path = None
    if payload.image_base64:
        img_data = base64.b64decode(
            payload.image_base64.split(",")[-1] if "," in payload.image_base64 else payload.image_base64
        )
        image_path = str(
            settings.uploads_path / f"{uuid.uuid4().hex}_chat_image.png"
        )
        with open(image_path, "wb") as f:
            f.write(img_data)

    db.add_message(
        payload.conversation_id,
        "user",
        payload.message,
        image_path,
    )

    msgs = db.get_messages(payload.conversation_id)
    if len(msgs) == 1:
        title = payload.message[:48] + ("..." if len(payload.message) > 48 else "")
        db.update_conversation_title(payload.conversation_id, title)

    result = rag_engine.chat(
        user_id=payload.user_id,
        conversation_id=payload.conversation_id,
        message=payload.message,
        image_base64=payload.image_base64,
        use_web_search=payload.use_web_search,
    )

    saved = db.add_message(
        payload.conversation_id,
        "assistant",
        result["reply"],
    )

    return ChatResponse(
        reply=result["reply"],
        sources=result.get("sources", []),
        web_sources=result.get("web_sources", []),
        used_web_search=result.get("used_web_search", False),
        message_id=saved["message_id"],
    )


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str | None = None,
):
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not ready")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        chunks = rag_engine.ingestion.ingest_bytes(
            data, file.filename or "upload", user_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DocumentUploadResponse(
        filename=file.filename or "upload",
        chunks_indexed=chunks,
        message=f"Indexed {chunks} chunks from {file.filename}",
    )


@app.get("/memory/{user_id}")
def get_user_memory(user_id: str):
    if not db.get_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    memories = db.get_user_memories(user_id)
    return {"user_id": user_id, "memories": memories}
