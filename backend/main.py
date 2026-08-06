from __future__ import annotations

import base64
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.auth import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
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
    TokenResponse,
    UserCreate,
    UserLogin,
    UserRegister,
    UserResponse,
)
from backend.rag.engine import AdvancedRAGEngine

db = Database(settings.db_path)
rag_engine: AdvancedRAGEngine | None = None
security = HTTPBearer()


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
    title="RAG Chatbot API (Authenticated)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("sub") or payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid auth payload")
    user = db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model=settings.groq_model,
        ollama_url="",
    )


# ── Auth Endpoints ───────────────────────────────────────────────────────────
@app.post("/auth/register", response_model=TokenResponse)
def register(payload: UserRegister):
    existing = db.get_user_by_username(payload.username.strip())
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed = hash_password(payload.password)
    user = db.create_user(payload.username.strip(), hashed_password=hashed)
    token = create_access_token({"sub": user["user_id"], "user_id": user["user_id"], "username": user["username"]})
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user_id=user["user_id"],
        username=user["username"],
    )


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: UserLogin):
    user = db.get_user_by_username(payload.username.strip())
    if not user or not verify_password(payload.password, user.get("hashed_password")):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    token = create_access_token({"sub": user["user_id"], "user_id": user["user_id"], "username": user["username"]})
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user_id=user["user_id"],
        username=user["username"],
    )


@app.get("/auth/me", response_model=UserResponse)
def get_me(current_user: dict = Depends(get_current_user)):
    return UserResponse(
        user_id=current_user["user_id"],
        username=current_user["username"],
        created_at=datetime.fromisoformat(current_user["created_at"]),
    )


# ── Conversation Endpoints (Per-User Filtered) ────────────────────────────────
@app.get("/conversations", response_model=list[ConversationResponse])
def list_user_conversations(current_user: dict = Depends(get_current_user)):
    """List ONLY the current authenticated user's conversations."""
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


@app.post("/conversations", response_model=ConversationResponse)
def create_conversation(payload: ConversationCreate, current_user: dict = Depends(get_current_user)):
    """Create a new conversation permanently tagged with current_user['user_id']."""
    conv = db.create_conversation(current_user["user_id"], payload.title)
    return ConversationResponse(
        conversation_id=conv["conversation_id"],
        user_id=conv["user_id"],
        title=conv["title"],
        created_at=datetime.fromisoformat(conv["created_at"]),
        updated_at=datetime.fromisoformat(conv["updated_at"]),
    )


@app.get("/conversations/{conversation_id}/messages", response_model=list[MessageResponse])
def get_conversation_messages(conversation_id: str, current_user: dict = Depends(get_current_user)):
    """Get messages for a conversation strictly checking ownership."""
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


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a conversation strictly verifying ownership."""
    conv = db.get_conversation(conversation_id)
    if not conv or conv["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied")
    db.delete_conversation(conversation_id)
    return {"status": "deleted"}


@app.post("/conversations/{conversation_id}/messages", response_model=ChatResponse)
def send_message(
    conversation_id: str,
    payload: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """Post message in a thread, execute RAG pipeline, append response."""
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


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    conversation_id: str | None = None,
    current_user: dict = Depends(get_current_user),
):
    """Upload document tagged with current_user['user_id'] and optional conversation_id."""
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
