from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.db.database import Database
from backend.models.schemas import HealthResponse
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
    title="Advanced RAG Chatbot API (Modular & Authenticated)",
    version="2.1.0",
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


# ── Register Routers ──────────────────────────────────────────────────────────
from backend.routers.auth import router as auth_router
from backend.routers.conversations import router as conversations_router
from backend.routers.ingest import router as ingest_router
from backend.routers.query import router as query_router

app.include_router(auth_router)
app.include_router(conversations_router)
app.include_router(ingest_router)
app.include_router(query_router)
