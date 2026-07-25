from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)


class UserResponse(BaseModel):
    user_id: str
    username: str
    created_at: datetime


class ConversationCreate(BaseModel):
    user_id: str
    title: str = "New Chat"


class ConversationResponse(BaseModel):
    conversation_id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    message_id: str
    conversation_id: str
    role: Literal["user", "assistant"]
    content: str
    image_path: str | None = None
    created_at: datetime


class ChatRequest(BaseModel):
    user_id: str
    conversation_id: str
    message: str
    image_base64: str | None = None
    use_web_search: bool | None = None


class ChatResponse(BaseModel):
    reply: str
    sources: list[str] = Field(default_factory=list)
    web_sources: list[str] = Field(default_factory=list)
    used_web_search: bool = False
    message_id: str


class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str


class HealthResponse(BaseModel):
    status: str
    model: str
    ollama_url: str
