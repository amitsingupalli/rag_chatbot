from __future__ import annotations

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status

from backend.auth import (
    create_access_token,
    get_current_user_token,
    hash_password,
    verify_password,
)
from backend.db.database import Database
from backend.models.schemas import TokenResponse, UserLogin, UserRegister, UserResponse

router = APIRouter(prefix="/auth", tags=["Auth"])


def get_db():
    from backend.main import db
    return db


def get_current_user(payload: dict = Depends(get_current_user_token)) -> dict:
    db = get_db()
    user_id = payload.get("sub") or payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    user = db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


@router.post("/register", response_model=TokenResponse)
def register(payload: UserRegister):
    db = get_db()
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


@router.post("/login", response_model=TokenResponse)
def login(payload: UserLogin):
    db = get_db()
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


@router.get("/me", response_model=UserResponse)
def get_me(current_user: dict = Depends(get_current_user)):
    return UserResponse(
        user_id=current_user["user_id"],
        username=current_user["username"],
        created_at=datetime.fromisoformat(current_user["created_at"]),
    )
