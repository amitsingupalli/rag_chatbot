"""Streamlit frontend — Claude-inspired chat UI."""

from __future__ import annotations

import base64
import os
from io import BytesIO

import httpx
import streamlit as st
from PIL import Image

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Claude-inspired CSS ───────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 1rem;
    padding-bottom: 6rem;
    max-width: 820px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #1a1a1a;
    border-right: 1px solid #2e2e2e;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    color: #e8e8e8 !important;
}

/* Chat bubbles */
.user-bubble {
    background: #2d2d2d;
    border-radius: 18px 18px 4px 18px;
    padding: 14px 18px;
    margin: 8px 0 8px 60px;
    color: #f0f0f0;
    font-size: 15px;
    line-height: 1.6;
}
.assistant-bubble {
    background: transparent;
    border-radius: 4px 18px 18px 18px;
    padding: 14px 18px;
    margin: 8px 60px 8px 0;
    color: #e8e8e8;
    font-size: 15px;
    line-height: 1.6;
    border-left: 2px solid #d97757;
    padding-left: 16px;
}
.role-label {
    font-size: 12px;
    font-weight: 600;
    color: #888;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.assistant-label { color: #d97757; }

/* Input area */
.stChatInput > div {
    border-radius: 24px !important;
    border: 1px solid #3a3a3a !important;
    background: #1e1e1e !important;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.15s;
}
.stButton > button[kind="primary"] {
    background: #d97757;
    border: none;
    color: white;
}
.stButton > button[kind="primary"]:hover {
    background: #c4664a;
}

/* Conversation list item */
.conv-item {
    padding: 8px 12px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 13px;
    color: #ccc;
    margin-bottom: 2px;
}
.conv-item:hover { background: #2a2a2a; }
.conv-active { background: #333 !important; color: #fff !important; }

/* Sources badge */
.source-badge {
    display: inline-block;
    background: #2a2a2a;
    border: 1px solid #444;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 11px;
    color: #aaa;
    margin: 2px;
}
.web-source-badge {
    display: inline-block;
    background: #1a2a1a;
    border: 1px solid #3a5a3a;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 11px;
    color: #8bc48b;
    margin: 2px;
}

/* Hide streamlit branding */
.stDeployButton { display: none; }
</style>
""",
    unsafe_allow_html=True,
)


def api_get(path: str):
    with httpx.Client(timeout=120.0) as client:
        r = client.get(f"{BACKEND_URL}{path}")
        r.raise_for_status()
        return r.json()


def api_post(path: str, payload: dict):
    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{BACKEND_URL}{path}", json=payload)
        r.raise_for_status()
        return r.json()


def api_delete(path: str):
    with httpx.Client(timeout=30.0) as client:
        r = client.delete(f"{BACKEND_URL}{path}")
        r.raise_for_status()
        return r.json()


def api_upload(file_bytes: bytes, filename: str, user_id: str):
    with httpx.Client(timeout=120.0) as client:
        r = client.post(
            f"{BACKEND_URL}/documents/upload",
            files={"file": (filename, file_bytes)},
            params={"user_id": user_id},
        )
        r.raise_for_status()
        return r.json()


def image_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def init_state():
    defaults = {
        "user_id": None,
        "username": None,
        "conversation_id": None,
        "messages": [],
        "conversations": [],
        "pending_image": None,
        "pending_image_name": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


with st.sidebar:
    st.markdown("### ✦ RAG Chatbot")
    st.markdown(
        "<p style='color:#888;font-size:12px;margin-top:-10px'>Advanced Multimodal RAG</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # User login / register
    username_input = st.text_input(
        "Your name",
        value=st.session_state.username or "",
        placeholder="Enter username to start",
        key="username_input",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sign In", use_container_width=True, type="primary"):
            if username_input.strip():
                try:
                    user = api_post("/users", {"username": username_input.strip()})
                    st.session_state.user_id = user["user_id"]
                    st.session_state.username = user["username"]
                    convs = api_get(f"/conversations/{user['user_id']}")
                    st.session_state.conversations = convs
                    if convs:
                        st.session_state.conversation_id = convs[0]["conversation_id"]
                        msgs = api_get(f"/messages/{convs[0]['conversation_id']}")
                        st.session_state.messages = msgs
                    else:
                        new_conv = api_post(
                            "/conversations",
                            {"user_id": user["user_id"], "title": "New Chat"},
                        )
                        st.session_state.conversation_id = new_conv["conversation_id"]
                        st.session_state.messages = []
                        st.session_state.conversations = [new_conv]
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not connect to backend: {exc}")
            else:
                st.warning("Please enter a username.")

    with col2:
        if st.button("New Chat", use_container_width=True):
            if st.session_state.user_id:
                try:
                    new_conv = api_post(
                        "/conversations",
                        {"user_id": st.session_state.user_id, "title": "New Chat"},
                    )
                    st.session_state.conversation_id = new_conv["conversation_id"]
                    st.session_state.messages = []
                    st.session_state.conversations.insert(0, new_conv)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    if st.session_state.username:
        st.markdown(
            f"<p style='color:#888;font-size:12px'>Signed in as "
            f"<b style='color:#d97757'>{st.session_state.username}</b></p>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**Conversations**")

    if st.session_state.conversations:
        for conv in st.session_state.conversations:
            is_active = conv["conversation_id"] == st.session_state.conversation_id
            label = conv["title"][:40] + ("…" if len(conv["title"]) > 40 else "")
            if st.button(
                label,
                key=f"conv_{conv['conversation_id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.conversation_id = conv["conversation_id"]
                try:
                    msgs = api_get(f"/messages/{conv['conversation_id']}")
                    st.session_state.messages = msgs
                except Exception:
                    st.session_state.messages = []
                st.rerun()

    st.divider()
    st.markdown("**Web Search**")
    st.session_state.setdefault("use_web_search", True)
    use_web = st.toggle(
        "Auto-search for current affairs",
        value=st.session_state.use_web_search,
        help="Automatically browses the internet for news, latest events, and time-sensitive questions.",
    )
    st.session_state.use_web_search = use_web

    st.divider()
    st.markdown("**Upload Documents**")
    uploaded_doc = st.file_uploader(
        "PDF, TXT, MD, Images",
        type=["pdf", "txt", "md", "csv", "png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
        key="doc_uploader",
    )
    if uploaded_doc and st.session_state.user_id:
        if st.button("Index Document", use_container_width=True):
            with st.spinner("Indexing…"):
                try:
                    result = api_upload(
                        uploaded_doc.read(),
                        uploaded_doc.name,
                        st.session_state.user_id,
                    )
                    st.success(result["message"])
                except Exception as exc:
                    st.error(str(exc))

    st.divider()
    st.markdown(
        "<p style='color:#555;font-size:11px'>Powered by LlamaIndex + Groq</p>",
        unsafe_allow_html=True,
    )


if not st.session_state.user_id:
    st.markdown(
        """
        <div style="text-align:center;padding:80px 20px">
            <h1 style="color:#d97757;font-size:2.5rem;font-weight:600">✦ RAG Chatbot</h1>
            <p style="color:#888;font-size:16px;margin-top:12px">
                Advanced Multimodal RAG with persistent memory<br>
                Enter your username in the sidebar to begin
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Render message history
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    if role == "user":
        st.markdown(
            f'<div class="role-label">You</div>'
            f'<div class="user-bubble">{content}</div>',
            unsafe_allow_html=True,
        )
        if msg.get("image_path"):
            try:
                st.image(msg["image_path"], width=300)
            except Exception:
                pass
    else:
        st.markdown(
            f'<div class="role-label assistant-label">Assistant</div>'
            f'<div class="assistant-bubble">{content}</div>',
            unsafe_allow_html=True,
        )

# Image attachment for current message
col_img, col_clear = st.columns([4, 1])
with col_img:
    chat_image = st.file_uploader(
        "Attach image (OCR enabled)",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
        key="chat_image_uploader",
    )
with col_clear:
    if chat_image:
        st.image(chat_image, width=80)

if chat_image:
    st.session_state.pending_image = Image.open(chat_image).convert("RGB")
    st.session_state.pending_image_name = chat_image.name

# Chat input
user_input = st.chat_input("Message RAG Chatbot…")

if user_input and st.session_state.conversation_id:
    image_b64 = None
    if st.session_state.pending_image:
        image_b64 = image_to_base64(st.session_state.pending_image)

    # Show user message immediately
    st.markdown(
        f'<div class="role-label">You</div>'
        f'<div class="user-bubble">{user_input}</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.pending_image:
        st.image(st.session_state.pending_image, width=300)

    with st.spinner("Searching the web & thinking…" if st.session_state.get("use_web_search") else "Thinking…"):
        try:
            response = api_post(
                "/chat",
                {
                    "user_id": st.session_state.user_id,
                    "conversation_id": st.session_state.conversation_id,
                    "message": user_input,
                    "image_base64": image_b64,
                    "use_web_search": None if st.session_state.get("use_web_search") else False,
                },
            )
            reply = response["reply"]
            sources = response.get("sources", [])
            web_sources = response.get("web_sources", [])

            st.markdown(
                f'<div class="role-label assistant-label">Assistant</div>'
                f'<div class="assistant-bubble">{reply}</div>',
                unsafe_allow_html=True,
            )
            if sources:
                badges = " ".join(
                    f'<span class="source-badge">{s}</span>' for s in sources[:5]
                )
                st.markdown(
                    f"<div style='margin:4px 0 4px 0'><small style='color:#888'>Documents:</small> {badges}</div>",
                    unsafe_allow_html=True,
                )
            if web_sources:
                web_badges = " ".join(
                    f'<span class="web-source-badge">{s}</span>' for s in web_sources[:5]
                )
                st.markdown(
                    f"<div style='margin:4px 0 16px 0'><small style='color:#8bc48b'>Web:</small> {web_badges}</div>",
                    unsafe_allow_html=True,
                )

            # Refresh state from backend
            msgs = api_get(f"/messages/{st.session_state.conversation_id}")
            st.session_state.messages = msgs
            convs = api_get(f"/conversations/{st.session_state.user_id}")
            st.session_state.conversations = convs
            st.session_state.pending_image = None
            st.session_state.pending_image_name = None
            st.rerun()

        except httpx.ConnectError:
            st.error("Cannot reach backend. Make sure FastAPI is running on port 8000.")
        except Exception as exc:
            st.error(f"Error: {exc}")
