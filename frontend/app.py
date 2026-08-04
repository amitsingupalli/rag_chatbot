from __future__ import annotations

import base64
import os
import uuid
from datetime import datetime
from io import BytesIO

import sys
from pathlib import Path

# Add project root directory to sys.path so Streamlit Cloud can find 'backend'
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from PIL import Image

from backend.config import settings
from backend.db.database import Database
from backend.rag.engine import AdvancedRAGEngine

st.set_page_config(
    page_title="Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialize Database & RAG Engine ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_db_and_engine():
    settings.data_path.mkdir(parents=True, exist_ok=True)
    settings.storage_path.mkdir(parents=True, exist_ok=True)
    settings.documents_path.mkdir(parents=True, exist_ok=True)
    settings.uploads_path.mkdir(parents=True, exist_ok=True)
    
    db = Database(settings.db_path)
    engine = AdvancedRAGEngine(db)
    return db, engine

try:
    db, rag_engine = get_db_and_engine()
except Exception as e:
    st.error(f"Error initializing RAG Engine: {e}")
    db, rag_engine = None, None

# ── Styling (Claude Theme & Categorized Popover Attachment) ───────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #18181b !important;
    color: #e4e4e7 !important;
}

#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }

/* Keep sidebar toggle controls visible at all times */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[aria-label="Expand sidebar"],
button[aria-label="Collapse sidebar"] {
    visibility: visible !important;
    display: flex !important;
    color: #d97757 !important;
    background: #27272a !important;
    border: 1px solid #3f3f46 !important;
    border-radius: 8px !important;
    z-index: 999999 !important;
}

[data-testid="stSidebarCollapsedControl"]:hover,
[data-testid="stSidebarCollapseButton"]:hover {
    background: #3f3f46 !important;
    color: #ffffff !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 7rem;
    max-width: 840px;
}

/* Sidebar Styling (Claude Minimal) */
section[data-testid="stSidebar"] {
    background: #1f1f23 !important;
    border-right: 1px solid #27272a !important;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #d4d4d8 !important;
}

/* Hide file uploader instructions (200MB per file...) */
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploadDropzone"] small,
.stFileUploader small {
    display: none !important;
}

/* Chat bubbles */
.user-bubble {
    background: #27272a;
    border: 1px solid #3f3f46;
    border-radius: 16px 16px 4px 16px;
    padding: 14px 18px;
    margin: 8px 0 12px 48px;
    color: #f4f4f5;
    font-size: 15px;
    line-height: 1.65;
}

.assistant-bubble {
    background: transparent;
    border-radius: 4px 16px 16px 16px;
    padding: 14px 18px 14px 0px;
    margin: 8px 48px 12px 0;
    color: #e4e4e7;
    font-size: 15px;
    line-height: 1.65;
}

.role-label {
    font-size: 12px;
    font-weight: 600;
    color: #a1a1aa;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.assistant-label { color: #d97757; }

/* Claude Chat Attachment '+' Popover Button */
div[data-testid="stPopover"] > button {
    border-radius: 50% !important;
    width: 42px !important;
    height: 42px !important;
    padding: 0 !important;
    margin-top: 4px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: #27272a !important;
    border: 1px solid #3f3f46 !important;
    color: #e4e4e7 !important;
    font-size: 20px !important;
    font-weight: 400 !important;
    transition: all 0.2s !important;
}

div[data-testid="stPopover"] > button:hover {
    background: #3f3f46 !important;
    color: #ffffff !important;
    border-color: #d97757 !important;
}

/* Input area */
.stChatInput > div {
    border-radius: 16px !important;
    border: 1px solid #3f3f46 !important;
    background: #27272a !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
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

/* Sources badge */
.source-badge {
    display: inline-block;
    background: #27272a;
    border: 1px solid #3f3f46;
    border-radius: 8px;
    padding: 3px 10px;
    font-size: 12px;
    color: #a1a1aa;
    margin: 2px;
}
.web-source-badge {
    display: inline-block;
    background: #1c2b1e;
    border: 1px solid #2d4731;
    border-radius: 8px;
    padding: 3px 10px;
    font-size: 12px;
    color: #a3e635;
    margin: 2px;
}

.attached-file-chip {
    display: inline-flex;
    align-items: center;
    background: #27272a;
    border: 1px solid #d97757;
    border-radius: 12px;
    padding: 6px 14px;
    font-size: 13px;
    color: #f4f4f5;
    margin-bottom: 8px;
}

.stDeployButton { display: none; }
</style>
""",
    unsafe_allow_html=True,
)


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
        "pending_doc_name": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    username_input = st.text_input(
        "Your name",
        value=st.session_state.username or "",
        placeholder="Enter username to start",
        key="username_input",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sign In", use_container_width=True, type="primary"):
            if username_input.strip() and db:
                existing = db.get_user_by_username(username_input.strip())
                if not existing:
                    user = db.create_user(username_input.strip())
                else:
                    user = existing
                
                st.session_state.user_id = user["user_id"]
                st.session_state.username = user["username"]
                convs = db.list_conversations(user["user_id"])
                st.session_state.conversations = convs
                if convs:
                    st.session_state.conversation_id = convs[0]["conversation_id"]
                    st.session_state.messages = db.get_messages(convs[0]["conversation_id"])
                else:
                    new_conv = db.create_conversation(user["user_id"], "New Chat")
                    st.session_state.conversation_id = new_conv["conversation_id"]
                    st.session_state.messages = []
                    st.session_state.conversations = [new_conv]
                st.rerun()
            else:
                st.warning("Please enter a username.")

    with col2:
        if st.button("New Chat", use_container_width=True):
            if st.session_state.user_id and db:
                new_conv = db.create_conversation(st.session_state.user_id, "New Chat")
                st.session_state.conversation_id = new_conv["conversation_id"]
                st.session_state.messages = []
                st.session_state.conversations.insert(0, new_conv)
                st.rerun()

    if st.session_state.username:
        st.markdown(
            f"<p style='color:#a1a1aa;font-size:12px;margin-top:6px'>Signed in as "
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
                if db:
                    st.session_state.messages = db.get_messages(conv["conversation_id"])
                st.rerun()

    st.divider()
    st.markdown("**Web Search**")
    st.session_state.setdefault("use_web_search", True)
    use_web = st.toggle(
        "Auto-search for current affairs",
        value=st.session_state.use_web_search,
        help="Automatically browses the internet for news and time-sensitive questions.",
    )
    st.session_state.use_web_search = use_web

    st.divider()
    st.markdown("**Upload Documents**")
    uploaded_doc = st.file_uploader(
        "PDF, CSV, PPT, PPTX, TXT, MD, Images",
        type=["pdf", "csv", "ppt", "pptx", "txt", "md", "json", "tsv", "png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
        key="doc_uploader",
    )
    if uploaded_doc and st.session_state.user_id and rag_engine:
        file_key = f"{uploaded_doc.name}_{uploaded_doc.size}"
        if st.session_state.get("last_indexed_file") != file_key:
            with st.spinner(f"Indexing {uploaded_doc.name}…"):
                try:
                    data = uploaded_doc.read()
                    chunks = rag_engine.ingestion.ingest_bytes(
                        data, uploaded_doc.name, st.session_state.user_id
                    )
                    st.session_state["last_indexed_file"] = file_key
                    st.success(f"Indexed {chunks} chunks from {uploaded_doc.name}")
                except Exception as exc:
                    st.error(f"Error indexing {uploaded_doc.name}: {exc}")
        else:
            st.success(f"Indexed & ready: {uploaded_doc.name}")


# ── Main chat area ────────────────────────────────────────────────────────────
if not st.session_state.user_id:
    st.markdown(
        """
        <div style="text-align:center;padding:100px 20px">
            <h1 style="color:#d97757;font-size:2.8rem;font-weight:600;letter-spacing:-0.02em">✦ Assistant</h1>
            <p style="color:#a1a1aa;font-size:16px;margin-top:14px">
                Upload documents (PDF, PPT, CSV, Images) or chat directly.<br>
                Please enter your username in the sidebar to start.
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

# Display attached file chip if present
if st.session_state.get("pending_doc_name"):
    st.markdown(
        f'<div class="attached-file-chip">📄 {st.session_state.pending_doc_name} &nbsp; <small style="color:#a3e635">(Indexed & Ready)</small></div>',
        unsafe_allow_html=True,
    )
elif st.session_state.get("pending_image_name"):
    st.markdown(
        f'<div class="attached-file-chip">🖼️ {st.session_state.pending_image_name} &nbsp; <small style="color:#a3e635">(Image Ready)</small></div>',
        unsafe_allow_html=True,
    )

# Claude-Style Bottom Attachment Bar with 2 Categorized Sections
attach_col, input_col = st.columns([1, 12])

with attach_col:
    with st.popover("+", help="Attach document or image"):
        st.markdown("<p style='font-size:13px;font-weight:600;color:#e4e4e7;margin-bottom:6px'>Add Attachment</p>", unsafe_allow_html=True)
        tab_doc, tab_img = st.tabs(["📄 Document", "🖼️ Image"])

        with tab_doc:
            doc_file = st.file_uploader(
                "Upload Document",
                type=["pdf", "csv", "ppt", "pptx", "txt", "md", "json", "tsv"],
                label_visibility="collapsed",
                key="chat_doc_uploader_segmented",
            )
            if doc_file and rag_engine and st.session_state.user_id:
                file_key = f"doc_{doc_file.name}_{doc_file.size}"
                if st.session_state.get("last_indexed_chat_file") != file_key:
                    with st.spinner(f"Indexing {doc_file.name}…"):
                        try:
                            data = doc_file.read()
                            chunks = rag_engine.ingestion.ingest_bytes(
                                data, doc_file.name, st.session_state.user_id
                            )
                            st.session_state["last_indexed_chat_file"] = file_key
                            st.session_state.pending_doc_name = doc_file.name
                            st.caption(f"✓ Indexed {chunks} chunks from {doc_file.name}")
                        except Exception as exc:
                            st.error(f"Error indexing {doc_file.name}: {exc}")
                else:
                    st.caption(f"✓ {doc_file.name} ready")

        with tab_img:
            img_file = st.file_uploader(
                "Upload Image",
                type=["png", "jpg", "jpeg", "webp"],
                label_visibility="collapsed",
                key="chat_img_uploader_segmented",
            )
            if img_file:
                st.session_state.pending_image = Image.open(img_file).convert("RGB")
                st.session_state.pending_image_name = img_file.name
                st.caption(f"✓ Image ready: {img_file.name}")

with input_col:
    user_input = st.chat_input("Write a message…")

if user_input and st.session_state.conversation_id and rag_engine and db:
    image_b64 = None
    image_path = None
    if st.session_state.pending_image:
        image_b64 = image_to_base64(st.session_state.pending_image)
        img_data = base64.b64decode(image_b64)
        image_path = str(settings.uploads_path / f"{uuid.uuid4().hex}_chat_image.png")
        with open(image_path, "wb") as f:
            f.write(img_data)

    db.add_message(
        st.session_state.conversation_id,
        "user",
        user_input,
        image_path,
    )

    msgs = db.get_messages(st.session_state.conversation_id)
    if len(msgs) == 1:
        title = user_input[:48] + ("..." if len(user_input) > 48 else "")
        db.update_conversation_title(st.session_state.conversation_id, title)

    st.markdown(
        f'<div class="role-label">You</div>'
        f'<div class="user-bubble">{user_input}</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.pending_image:
        st.image(st.session_state.pending_image, width=300)

    with st.spinner("Thinking…"):
        try:
            result = rag_engine.chat(
                user_id=st.session_state.user_id,
                conversation_id=st.session_state.conversation_id,
                message=user_input,
                image_base64=image_b64,
                use_web_search=st.session_state.get("use_web_search", True),
            )
            reply = result["reply"]
            sources = result.get("sources", [])
            web_sources = result.get("web_sources", [])

            db.add_message(
                st.session_state.conversation_id,
                "assistant",
                reply,
            )

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
                    f"<div style='margin:4px 0 16px 0'><small style='color:#888'>Documents:</small> {badges}</div>",
                    unsafe_allow_html=True,
                )
            if web_sources:
                w_badges = " ".join(
                    f'<span class="web-source-badge">{s}</span>' for s in web_sources[:5]
                )
                st.markdown(
                    f"<div style='margin:4px 0 16px 0'><small style='color:#888'>Web Sources:</small> {w_badges}</div>",
                    unsafe_allow_html=True,
                )

            st.session_state.messages = db.get_messages(st.session_state.conversation_id)
            st.session_state.conversations = db.list_conversations(st.session_state.user_id)
            st.session_state.pending_image = None
            st.session_state.pending_image_name = None
            st.session_state.pending_doc_name = None
            st.rerun()

        except Exception as exc:
            st.error(f"Error: {exc}")
