from __future__ import annotations

import base64
import json
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
    page_title="RAG Chat App",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialize Database & RAG Engine ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_engine_instance():
    settings.data_path.mkdir(parents=True, exist_ok=True)
    settings.storage_path.mkdir(parents=True, exist_ok=True)
    settings.documents_path.mkdir(parents=True, exist_ok=True)
    settings.uploads_path.mkdir(parents=True, exist_ok=True)
    
    db = Database(settings.db_path)
    engine = AdvancedRAGEngine(db)
    return db, engine


def get_db_and_engine():
    try:
        db, engine = _load_engine_instance()
        import inspect
        if "sources" not in inspect.signature(db.add_message).parameters:
            st.cache_resource.clear()
            return _load_engine_instance()
        return db, engine
    except Exception:
        st.cache_resource.clear()
        return _load_engine_instance()


try:
    db, rag_engine = get_db_and_engine()
except Exception as e:
    st.error(f"Error initializing RAG Engine: {e}")
    db, rag_engine = None, None

# ── State Initialization ─────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversations" not in st.session_state:
    st.session_state.conversations = []
if "pending_image" not in st.session_state:
    st.session_state.pending_image = None
if "pending_image_name" not in st.session_state:
    st.session_state.pending_image_name = None
if "pending_doc_name" not in st.session_state:
    st.session_state.pending_doc_name = None
if "selected_citation" not in st.session_state:
    st.session_state.selected_citation = None

# Auto sign-in default user if not logged in
if not st.session_state.user_id and db:
    user = db.get_user_by_username("default_user")
    if not user:
        user = db.create_user("default_user")
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

# ── Theme Dynamic CSS ────────────────────────────────────────────────────────
is_dark = st.session_state.theme == "dark"

bg_canvas = "#1F1E1B" if is_dark else "#F5F3EE"
bg_sidebar = "#181715" if is_dark else "#EDEAE1"
bg_surface = "#2A2925" if is_dark else "#FFFFFF"
text_primary = "#EDEAE1" if is_dark else "#2B2926"
text_muted = "#A19B91" if is_dark else "#7A756C"
border_subtle = "#363430" if is_dark else "#E3DFD5"
accent_clay = "#E08A63" if is_dark else "#D97757"
accent_hover = "#C4633F"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,500;0,600;1,400&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: {bg_canvas} !important;
    color: {text_primary} !important;
}}

#MainMenu, footer {{ visibility: hidden; }}
header {{ background: transparent !important; }}

/* Sidebar Toggle Button */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[aria-label="Expand sidebar"],
button[aria-label="Collapse sidebar"] {{
    visibility: visible !important;
    display: flex !important;
    color: {accent_clay} !important;
    background: {bg_surface} !important;
    border: 1px solid {border_subtle} !important;
    border-radius: 8px !important;
    z-index: 999999 !important;
}}

.block-container {{
    padding-top: 1rem;
    padding-bottom: 7rem;
    max-width: 740px;
}}

/* Serif Headings */
.serif-font {{
    font-family: 'Lora', Georgia, serif;
}}

/* Top Navigation Bar */
.top-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0px 14px 0px;
    border-bottom: 1px solid {border_subtle};
    margin-bottom: 1.5rem;
}}

.top-bar-title {{
    font-family: 'Lora', Georgia, serif;
    font-size: 20px;
    font-weight: 600;
    color: {text_primary};
    letter-spacing: -0.01em;
}}

/* Sidebar Layout */
section[data-testid="stSidebar"] {{
    background: {bg_sidebar} !important;
    border-right: 1px solid {border_subtle} !important;
}}

section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {{
    color: {text_primary} !important;
}}

/* Primary Buttons */
.stButton > button {{
    border-radius: 10px !important;
    border: 1px solid {border_subtle} !important;
    background-color: {bg_surface} !important;
    color: {text_primary} !important;
    transition: all 0.2s ease !important;
}}

.stButton > button:hover {{
    border-color: {accent_clay} !important;
    color: {accent_clay} !important;
}}

/* New Chat Button */
.new-chat-btn button {{
    background-color: {bg_surface} !important;
    color: {accent_clay} !important;
    border: 1.5px solid {accent_clay} !important;
    font-weight: 600 !important;
}}

.new-chat-btn button:hover {{
    background-color: {accent_clay} !important;
    color: #ffffff !important;
}}

/* Message Thread Bubbles */
.user-bubble {{
    background-color: {bg_surface};
    color: {text_primary};
    border: 1px solid {border_subtle};
    padding: 14px 18px;
    border-radius: 18px 18px 4px 18px;
    margin-bottom: 1.2rem;
    margin-left: auto;
    max-width: 82%;
    font-size: 15px;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}}

.assistant-container {{
    margin-bottom: 1.8rem;
}}

.assistant-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}}

.assistant-avatar {{
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: {accent_clay};
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: bold;
}}

.assistant-label {{
    font-size: 13px;
    font-weight: 600;
    color: {text_muted};
}}

.assistant-content {{
    color: {text_primary};
    font-size: 15px;
    line-height: 1.65;
    padding-left: 32px;
}}

/* Citation Strip */
.citation-strip {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding-left: 32px;
    margin-top: 8px;
}}

.citation-chip {{
    background-color: {bg_surface};
    color: {accent_clay};
    border: 1px solid {border_subtle};
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
}}

.citation-chip:hover {{
    background-color: {accent_clay};
    color: #ffffff;
    border-color: {accent_clay};
}}

/* Shimmer Animation */
@keyframes shimmer {{
    0% {{ opacity: 0.4; }}
    50% {{ opacity: 1; }}
    100% {{ opacity: 0.4; }}
}}

.shimmer-text {{
    color: {accent_clay};
    font-size: 13px;
    font-weight: 500;
    animation: shimmer 1.5s infinite ease-in-out;
    padding-left: 32px;
    margin-bottom: 10px;
}}

/* Document Tray Status Dots */
.status-dot {{
    height: 8px;
    width: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}}
.dot-green {{ background-color: #22c55e; }}
.dot-yellow {{ background-color: #eab308; }}

/* Chat Input Bar */
div[data-testid="stChatInput"] {{
    background-color: {bg_surface} !important;
    border: 1.5px solid {border_subtle} !important;
    border-radius: 18px !important;
}}

div[data-testid="stChatInput"]:focus-within {{
    border-color: {accent_clay} !important;
    box-shadow: 0 0 0 2px rgba(217, 119, 87, 0.2) !important;
}}

/* Hide default file uploader instructions */
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderFileData"] small {{
    display: none !important;
}}
</style>
""",
    unsafe_allow_html=True,
)

def image_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ── Sidebar Content ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<p style='font-size:12px;font-weight:600;letter-spacing:0.05em;color:#7A756C;margin-bottom:10px'>WORKSPACE</p>", unsafe_allow_html=True)
    
    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("+ New Chat", use_container_width=True):
        if st.session_state.user_id and db:
            new_conv = db.create_conversation(st.session_state.user_id, "New Chat")
            st.session_state.conversation_id = new_conv["conversation_id"]
            st.session_state.messages = []
            st.session_state.conversations.insert(0, new_conv)
            st.session_state.pending_doc_name = None
            st.session_state.pending_image_name = None
            st.session_state.pending_image = None
            st.session_state.pop("last_indexed_file", None)
            st.session_state.pop("last_indexed_chat_file", None)
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<p style='font-size:12px;font-weight:600;letter-spacing:0.05em;color:#7A756C;margin-top:20px;margin-bottom:8px'>RECENT CHATS</p>", unsafe_allow_html=True)

    if st.session_state.conversations:
        for conv in st.session_state.conversations:
            is_active = conv["conversation_id"] == st.session_state.conversation_id
            label = conv["title"][:32] + ("…" if len(conv["title"]) > 32 else "")
            if st.button(
                label,
                key=f"conv_{conv['conversation_id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.conversation_id = conv["conversation_id"]
                st.session_state.pending_doc_name = None
                st.session_state.pending_image_name = None
                st.session_state.pending_image = None
                st.session_state.pop("last_indexed_file", None)
                st.session_state.pop("last_indexed_chat_file", None)
                if db:
                    st.session_state.messages = db.get_messages(conv["conversation_id"])
                st.rerun()

    st.divider()
    st.markdown("<p style='font-size:12px;font-weight:600;letter-spacing:0.05em;color:#7A756C;margin-bottom:8px'>DOCUMENTS (RAG TRAY)</p>", unsafe_allow_html=True)
    
    # Document Tray listing uploaded files in this session
    uploaded_doc = st.file_uploader(
        "Upload to Knowledge Base",
        type=["pdf", "csv", "ppt", "pptx", "txt", "md", "json", "tsv", "png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
        key="sidebar_doc_uploader",
    )
    if uploaded_doc and st.session_state.user_id and rag_engine:
        file_key = f"{uploaded_doc.name}_{uploaded_doc.size}_{st.session_state.conversation_id}"
        if st.session_state.get("last_indexed_file") != file_key:
            with st.spinner(f"Indexing {uploaded_doc.name}…"):
                try:
                    data = uploaded_doc.read()
                    chunks = rag_engine.ingestion.ingest_bytes(
                        data, uploaded_doc.name, st.session_state.user_id, st.session_state.conversation_id
                    )
                    st.session_state["last_indexed_file"] = file_key
                    st.session_state.pending_doc_name = uploaded_doc.name
                    st.markdown(f"<p style='font-size:13px;color:#22c55e'><span class='status-dot dot-green'></span>{uploaded_doc.name} ({chunks} chunks)</p>", unsafe_allow_html=True)
                except Exception as exc:
                    st.markdown(f"<p style='font-size:13px;color:#ef4444'>🔴 Failed indexing {uploaded_doc.name}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='font-size:13px;'><span class='status-dot dot-green'></span>{uploaded_doc.name} (Ready)</p>", unsafe_allow_html=True)
    elif st.session_state.get("pending_doc_name"):
        st.markdown(f"<p style='font-size:13px;'><span class='status-dot dot-green'></span>{st.session_state.pending_doc_name} (Active)</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='font-size:12px;color:#7A756C;font-style:italic'>No active document uploaded for this chat.</p>", unsafe_allow_html=True)

# ── Top Bar Header ───────────────────────────────────────────────────────────
col_top_left, col_top_right = st.columns([3, 2])

with col_top_left:
    st.markdown('<div class="top-bar-title">Source RAG</div>', unsafe_allow_html=True)

with col_top_right:
    col_kb, col_theme = st.columns([3, 2])
    with col_kb:
        kb_option = st.selectbox(
            "Knowledge Base",
            ["📚 Current Chat Docs", "🌐 Web & All Docs"],
            label_visibility="collapsed",
            key="kb_selector",
        )
    with col_theme:
        theme_icon = "☀️ Light" if is_dark else "🌙 Dark"
        if st.button(theme_icon, use_container_width=True):
            st.session_state.theme = "light" if is_dark else "dark"
            st.rerun()

st.markdown(f'<div style="border-bottom:1px solid {border_subtle};margin-bottom:1.5rem"></div>', unsafe_allow_html=True)

# ── Message Thread Display ───────────────────────────────────────────────────
if not st.session_state.messages:
    # Empty State Callout
    st.markdown(
        f"""
        <div style="text-align:center;padding:3rem 1.5rem;background:{bg_surface};border:1px dashed {border_subtle};border-radius:18px;margin:2rem 0;">
            <p class="serif-font" style="font-size:22px;font-weight:600;margin-bottom:8px;color:{text_primary};">Upload a document to start asking questions</p>
            <p style="font-size:14px;color:{text_muted};max-width:480px;margin:0 auto 1.5rem auto;">
                Ask questions grounded directly in your PDFs, spreadsheets, and files with transparent source citations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        if msg.get("image_path") and os.path.exists(msg["image_path"]):
            st.image(msg["image_path"], width=300)
    else:
        # Assistant Message (No bubble card, plain text with avatar)
        st.markdown(
            f"""
            <div class="assistant-container">
                <div class="assistant-header">
                    <div class="assistant-avatar">✦</div>
                    <div class="assistant-label">Assistant</div>
                </div>
                <div class="assistant-content">{msg["content"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Render Source Citation Chips if present
        raw_sources = msg.get("sources")
        sources_list = []
        if raw_sources:
            try:
                sources_list = json.loads(raw_sources) if isinstance(raw_sources, str) else raw_sources
            except Exception:
                sources_list = [str(raw_sources)]
        
        if sources_list:
            citation_html = '<div class="citation-strip">'
            for idx, src in enumerate(sources_list, start=1):
                clean_src = os.path.basename(str(src))
                citation_html += f'<span class="citation-chip">[{idx}] 📄 {clean_src}</span>'
            citation_html += '</div>'
            st.markdown(citation_html, unsafe_allow_html=True)

# Display attached file chip if present
if st.session_state.get("pending_doc_name"):
    st.markdown(
        f'<div style="background:{bg_surface};border:1px solid {border_subtle};padding:6px 12px;border-radius:12px;display:inline-block;font-size:13px;margin-bottom:10px;">📄 {st.session_state.pending_doc_name} &nbsp; <small style="color:{accent_clay}">(Indexed & Ready)</small></div>',
        unsafe_allow_html=True,
    )
elif st.session_state.get("pending_image_name"):
    st.markdown(
        f'<div style="background:{bg_surface};border:1px solid {border_subtle};padding:6px 12px;border-radius:12px;display:inline-block;font-size:13px;margin-bottom:10px;">🖼️ {st.session_state.pending_image_name} &nbsp; <small style="color:{accent_clay}">(Image Ready)</small></div>',
        unsafe_allow_html=True,
    )

# ── Bottom Input & Attachment Bar ─────────────────────────────────────────────
attach_col, input_col = st.columns([1, 12])

with attach_col:
    with st.popover("+", help="Attach document or image"):
        st.markdown("<p style='font-size:13px;font-weight:600;margin-bottom:6px'>Add Attachment</p>", unsafe_allow_html=True)
        tab_doc, tab_img = st.tabs(["📄 Document", "🖼️ Image"])

        with tab_doc:
            doc_file = st.file_uploader(
                "Upload Document",
                type=["pdf", "csv", "ppt", "pptx", "txt", "md", "json", "tsv"],
                label_visibility="collapsed",
                key="chat_doc_uploader_segmented",
            )
            if doc_file and rag_engine and st.session_state.user_id:
                file_key = f"doc_{doc_file.name}_{doc_file.size}_{st.session_state.conversation_id}"
                if st.session_state.get("last_indexed_chat_file") != file_key:
                    with st.spinner(f"Indexing {doc_file.name}…"):
                        try:
                            data = doc_file.read()
                            chunks = rag_engine.ingestion.ingest_bytes(
                                data, doc_file.name, st.session_state.user_id, st.session_state.conversation_id
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
    user_input = st.chat_input("Ask about your documents...")

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
        title = user_input[:44] + ("..." if len(user_input) > 44 else "")
        db.update_conversation_title(st.session_state.conversation_id, title)
        st.session_state.conversations = db.list_conversations(st.session_state.user_id)

    st.markdown(
        f'<div class="user-bubble">{user_input}</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.pending_image:
        st.image(st.session_state.pending_image, width=300)

    # Animated Shimmer Indicator
    shimmer_placeholder = st.empty()
    shimmer_placeholder.markdown('<div class="shimmer-text">✦ Searching documents & analyzing knowledge base…</div>', unsafe_allow_html=True)

    try:
        use_web = st.session_state.get("kb_selector") == "🌐 Web & All Docs"
        result = rag_engine.chat(
            user_id=st.session_state.user_id,
            conversation_id=st.session_state.conversation_id,
            message=user_input,
            image_base64=image_b64,
            use_web_search=use_web,
        )
        reply = result["reply"]
        sources = result.get("sources", [])
        web_sources = result.get("web_sources", [])
        all_sources = sources + web_sources

        try:
            db.add_message(
                st.session_state.conversation_id,
                "assistant",
                reply,
                image_path=None,
                sources=all_sources,
            )
        except TypeError:
            db.add_message(
                st.session_state.conversation_id,
                "assistant",
                reply,
                image_path=None,
            )

        st.session_state.pending_image = None
        st.session_state.pending_image_name = None

        shimmer_placeholder.empty()

        # Render Assistant Response cleanly without container bubble
        st.markdown(
            f"""
            <div class="assistant-container">
                <div class="assistant-header">
                    <div class="assistant-avatar">✦</div>
                    <div class="assistant-label">Assistant</div>
                </div>
                <div class="assistant-content">{reply}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if all_sources:
            citation_html = '<div class="citation-strip">'
            for idx, src in enumerate(all_sources, start=1):
                clean_src = os.path.basename(str(src))
                citation_html += f'<span class="citation-chip">[{idx}] 📄 {clean_src}</span>'
            citation_html += '</div>'
            st.markdown(citation_html, unsafe_allow_html=True)

        st.session_state.messages = db.get_messages(st.session_state.conversation_id)
        st.rerun()

    except Exception as exc:
        shimmer_placeholder.empty()
        st.error(f"Error generating response: {exc}")
