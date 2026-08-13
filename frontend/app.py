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
    page_title="RAG Analysis Agent",
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
if "preset_prompt" not in st.session_state:
    st.session_state.preset_prompt = None

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

# ── Theme Dynamic CSS matching Mini Project Claude UI ─────────────────────────
is_dark = st.session_state.theme == "dark"

bg_app = "#18181b" if is_dark else "#f5f3ee"
bg_sidebar = "#09090b" if is_dark else "#edeae1"
bg_card = "#27272a" if is_dark else "#ffffff"
border_color = "rgba(255, 255, 255, 0.08)" if is_dark else "rgba(0, 0, 0, 0.08)"
border_hover = "rgba(255, 255, 255, 0.15)" if is_dark else "rgba(0, 0, 0, 0.15)"
text_main = "#f4f4f5" if is_dark else "#18181b"
text_muted = "#a1a1aa" if is_dark else "#71717a"
text_dim = "#71717a" if is_dark else "#a1a1aa"
accent_claude = "#da7756"
accent_hover = "#e0886b"
emerald = "#10b981"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: {bg_app} !important;
    color: {text_main} !important;
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
    color: {accent_claude} !important;
    background: {bg_card} !important;
    border: 1px solid {border_color} !important;
    border-radius: 8px !important;
    z-index: 999999 !important;
}}

.block-container {{
    padding-top: 1rem;
    padding-bottom: 7rem;
    max-width: 820px;
}}

/* Top Header Controls */
.top-header-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 18px;
    background: {bg_card};
    border: 1px solid {border_color};
    border-radius: 14px;
    margin-bottom: 1.5rem;
}}

.chat-title-group {{
    display: flex;
    align-items: center;
    gap: 10px;
}}

.chat-title-group h2 {{
    font-size: 1rem;
    font-weight: 600;
    margin: 0;
    color: {text_main};
}}

.model-badge {{
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    color: {text_muted};
    background: rgba(255, 255, 255, 0.06);
    padding: 3px 10px;
    border-radius: 12px;
    border: 1px solid {border_color};
}}

/* Sidebar Layout */
section[data-testid="stSidebar"] {{
    background: {bg_sidebar} !important;
    border-right: 1px solid {border_color} !important;
}}

section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {{
    color: {text_main} !important;
}}

/* Primary & Sidebar Buttons */
.stButton > button {{
    border-radius: 10px !important;
    border: 1px solid {border_color} !important;
    background-color: {bg_card} !important;
    color: {text_main} !important;
    transition: all 0.2s ease !important;
}}

.stButton > button:hover {{
    border-color: {accent_claude} !important;
    color: {accent_claude} !important;
}}

/* New Chat Button */
.new-chat-btn button {{
    background-color: rgba(218, 119, 86, 0.12) !important;
    color: {accent_claude} !important;
    border: 1.5px solid {accent_claude} !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}}

.new-chat-btn button:hover {{
    background-color: {accent_claude} !important;
    color: #ffffff !important;
}}

/* Welcome Screen */
.welcome-screen {{
    max-width: 680px;
    width: 100%;
    margin: 1.5rem auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    gap: 14px;
}}

.claude-avatar-large {{
    width: 56px;
    height: 56px;
    border-radius: 16px;
    background: linear-gradient(135deg, #da7756 0%, #a855f7 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 26px;
    box-shadow: 0 8px 20px rgba(218, 119, 86, 0.3);
    margin-bottom: 4px;
}}

.welcome-screen h1 {{
    font-size: 1.6rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: {text_main};
    margin: 0;
}}

.welcome-sub {{
    font-size: 0.9rem;
    color: {text_muted};
    max-width: 480px;
    line-height: 1.5;
}}

/* Starter Cards Grid */
.starter-card {{
    background: {bg_card};
    border: 1px solid {border_color};
    border-radius: 14px;
    padding: 14px 16px;
    text-align: left;
    display: flex;
    flex-direction: column;
    gap: 4px;
    transition: all 0.2s ease;
}}

.starter-title {{
    font-size: 0.88rem;
    font-weight: 600;
    color: {text_main};
}}

.starter-desc {{
    font-size: 0.78rem;
    color: {text_muted};
}}

/* Message Bubbles & Feeds */
.user-bubble {{
    background-color: {bg_card};
    color: {text_main};
    border: 1px solid {border_color};
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin-bottom: 1.2rem;
    margin-left: auto;
    max-width: 82%;
    font-size: 15px;
    line-height: 1.6;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
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
    border-radius: 6px;
    background: linear-gradient(135deg, #da7756 0%, #a855f7 100%);
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: bold;
}}

.assistant-label {{
    font-size: 13px;
    font-weight: 600;
    color: {accent_claude};
}}

.assistant-content {{
    color: {text_main};
    font-size: 15px;
    line-height: 1.65;
    padding-left: 32px;
}}

/* Collapsible Thought Accordion */
.thought-accordion {{
    background: rgba(0, 0, 0, 0.25);
    border: 1px solid {border_color};
    border-radius: 10px;
    margin: 8px 0 12px 32px;
    overflow: hidden;
}}

.thought-accordion summary {{
    padding: 8px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    color: {text_muted};
    cursor: pointer;
    user-select: none;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.thought-inner {{
    padding: 10px 12px;
    border-top: 1px solid {border_color};
    font-size: 0.78rem;
    color: #93c5fd;
    font-family: 'JetBrains Mono', monospace;
    background: rgba(0, 0, 0, 0.4);
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
    background-color: {bg_card};
    color: {accent_claude};
    border: 1px solid {border_color};
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
    transition: all 0.15s ease;
}}

.citation-chip:hover {{
    background-color: {accent_claude};
    color: #ffffff;
    border-color: {accent_claude};
}}

/* Shimmer Animation */
@keyframes shimmer {{
    0% {{ opacity: 0.4; }}
    50% {{ opacity: 1; }}
    100% {{ opacity: 0.4; }}
}}

.shimmer-text {{
    color: {accent_claude};
    font-size: 13px;
    font-weight: 500;
    animation: shimmer 1.5s infinite ease-in-out;
    padding-left: 32px;
    margin-bottom: 10px;
}}

/* Status Dot */
.status-dot {{
    height: 7px;
    width: 7px;
    border-radius: 50%;
    display: inline-block;
    background-color: {emerald};
    margin-right: 6px;
}}

/* Sleek Left-Side Attachment + Button */
div[data-testid="stPopover"] {{
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}}

div[data-testid="stPopover"] > button {{
    border-radius: 50% !important;
    background-color: {bg_card} !important;
    border: 1.5px solid {border_color} !important;
    color: #ffffff !important;
    font-size: 1.25rem !important;
    font-weight: 500 !important;
    width: 44px !important;
    height: 44px !important;
    min-width: 44px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.2s ease !important;
    margin-top: 2px !important;
}}

div[data-testid="stPopover"] > button:hover {{
    border-color: {accent_claude} !important;
    background-color: {accent_claude} !important;
    color: #ffffff !important;
    transform: scale(1.08) !important;
}}

/* Fixed Bottom Dock Container (Claude & Gemini style) */
div[data-testid="stBottom"] {{
    background: linear-gradient(180deg, rgba(24, 24, 27, 0) 0%, rgba(24, 24, 27, 0.95) 40%, {bg_app} 100%) !important;
    padding-bottom: 1rem !important;
}}

div[data-testid="stChatInput"] {{
    background-color: {bg_card} !important;
    border: 1.5px solid {border_color} !important;
    border-radius: 18px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
}}

div[data-testid="stChatInput"]:focus-within {{
    border-color: {accent_claude} !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7), 0 0 15px rgba(218, 119, 86, 0.25) !important;
}}

/* Dock Footer info */
.dock-footer-info {{
    font-size: 0.72rem;
    color: {text_dim};
    text-align: center;
    margin-top: 6px;
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
    st.markdown("<p style='font-size:11px;font-weight:700;letter-spacing:0.06em;color:#71717a;margin-bottom:8px'>PERSONAL WORKSPACE</p>", unsafe_allow_html=True)
    
    current_uname = st.session_state.get("username", "Default User")
    
    st.markdown(
        f"""
        <div style="background:{bg_card};border:1px solid {border_color};border-radius:10px;padding:10px;margin-bottom:10px;">
            <div style="font-size:13px;font-weight:600;color:{text_main};display:flex;align-items:center;gap:6px;">
                <span>👤</span>
                <span>{current_uname}</span>
            </div>
            <div style="font-size:11px;color:#10b981;margin-top:4px;">
                🔒 Private Memory & History Active
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_acct, col_mem = st.columns(2)
    with col_acct:
        with st.popover("👤 User", use_container_width=True):
            st.markdown("<p style='font-size:13px;font-weight:600;margin-bottom:4px;'>Switch / Create Account</p>", unsafe_allow_html=True)
            new_uname = st.text_input("Username", value="", placeholder="Enter name or ID", key="new_user_input_field")
            if st.button("Switch Workspace", key="submit_new_user_btn", use_container_width=True):
                if new_uname.strip() and db:
                    u_clean = new_uname.strip()
                    user = db.get_user_by_username(u_clean)
                    if not user:
                        user = db.create_user(u_clean)
                    st.session_state.user_id = user["user_id"]
                    st.session_state.username = user["username"]
                    convs = db.list_conversations(user["user_id"])
                    st.session_state.conversations = convs
                    if convs:
                        st.session_state.conversation_id = convs[0]["conversation_id"]
                        st.session_state.messages = db.get_messages(convs[0]["conversation_id"])
                    else:
                        new_c = db.create_conversation(user["user_id"], "New Chat")
                        st.session_state.conversation_id = new_c["conversation_id"]
                        st.session_state.messages = []
                        st.session_state.conversations = [new_c]
                    st.rerun()

    with col_mem:
        with st.popover("🧠 Memory", use_container_width=True):
            st.markdown(f"<p style='font-size:13px;font-weight:600;margin-bottom:4px;'>🧠 AI Personal Memory</p>", unsafe_allow_html=True)
            if db and st.session_state.user_id:
                mems = db.get_user_memories(st.session_state.user_id, limit=20)
                if mems:
                    st.caption(f"Facts learned about {current_uname}:")
                    for m in mems:
                        st.markdown(f"• {m['content']}")
                    if st.button("🗑️ Clear Memory", key="clear_mem_btn", use_container_width=True):
                        db.clear_user_memories(st.session_state.user_id)
                        st.success("Memory cleared!")
                        st.rerun()
                else:
                    st.info("No personal memories saved yet. As you chat, the AI automatically remembers your preferences & facts!")

    st.markdown("<p style='font-size:11px;font-weight:700;letter-spacing:0.06em;color:#71717a;margin-top:14px;margin-bottom:8px'>CONVERSATIONS</p>", unsafe_allow_html=True)
    
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

    st.markdown("<p style='font-size:11px;font-weight:700;letter-spacing:0.06em;color:#71717a;margin-top:14px;margin-bottom:8px'>RECENTS</p>", unsafe_allow_html=True)

    if st.session_state.conversations:
        for conv in st.session_state.conversations:
            cid = conv["conversation_id"]
            is_active = cid == st.session_state.conversation_id
            label = conv["title"][:22] + ("…" if len(conv["title"]) > 22 else "")

            c_title, c_opts = st.columns([0.78, 0.22])
            with c_title:
                if st.button(
                    label,
                    key=f"conv_{cid}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.conversation_id = cid
                    st.session_state.pending_doc_name = None
                    st.session_state.pending_image_name = None
                    st.session_state.pending_image = None
                    st.session_state.pop("last_indexed_file", None)
                    st.session_state.pop("last_indexed_chat_file", None)
                    if db:
                        st.session_state.messages = db.get_messages(cid)
                    st.rerun()

            with c_opts:
                with st.popover("⋮", use_container_width=True):
                    st.markdown("<p style='font-size:12px;font-weight:700;margin-bottom:6px;'>Chat Options</p>", unsafe_allow_html=True)
                    
                    # 1. SHARE / COPY CONVERSATION
                    with st.expander("📋 Share (Copy Chat)"):
                        msgs = db.get_messages(cid) if db else []
                        if msgs:
                            formatted_chat = f"# Conversation: {conv['title']}\n\n"
                            for m in msgs:
                                r_label = "User" if m["role"] == "user" else "AI Assistant"
                                formatted_chat += f"**{r_label}**:\n{m['content']}\n\n---\n\n"
                            st.text_area("Copy transcript below:", value=formatted_chat, height=140, key=f"share_txt_{cid}")
                        else:
                            st.info("No messages in this chat yet.")

                    # 2. RENAME CHAT
                    with st.expander("✏️ Rename Chat"):
                        new_title_val = st.text_input("New Name", value=conv["title"], key=f"rename_in_{cid}")
                        if st.button("Save Name", key=f"save_rename_{cid}", use_container_width=True):
                            if new_title_val.strip() and db:
                                db.update_conversation_title(cid, new_title_val.strip())
                                st.session_state.conversations = db.list_conversations(st.session_state.user_id)
                                st.rerun()

                    # 3. DELETE CHAT
                    with st.expander("🗑️ Delete Chat"):
                        st.caption("Delete this chat permanently?")
                        if st.button("Confirm Delete", key=f"del_confirm_{cid}", type="primary", use_container_width=True):
                            if db:
                                db.delete_conversation(cid)
                                convs = db.list_conversations(st.session_state.user_id)
                                st.session_state.conversations = convs
                                if convs:
                                    st.session_state.conversation_id = convs[0]["conversation_id"]
                                    st.session_state.messages = db.get_messages(convs[0]["conversation_id"])
                                else:
                                    new_c = db.create_conversation(st.session_state.user_id, "New Chat")
                                    st.session_state.conversation_id = new_c["conversation_id"]
                                    st.session_state.messages = []
                                    st.session_state.conversations = [new_c]
                                st.rerun()



    st.divider()
    st.markdown(
        """
        <div style="font-size:0.75rem;color:#10b981;display:flex;align-items:center;gap:6px;padding:4px 0;">
            <span class="status-dot"></span>
            <span>Read-Only Guardrails Active</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Top Bar Navigation Header ──────────────────────────────────────────────────
st.markdown(
    """
    <div class="chat-title-group">
        <h2 style="font-size:1.1rem;font-weight:600;margin:0;">RAG Analysis Agent</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(f'<div style="border-bottom:1px solid {border_color};margin-bottom:1.5rem"></div>', unsafe_allow_html=True)


# ── Welcome Screen & Starter Cards (Shown when chat is empty) ──────────────────
if not st.session_state.messages:
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        greeting = "Good morning! What are we doing today?"
    elif 12 <= current_hour < 17:
        greeting = "Good afternoon! How can I help you today?"
    elif 17 <= current_hour < 22:
        greeting = "Good evening! How can I help you today?"
    else:
        greeting = "Good night! How can I assist with your analysis?"

    st.markdown(
        f"""
        <div class="welcome-screen">
            <div class="claude-avatar-large">✦</div>
            <h1>{greeting}</h1>
            <p class="welcome-sub">Upload your documents, ask intelligent questions, and get precise context-aware answers with transparent citations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )




# ── Render Chat Message Stream ─────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        if msg.get("image_path") and os.path.exists(msg["image_path"]):
            st.image(msg["image_path"], width=300)
    else:
        # Collapsible Thought Accordion & Response
        raw_sources = msg.get("sources")
        sources_list = []
        if raw_sources:
            try:
                sources_list = json.loads(raw_sources) if isinstance(raw_sources, str) else raw_sources
            except Exception:
                sources_list = [str(raw_sources)]

        thought_html = f"""
        <div class="assistant-container">
            <div class="assistant-header">
                <div class="assistant-avatar">✦</div>
                <div class="assistant-label">Assistant</div>
            </div>
            <div class="assistant-content">{msg["content"]}</div>
        </div>
        """
        st.markdown(thought_html, unsafe_allow_html=True)
        



# Render attached preview chips if pending
if st.session_state.get("pending_doc_name"):
    st.markdown(
        f'<div style="background:{bg_card};border:1px solid {border_color};padding:6px 12px;border-radius:12px;display:inline-block;font-size:13px;margin-bottom:10px;">📄 {st.session_state.pending_doc_name} &nbsp; <small style="color:{accent_claude}">(Indexed & Ready)</small></div>',
        unsafe_allow_html=True,
    )
elif st.session_state.get("pending_image_name"):
    st.markdown(
        f'<div style="background:{bg_card};border:1px solid {border_color};padding:6px 12px;border-radius:12px;display:inline-block;font-size:13px;margin-bottom:10px;">🖼️ {st.session_state.pending_image_name} &nbsp; <small style="color:{accent_claude}">(Image Ready)</small></div>',
        unsafe_allow_html=True,
    )


# ── Floating Input Dock with Left-Side + Button ───────────────────────────────
attach_col, input_col = st.columns([1, 14])

with attach_col:
    with st.popover("+", help="Upload document or image"):
        st.markdown("<p style='font-size:13px;font-weight:600;margin-bottom:6px'>Upload Attachment</p>", unsafe_allow_html=True)
        tab_doc, tab_img = st.tabs(["📄 Document", "🖼️ Image"])

        with tab_doc:
            doc_file = st.file_uploader(
                "Upload Document",
                type=["pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls", "html", "htm", "md", "txt", "csv", "json", "tsv"],
                label_visibility="collapsed",
                key="chat_doc_uploader_popover",
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
                key="chat_img_uploader_popover",
            )
            if img_file:
                st.session_state.pending_image = Image.open(img_file).convert("RGB")
                st.session_state.pending_image_name = img_file.name
                st.caption(f"✓ Image ready: {img_file.name}")

with input_col:
    initial_val = ""
    if st.session_state.get("preset_prompt"):
        initial_val = st.session_state.pop("preset_prompt")

    user_input = st.chat_input("Ask Anything", key="chat_input_field")

if not user_input and initial_val:
    user_input = initial_val

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

    # Shimmer indicator
    shimmer_placeholder = st.empty()
    shimmer_placeholder.markdown('<div class="shimmer-text">✦ Searching documents & executing RAG perception chain…</div>', unsafe_allow_html=True)

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

        thought_html = f"""
        <div class="assistant-container">
            <div class="assistant-header">
                <div class="assistant-avatar">✦</div>
                <div class="assistant-label">Assistant</div>
            </div>
            <div class="assistant-content">{reply}</div>
        </div>
        """
        st.markdown(thought_html, unsafe_allow_html=True)



        st.session_state.messages = db.get_messages(st.session_state.conversation_id)
        st.rerun()

    except Exception as exc:
        shimmer_placeholder.empty()
        st.error(f"Error generating response: {exc}")


