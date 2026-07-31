#  Advanced RAG Chatbot

An intelligent, full-stack Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **Streamlit**, **LlamaIndex**, and **Groq**. 

This application lets you query local documents (PDFs, Markdown, text, and images via OCR) alongside real-time web search integration—delivering accurate, context-aware responses with sources cited.

---

## ✨ Features

- ⚡ **High-Speed Inference**: Powered by Groq LLM (`llama-3.3-70b-versatile`) with optional local fallback (Ollama).
- 📄 **Multimodal & File Support**: Extract and index knowledge from PDFs, plain text, Markdown, and images using OCR.
- 🌐 **Live Web Search**: Uses DuckDuckGo search and web page fetching when query context requires updated information.
- 🎯 **Advanced RAG Pipeline**: Uses hybrid retrieval (BM25 + Dense Vector Search with HuggingFace/ChromaDB) and re-ranking for maximum relevance.
- 💾 **Persistent Chat History**: Session management with SQLite and persistent vector storage.
- 🐳 **Docker-Ready**: Complete Docker and `docker-compose` setup with hot-reloading for fast local testing.

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/amitsingupalli/rag_chatbot.git
cd rag_chatbot
```

### 2. Configure Environment Variables

Copy `.env.example` to create your `.env` file:

```bash
cp .env.example .env
```

Open `.env` and add your **Groq API Key**:

```env
GROQ_API_KEY=your_groq_api_key_here
```

*(All other defaults like local database paths and model settings are pre-configured.)*

---

## 🛠️ Running the Application

### Option A: Local Python Setup

1. **Create and activate a virtual environment**:
   ```bash
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the application**:
   ```bash
   python Rag_Chatbot.py
   ```
   Open your browser at `http://localhost:8501`.

---

### Option B: Running with Docker Compose

If you have Docker installed, run the full stack effortlessly:

```bash
docker-compose up --build
```

Access the Streamlit UI at `http://localhost:8501` and FastAPI docs at `http://localhost:8000/docs`.

---

## 📁 Project Architecture

```text
rag_chatbot/
├── backend/            # FastAPI server & LlamaIndex RAG pipeline
│   ├── api/            # API endpoints for chat, documents, and search
│   ├── db/             # SQLite session and document metadata storage
│   └── rag/            # Vector store, hybrid retriever, and LLM engines
├── frontend/           # Streamlit Web UI
│   └── app.py          # Interactive chat interface & file uploader
├── .env.example        # Template for configuration settings
├── docker-compose.yml  # Docker multi-container orchestrator
├── Dockerfile          # Container build setup
├── Rag_Chatbot.py      # Entry point launcher
└── requirements.txt    # Python dependencies
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for improvements or new features.

---

## 📜 License

This project is licensed under the MIT License.
