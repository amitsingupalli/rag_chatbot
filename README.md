# RAG Chatbot

An Advanced RAG (Retrieval-Augmented Generation) Chatbot built with **LlamaIndex**, **FastAPI**, **Streamlit**, and **Groq**.

## Features
- **Fast & Powerful LLM**: Powered by Groq (`llama-3.3-70b-versatile`).
- **Multimodal OCR**: Support for PDF, Text, Markdown, and Image OCR.
- **Web Search Integration**: Automatic real-time web search for current events.
- **User Privacy & Storage**: Persistent per-user SQLite history and ChromaDB vector store.

## Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/amitsingupalli/rag_chatbot.git
   cd rag_chatbot
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file based on `.env.example`:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. Run the application:
   ```bash
   python Rag_Chatbot.py
   ```
