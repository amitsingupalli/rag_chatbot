"""Advanced RAG engine with hybrid retrieval, reranking, and chat memory."""

from __future__ import annotations

import logging
import os
from typing import Any

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import ChatMessage, MessageRole
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from backend.config import settings
from backend.db.database import Database
from backend.rag.ingestion import IngestionPipeline
from backend.rag.memory import PersistentMemory
from backend.rag.ocr import process_image_base64
from backend.rag.web_search import search_web, should_search_web

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful, precise AI assistant powered by an Advanced RAG system.
Use the provided context, live web search results (when available), and conversation history.
For current affairs, news, or time-sensitive questions, prioritize live web search results.
Cite sources when using web results (mention site names or URLs).
If neither local documents nor web results contain enough information, say so clearly.
When image OCR text is provided, analyze it carefully and reference specific details.
Be concise but thorough. Format responses with markdown when helpful."""


class AdvancedRAGEngine:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.memory = PersistentMemory(db)
        self.ingestion = IngestionPipeline()
        self._setup_llm_and_embeddings()
        self._setup_query_engine()

    def _setup_llm_and_embeddings(self) -> None:
        groq_key = os.getenv("GROQ_API_KEY") or settings.groq_api_key

        if not groq_key:
            raise ValueError(
                "GROQ_API_KEY environment variable or config setting is required to run the LLM."
            )

        self.llm = Groq(
            model=settings.groq_model,
            api_key=groq_key,
            temperature=0.3,
        )
        logger.info("Using Groq LLM with model %s", settings.groq_model)
        Settings.llm = self.llm

        # Set up HuggingFace embeddings
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=settings.embedding_model,
        )
        logger.info("Using HuggingFace Embedding with model %s", settings.embedding_model)

    def _setup_query_engine(self) -> None:
        index = self.ingestion.index
        vector_retriever = index.as_retriever(similarity_top_k=settings.similarity_top_k)
        retriever = vector_retriever

        try:
            doc_count = len(index.docstore.docs)
            if doc_count > 0:
                from llama_index.retrievers.bm25 import BM25Retriever

                bm25_retriever = BM25Retriever.from_defaults(
                    docstore=index.docstore,
                    similarity_top_k=settings.similarity_top_k,
                )
                retriever = QueryFusionRetriever(
                    [vector_retriever, bm25_retriever],
                    similarity_top_k=settings.similarity_top_k,
                    num_queries=2,
                    mode="reciprocal_rerank",
                    use_async=False,
                )
        except Exception as exc:
            logger.warning("BM25 retriever unavailable, using vector only: %s", exc)

        try:
            reranker = SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                top_n=settings.rerank_top_n,
            )
            node_postprocessors = [reranker]
        except Exception as exc:
            logger.warning("Reranker unavailable: %s", exc)
            node_postprocessors = []

        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=node_postprocessors,
            llm=self.llm,
        )
        self.query_transform = StepDecomposeQueryTransform(llm=self.llm, verbose=False)

    def _build_chat_history(self, conversation_id: str) -> list[ChatMessage]:
        messages = self.db.get_messages(conversation_id)
        history: list[ChatMessage] = []
        for msg in messages[-20:]:
            role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
            history.append(ChatMessage(role=role, content=msg["content"]))
        return history

    def _retrieve_context(self, query: str) -> tuple[str, list[str]]:
        try:
            transformed = self.query_transform.run(query)
            sub_queries = transformed if isinstance(transformed, list) else [query]

            all_nodes = []
            for sub_q in sub_queries[:3]:
                response = self.query_engine.retrieve(str(sub_q))
                all_nodes.extend(response)

            seen: set[str] = set()
            sources: list[str] = []
            chunks: list[str] = []
            for node in all_nodes:
                node_id = node.node.node_id
                if node_id in seen:
                    continue
                seen.add(node_id)
                source = node.node.metadata.get("source", "unknown")
                sources.append(source)
                chunks.append(node.node.get_content())

            context = "\n\n---\n\n".join(chunks[: settings.rerank_top_n * 2])
            return context, list(dict.fromkeys(sources))
        except Exception as exc:
            logger.warning("Retrieval failed: %s", exc)
            return "", []

    def chat(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        image_base64: str | None = None,
        use_web_search: bool | None = None,
    ) -> dict[str, Any]:
        enriched_message = message
        if image_base64:
            ocr_result = process_image_base64(image_base64)
            enriched_message = (
                f"{message}\n\n[Image OCR Content]\n{ocr_result['ocr_text']}"
            )

        user_memory = self.memory.get_context(user_id)
        rag_context, sources = self._retrieve_context(enriched_message)

        web_context = ""
        web_sources: list[str] = []
        used_web_search = should_search_web(message, force=use_web_search)
        if used_web_search:
            web_context, web_sources = search_web(enriched_message)

        history = self._build_chat_history(conversation_id)

        history_text = ""
        if history:
            history_text = "\n".join(
                f"{'User' if m.role == MessageRole.USER else 'Assistant'}: {m.content}"
                for m in history[-10:]
            )

        prompt = f"""{SYSTEM_PROMPT}

{user_memory}

Conversation history:
{history_text or '(none)'}

Retrieved knowledge base context:
{rag_context or '(no relevant documents found)'}

{web_context or '(no web search performed)'}

User question:
{enriched_message}

Provide a helpful answer:"""

        response = self.llm.complete(prompt)
        reply = str(response).strip()

        self.memory.extract_and_store(user_id, message, reply)

        if "remember" in message.lower():
            self.memory.store_explicit(user_id, message)

        return {
            "reply": reply,
            "sources": sources,
            "web_sources": web_sources,
            "used_web_search": used_web_search,
        }
