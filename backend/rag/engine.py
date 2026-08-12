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
try:
    from llama_index.core.llms import ChatMessage, MessageRole
except ImportError:
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

SYSTEM_PROMPT = """You are a highly precise, detail-oriented RAG AI research assistant.
Analyze all retrieved document context thoroughly and answer the user's question with maximum technical accuracy.

CRITICAL PRECISION RULES:
- ACCURACY ON NUMERICAL METRICS & DATASETS: Pay meticulous attention to exact numbers, dataset image counts (e.g., specific CT/PET image totals), sample sizes, and statistics in the text.
- MULTI-STAGE WORKFLOW & PIPELINE DETAILS: Accurately capture every stage of multi-step model workflows, section descriptions, figure captions, and model execution steps (e.g., explicit multi-pass model runs in Stage 3 rather than over-simplifying them).
- COMPLETE & DETAILED RESPONSES: Provide complete, fully fleshed-out explanations without truncating or cutting off steps mid-sentence.
- Do NOT output internal chunking process or raw unformatted context dumps.
- Format responses cleanly with markdown headings, bold terms, bullet points, and numbered lists when appropriate."""


class AdvancedRAGEngine:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.memory = PersistentMemory(db)
        self.ingestion = IngestionPipeline()
        self._setup_llm_and_embeddings()
        self._setup_query_engine()

    def _setup_llm_and_embeddings(self) -> None:
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or settings.gemini_api_key

        if not gemini_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable or config setting is required to run the LLM."
            )

        from llama_index.llms.gemini import Gemini

        model_name = settings.gemini_model
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        self.llm = Gemini(
            model=model_name,
            api_key=gemini_key,
            temperature=0.2,
            max_tokens=settings.max_tokens,
        )
        self.fallback_llm = None
        logger.info("Using ONLY Google Gemini LLM with model %s (max_tokens=%d)", model_name, settings.max_tokens)
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

        node_postprocessors = []
        self._reranker = None
        self._reranker_initialized = False

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

    def _retrieve_context(
        self, query: str, user_id: str | None = None, conversation_id: str | None = None
    ) -> tuple[str, list[str]]:
        try:
            from llama_index.core.schema import QueryBundle
            all_nodes = []

            # 1. Retrieve directly with raw query first
            try:
                raw_nodes = self.query_engine.retrieve(QueryBundle(query))
                all_nodes.extend(raw_nodes)
            except Exception as exc:
                logger.warning("Raw query retrieval error: %s", exc)

            # 2. Check for overview / document summary intent
            lower_q = query.lower()
            is_summary_q = any(w in lower_q for w in ["pdf", "document", "file", "about", "summary", "overview", "detail", "what is", "explain"])

            if not all_nodes or is_summary_q:
                try:
                    transformed = self.query_transform.run(query)
                    sub_queries = transformed if isinstance(transformed, list) else []
                    for sub_q in sub_queries[:2]:
                        if str(sub_q).strip() != query.strip():
                            sub_nodes = self.query_engine.retrieve(QueryBundle(str(sub_q)))
                            all_nodes.extend(sub_nodes)
                except Exception:
                    pass

            seen: set[str] = set()
            sources: list[str] = []
            chunks: list[str] = []
            chunk_sources: list[str] = []

            for node in all_nodes:
                node_id = node.node.node_id
                if node_id in seen:
                    continue
                node_conv = node.node.metadata.get("conversation_id")
                node_user = node.node.metadata.get("user_id")
                if conversation_id and node_conv and node_conv != "global" and node_conv != conversation_id:
                    continue
                if user_id and node_user and node_user != "global" and node_user != user_id:
                    continue

                seen.add(node_id)
                source = node.node.metadata.get("source", "unknown")
                sources.append(source)
                chunks.append(node.node.get_content())
                chunk_sources.append(source)

            # 3. Robust direct ChromaDB collection query fallback
            if not chunks or is_summary_q:
                try:
                    filter_dict = {}
                    if conversation_id:
                        filter_dict["conversation_id"] = conversation_id
                    elif user_id:
                        filter_dict["user_id"] = user_id

                    res_direct = self.ingestion._collection.get(
                        where=filter_dict if filter_dict else None,
                        limit=25
                    )
                    
                    direct_docs = res_direct.get("documents", [])
                    direct_metas = res_direct.get("metadatas", [])
                    
                    for doc_text, meta in zip(direct_docs, direct_metas):
                        if doc_text and doc_text.strip():
                            src = meta.get("source", "uploaded_doc")
                            
                            # Deduplicate content
                            text_hash = str(hash(doc_text))
                            if text_hash in seen:
                                continue
                            seen.add(text_hash)
                            
                            sources.append(src)
                            chunks.append(doc_text)
                            chunk_sources.append(src)
                except Exception as exc:
                    logger.warning("ChromaDB direct query fallback failed: %s", exc)

            # 4. Format each context chunk with clear source filename
            import os
            formatted_chunks = []
            for doc_text, src in zip(chunks, chunk_sources):
                clean_src = os.path.basename(str(src))
                # Strip out UUID prefixes from filenames for clean presentation to LLM
                display_src = clean_src
                if len(clean_src) > 33 and clean_src[32] == "_":
                    display_src = clean_src[33:]
                formatted_chunks.append(f"[Source Document: {display_src}]\n{doc_text}")

            context = "\n\n---\n\n".join(formatted_chunks[: settings.rerank_top_n * 4])
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
        pil_img = None
        if image_base64:
            try:
                from backend.rag.ocr import load_image_from_base64
                pil_img = load_image_from_base64(image_base64)
            except Exception as img_err:
                logger.warning("Failed loading image base64: %s", img_err)

        user_memory = self.memory.get_context(user_id)
        rag_context, sources = self._retrieve_context(enriched_message, user_id=user_id, conversation_id=conversation_id)

        web_context = ""
        web_sources: list[str] = []
        used_web = False
        if use_web_search is not False and should_search_web(message, force=use_web_search):
            try:
                web_context, web_sources = search_web(message)
                if web_context:
                    used_web = True
            except Exception as exc:
                logger.warning("Web search failed: %s", exc)

        # Handle multimodal image vision directly with Gemini
        if pil_img:
            try:
                import google.generativeai as genai
                gemini_key = os.getenv("GEMINI_API_KEY") or settings.gemini_api_key
                genai.configure(api_key=gemini_key)
                model_name = settings.gemini_model.replace("models/", "")
                vision_model = genai.GenerativeModel(model_name)

                vision_prompt = f"""{SYSTEM_PROMPT}

Retrieved Context:
{rag_context or '(none)'}

User Query:
{message}"""
                vision_res = vision_model.generate_content(
                    [vision_prompt, pil_img],
                    generation_config={"max_output_tokens": settings.max_tokens}
                )
                reply = vision_res.text.strip()

                self.memory.extract_and_store(user_id, message, reply)
                return {
                    "reply": reply,
                    "sources": sources,
                    "web_sources": web_sources,
                    "used_web_search": used_web,
                }
            except Exception as vision_err:
                logger.warning("Gemini Vision processing failed: %s", vision_err)

        history = self._build_chat_history(conversation_id)

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ]
        if user_memory:
            messages.append(ChatMessage(role=MessageRole.SYSTEM, content=f"User Context Memory:\n{user_memory}"))

        # Add conversation history
        for msg in history[-10:]:
            role = MessageRole.USER if msg.role == MessageRole.USER else MessageRole.ASSISTANT
            messages.append(ChatMessage(role=role, content=msg.content))

        # Add current user query and retrieved context
        user_message_content = f"""Retrieved knowledge base context:
{rag_context or '(no relevant documents found)'}

Web search context:
{web_context or '(no web search used)'}

User question:
{enriched_message}"""

        messages.append(ChatMessage(role=MessageRole.USER, content=user_message_content))

        try:
            response = self.llm.chat(messages)
            reply = str(response.message.content).strip()
        except Exception as exc:
            if self.fallback_llm and ("429" in str(exc) or "quota" in str(exc).lower() or "limit" in str(exc).lower()):
                logger.warning("Primary LLM quota limit hit (%s). Falling back to Groq...", exc)
                response = self.fallback_llm.chat(messages)
                reply = str(response.message.content).strip()
            else:
                raise exc

        self.memory.extract_and_store(user_id, message, reply)

        if "remember" in message.lower():
            self.memory.store_explicit(user_id, message)

        return {
            "reply": reply,
            "sources": sources,
            "web_sources": web_sources,
            "used_web_search": used_web,
        }

    def chat_stream(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        image_base64: str | None = None,
        use_web_search: bool | None = None,
    ):
        enriched_message = message
        if image_base64:
            ocr_result = process_image_base64(image_base64)
            enriched_message = (
                f"{message}\n\n[Image OCR Content]\n{ocr_result['ocr_text']}"
            )

        user_memory = self.memory.get_context(user_id)
        rag_context, sources = self._retrieve_context(enriched_message, user_id=user_id, conversation_id=conversation_id)

        web_context = ""
        web_sources: list[str] = []
        used_web = False
        if use_web_search is not False and should_search_web(message, force=use_web_search):
            try:
                web_context, web_sources = search_web(message)
                if web_context:
                    used_web = True
            except Exception as exc:
                logger.warning("Web search failed: %s", exc)

        history = self._build_chat_history(conversation_id)

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ]
        if user_memory:
            messages.append(ChatMessage(role=MessageRole.SYSTEM, content=f"User Context Memory:\n{user_memory}"))

        for msg in history[-10:]:
            role = MessageRole.USER if msg.role == MessageRole.USER else MessageRole.ASSISTANT
            messages.append(ChatMessage(role=role, content=msg.content))

        user_message_content = f"""Retrieved knowledge base context:
{rag_context or '(no relevant documents found)'}

Web search context:
{web_context or '(no web search used)'}

User question:
{enriched_message}"""

        messages.append(ChatMessage(role=MessageRole.USER, content=user_message_content))

        response_stream = self.llm.stream_chat(messages)
        full_reply = ""
        for chunk in response_stream:
            delta = chunk.delta or ""
            if delta:
                full_reply += delta
                yield delta, sources, web_sources, used_web

        self.memory.extract_and_store(user_id, message, full_reply)

        if "remember" in message.lower():
            self.memory.store_explicit(user_id, message)
