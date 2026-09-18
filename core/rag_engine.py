"""
Conversational RAG Engine for Video Question-Answering using Mistral AI and ChromaDB.
Retrieves relevant timestamped context chunks and forces the LLM to cite verifiable video timestamps.
"""

from __future__ import annotations
import os
from typing import Optional

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError:
    def retry(*args, **kwargs):
        def decorator(func):
            def wrapper(*a, **kw):
                return func(*a, **kw)
            return wrapper
        return decorator
    def stop_after_attempt(n): return n
    def wait_exponential(*args, **kwargs): return None

from core.models import Citation, RAGResponse
from core.vector_store import VectorStoreManager


class RAGQueryError(Exception):
    """Raised when RAG retrieval or answer generation fails."""
    pass


class VideoRAGEngine:
    """Conversational RAG orchestrator for indexed video transcripts."""

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        mistral_api_key: Optional[str] = None,
        model_name: str = "mistral-small-latest"
    ) -> None:
        self.vector_store = vector_store_manager
        self.api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY", "")
        self.model_name = model_name

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key:
            raise RAGQueryError("Mistral API key not configured.")

        try:
            from mistralai.client import Mistral
            client = Mistral(api_key=self.api_key)
            response = client.chat.complete(
                model=self.model_name,
                messages=messages,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except ImportError:
            try:
                from mistralai import Mistral
                client = Mistral(api_key=self.api_key)
                response = client.chat.complete(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.2
                )
                return response.choices[0].message.content.strip()
            except ImportError:
                try:
                    from mistralai.client import MistralClient
                    from mistralai.models.chat_completion import ChatMessage
                    client = MistralClient(api_key=self.api_key)
                    msgs = [ChatMessage(role=m["role"], content=m["content"]) for m in messages]
                    response = client.chat(
                        model=self.model_name,
                        messages=msgs,
                        temperature=0.2
                    )
                    return response.choices[0].message.content.strip()
                except ImportError:
                    raise RAGQueryError("mistralai library not installed or unsupported.")
        except Exception as e:
            raise RAGQueryError(f"Mistral API call failed during RAG query: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def ask(
        self,
        question: str,
        video_id: str,
        chat_history: Optional[list[dict[str, str]]] = None,
        k: int = 4
    ) -> RAGResponse:
        """
        Retrieves timestamped chunks, constructs grounded prompt, and produces response with citations.
        """
        citations: list[Citation] = self.vector_store.similarity_search(
            query=question,
            video_id=video_id,
            k=k
        )

        if not citations:
            return RAGResponse(
                query=question,
                answer=(
                    "No relevant video segments were found in the transcript to answer your question. "
                    "Please ensure the video has been processed and indexed."
                ),
                citations=[]
            )

        # Build context string with timestamp markers
        context_blocks = []
        for i, c in enumerate(citations, 1):
            context_blocks.append(
                f"[Source {i} | {c.timestamp_str}]:\n{c.text}\n"
            )
        context_text = "\n".join(context_blocks)

        system_prompt = (
            "You are an expert AI Video Assistant. Your task is to answer questions accurately and concisely "
            "based SOLELY on the provided video transcript excerpts.\n\n"
            "MANDATORY CITATION RULES:\n"
            "1. Whenever you make a factual assertion from a source excerpt, cite its exact timestamp range "
            "in square brackets at the end of the sentence, e.g. [04:12 - 04:45] or [12:30].\n"
            "2. If the answer cannot be found in the excerpts, clearly state that the provided excerpts do not mention it.\n"
            "3. Do not fabricate facts, timestamps, or outside knowledge."
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Inject conversation history if available
        if chat_history:
            for turn in chat_history[-6:]:  # Keep last 3 rounds
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        user_content = (
            f"RELEVANT VIDEO EXCERPTS:\n"
            f"----------------------------------------\n"
            f"{context_text}\n"
            f"----------------------------------------\n\n"
            f"QUESTION: {question}"
        )
        messages.append({"role": "user", "content": user_content})

        answer = self._call_llm(messages)

        return RAGResponse(
            query=question,
            answer=answer,
            citations=citations
        )
