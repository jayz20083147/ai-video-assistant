"""
Translation utility for converting transcripts between languages using Mistral AI,
preserving exact start and end timestamps for every segment.
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
from core.models import TranscriptionResult, TranscriptSegment


class TranslationError(Exception):
    """Raised when translation pipeline encounters an error."""
    pass


class ContentTranslator:
    """Translates transcripts using Mistral AI LLM."""

    def __init__(self, mistral_api_key: Optional[str] = None, model_name: str = "mistral-small-latest") -> None:
        self.api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY", "")
        self.model_name = model_name

    def _get_client(self):
        if not self.api_key:
            raise TranslationError("Mistral API key is required for translation.")
        try:
            from mistralai.client import Mistral
            return Mistral(api_key=self.api_key)
        except ImportError:
            try:
                from mistralai import Mistral
                return Mistral(api_key=self.api_key)
            except ImportError:
                try:
                    from mistralai.client import MistralClient
                    return MistralClient(api_key=self.api_key)
                except ImportError:
                    raise TranslationError("mistralai library is not installed or unsupported.")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def translate_text(self, text: str, target_lang: str = "English") -> str:
        """Translates a text string to target language."""
        if not text.strip():
            return ""

        client = self._get_client()
        prompt = (
            f"You are a professional translator. Translate the following text into natural, fluent {target_lang}. "
            f"Preserve technical terms and names. Return ONLY the translated text, nothing else.\n\n"
            f"Text:\n{text}"
        )

        try:
            # Handle both new and legacy mistralai SDK interfaces
            if hasattr(client, "chat") and hasattr(client.chat, "complete"):
                response = client.chat.complete(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content.strip()
            elif hasattr(client, "chat"):
                from mistralai.models.chat_completion import ChatMessage
                response = client.chat(
                    model=self.model_name,
                    messages=[ChatMessage(role="user", content=prompt)]
                )
                return response.choices[0].message.content.strip()
            else:
                raise TranslationError("Unsupported mistralai client interface.")
        except Exception as e:
            raise TranslationError(f"Mistral translation failed: {e}")

    def translate_transcript(
        self,
        transcript: TranscriptionResult,
        target_lang: str = "English"
    ) -> TranscriptionResult:
        """
        Translates all segments in the transcript to the target language,
        maintaining time-code alignment.
        """
        if not transcript.segments:
            return transcript

        # Batch segments into chunks to avoid excessive API calls
        batch_size = 15
        translated_segments: list[TranscriptSegment] = []
        translated_full_text: list[str] = []

        for i in range(0, len(transcript.segments), batch_size):
            batch = transcript.segments[i:i + batch_size]
            formatted_batch = "\n".join([f"[{seg.id}] {seg.text}" for seg in batch])

            prompt = (
                f"You are an expert subtitle and transcript translator. Translate the following numbered lines into {target_lang}.\n"
                f"RULES:\n"
                f"1. Keep the exact line numbers in square brackets like [0], [1], etc.\n"
                f"2. Return ONLY the numbered translated lines, no markdown code blocks, no conversational preamble.\n\n"
                f"{formatted_batch}"
            )

            client = self._get_client()
            try:
                if hasattr(client, "chat") and hasattr(client.chat, "complete"):
                    response = client.chat.complete(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content = response.choices[0].message.content.strip()
                else:
                    from mistralai.models.chat_completion import ChatMessage
                    response = client.chat(
                        model=self.model_name,
                        messages=[ChatMessage(role="user", content=prompt)]
                    )
                    content = response.choices[0].message.content.strip()

                # Parse back by line numbers
                trans_map = {}
                for line in content.splitlines():
                    line = line.strip()
                    if line.startswith("[") and "]" in line:
                        idx_str = line[1:line.index("]")]
                        trans_val = line[line.index("]") + 1:].strip()
                        if idx_str.isdigit():
                            trans_map[int(idx_str)] = trans_val

                for seg in batch:
                    translated_text = trans_map.get(seg.id, seg.text)
                    new_seg = TranscriptSegment(
                        id=seg.id,
                        start=seg.start,
                        end=seg.end,
                        text=translated_text,
                        speaker=seg.speaker
                    )
                    translated_segments.append(new_seg)
                    translated_full_text.append(translated_text)

            except Exception as e:
                # On translation failure of a batch, fallback to original segments
                for seg in batch:
                    translated_segments.append(seg)
                    translated_full_text.append(seg.text)

        return TranscriptionResult(
            video_id=transcript.video_id,
            title=f"{transcript.title} ({target_lang})",
            duration_seconds=transcript.duration_seconds,
            language=target_lang.lower()[:2],
            engine_used=f"{transcript.engine_used}+mistral-translation",
            segments=translated_segments,
            full_text=" ".join(translated_full_text)
        )
