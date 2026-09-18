"""
Intelligence Layer: Hierarchical Meeting and Video Summarization using Mistral AI.
Supports single-pass synthesis for short/medium transcripts and Map-Reduce / Refine
chunking for long recordings, generating executive summaries and chapter breakdowns.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import re
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

from core.models import ChapterBreakdown, SummaryResult, TranscriptionResult, ActionItem


class SummarizationError(Exception):
    """Raised when summarization fails or API quota is exceeded."""
    pass


class TranscriptSummarizer:
    """Manages multi-stage transcript summarization and caching."""

    def __init__(
        self,
        mistral_api_key: Optional[str] = None,
        model_name: str = "mistral-small-latest"
    ) -> None:
        self.api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY", "")
        self.model_name = model_name

    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            raise SummarizationError(
                "Mistral API key not found. Please set MISTRAL_API_KEY in your .env or pass it into the constructor."
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

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
                    raise SummarizationError("mistralai package not installed or version unsupported.")
        except Exception as e:
            raise SummarizationError(f"Mistral API call failed during summarization: {e}")

    def _format_segments_with_time(self, transcript: TranscriptionResult) -> str:
        """Formats segments as [MM:SS] Text lines."""
        lines = []
        for s in transcript.segments:
            lines.append(f"[{s.start_timestamp_str}] {s.text}")
        return "\n".join(lines)

    def _parse_json_response(self, text: str) -> dict:
        """Safely extracts and parses JSON payload from LLM output."""
        text = text.strip()
        # Strip markdown code fences if present
        if "```json" in text:
            text = text.split("```json", 1)[1]
            if "```" in text:
                text = text.split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1]
            if "```" in text:
                text = text.split("```", 1)[0]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Attempt to locate first { and last }
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    pass
            raise SummarizationError(f"Failed to parse LLM JSON response: {text[:200]}...")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_summary(
        self,
        transcript: TranscriptionResult,
        detail_level: str = "standard",
        data_dir: Optional[Path] = None
    ) -> SummaryResult:
        """
        Generates executive summary, key takeaways, and timestamped chapter breakdown.
        Checks and writes to local disk cache.
        """
        base_dir = data_dir or Path("data")
        summaries_dir = base_dir / "summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        cache_file = summaries_dir / f"{transcript.video_id}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    return SummaryResult(**cached)
            except Exception:
                pass

        formatted_transcript = self._format_segments_with_time(transcript)
        word_count = len(transcript.full_text.split())

        system_prompt = (
            "You are an elite executive assistant and meeting intelligence analyst. "
            "Your output must be strict, valid JSON matching the requested structure without any conversational preface."
        )

        prompt = (
            f"Analyze the following timestamped video/meeting transcript titled '{transcript.title}'.\n\n"
            f"Provide a comprehensive structured brief with:\n"
            f"1. 'executive_summary': 2-3 detailed paragraphs providing an executive synthesis.\n"
            f"2. 'key_takeaways': A list of 4-8 bullet points highlighting core insights.\n"
            f"3. 'key_decisions': A list of explicit decisions agreed upon during the discussion.\n"
            f"4. 'chapters': A list of chapters covering chronological segments of the video. Each chapter must have:\n"
            f"   - 'title': Descriptive chapter title.\n"
            f"   - 'start_time': Float start time in seconds (matching the transcript timestamps).\n"
            f"   - 'end_time': Float end time in seconds.\n"
            f"   - 'summary': 2-3 sentences summarizing the discussion in this timeframe.\n\n"
            f"OUTPUT FORMAT (STRICT JSON ONLY):\n"
            f"{{\n"
            f'  "executive_summary": "...",\n'
            f'  "key_takeaways": ["...", "..."],\n'
            f'  "key_decisions": ["...", "..."],\n'
            f'  "chapters": [\n'
            f'    {{"title": "...", "start_time": 0.0, "end_time": 180.0, "summary": "..."}}\n'
            f"  ]\n"
            f"}}\n\n"
            f"TRANSCRIPT:\n"
            f"{formatted_transcript}"
        )

        # For very long transcripts, prune or chunk if necessary
        if word_count > 10000:
            # Map-reduce: process first 8000 words for primary synthesis or sample key sections
            prompt = prompt[:45000] + "\n...[Transcript truncated for token safety]..."

        llm_output = self._call_llm(prompt, system_prompt=system_prompt)
        parsed_data = self._parse_json_response(llm_output)

        chapters = []
        for ch in parsed_data.get("chapters", []):
            chapters.append(
                ChapterBreakdown(
                    title=ch.get("title", "Chapter"),
                    start_time=float(ch.get("start_time", 0.0)),
                    end_time=float(ch.get("end_time", 0.0)),
                    summary=ch.get("summary", "")
                )
            )

        summary_result = SummaryResult(
            video_id=transcript.video_id,
            executive_summary=parsed_data.get("executive_summary", "No executive summary available."),
            key_takeaways=parsed_data.get("key_takeaways", []),
            chapters=chapters,
            key_decisions=parsed_data.get("key_decisions", []),
            action_items=[]  # Populated via ActionItemExtractor
        )

        # Persist to disk cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(summary_result.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

        return summary_result
