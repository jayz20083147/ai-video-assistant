"""
Action Item and Decision Extraction Engine using Mistral AI.
Parses transcripts to detect commitments, delegated tasks, deadlines, and priorities.
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

from core.models import ActionItem, PriorityEnum, TranscriptionResult, SummaryResult


class ExtractionError(Exception):
    """Raised when structured action item extraction fails."""
    pass


class ActionItemExtractor:
    """Extracts structured tasks and decisions from transcripts."""

    def __init__(
        self,
        mistral_api_key: Optional[str] = None,
        model_name: str = "mistral-small-latest"
    ) -> None:
        self.api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY", "")
        self.model_name = model_name

    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            raise ExtractionError("Mistral API key not configured.")

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
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except ImportError:
            try:
                from mistralai import Mistral
                client = Mistral(api_key=self.api_key)
                response = client.chat.complete(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1
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
                        temperature=0.1
                    )
                    return response.choices[0].message.content.strip()
                except ImportError:
                    raise ExtractionError("mistralai package not installed or unsupported.")
        except Exception as e:
            raise ExtractionError(f"Mistral API call failed during extraction: {e}")

    def _parse_json_response(self, text: str) -> list[dict]:
        """Parses list of action item objects from JSON."""
        text = text.strip()
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
            data = json.loads(text)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("action_items", [])
            return []
        except Exception:
            match = re.search(r"(\[.*\])", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    pass
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def extract(
        self,
        transcript: TranscriptionResult,
        data_dir: Optional[Path] = None
    ) -> list[ActionItem]:
        """
        Extracts action items and updates the cached SummaryResult if available.
        """
        formatted_segments = []
        for seg in transcript.segments:
            formatted_segments.append(f"[{seg.start_timestamp_str}] {seg.text}")
        transcript_text = "\n".join(formatted_segments)

        system_prompt = (
            "You are an expert project manager extracting commitments and tasks from a transcript. "
            "Output strictly a JSON list of objects without any conversational text."
        )

        prompt = (
            f"Review this timestamped meeting/video transcript and identify all explicit or strongly implied action items, "
            f"next steps, follow-ups, and assigned tasks.\n\n"
            f"For each action item, output:\n"
            f"- 'task': A direct, clear description of the action to be taken (verb-led, e.g. 'Deploy ChromaDB container').\n"
            f"- 'assignee': Name of the person or team responsible, or 'Unassigned'.\n"
            f"- 'priority': Exactly 'High', 'Medium', or 'Low'.\n"
            f"- 'context_timestamp': The timestamp where this task was mentioned (e.g. '12:35').\n"
            f"- 'due_date': Explicit deadline if stated, or null.\n\n"
            f"OUTPUT FORMAT (STRICT JSON ARRAY OF OBJECTS ONLY):\n"
            f"[\n"
            f"  {{\n"
            f'    "task": "Refactor vector retrieval pipeline",\n'
            f'    "assignee": "Jay",\n'
            f'    "priority": "High",\n'
            f'    "context_timestamp": "04:20",\n'
            f'    "due_date": null\n'
            f"  }}\n"
            f"]\n\n"
            f"TRANSCRIPT:\n"
            f"{transcript_text[:40000]}"
        )

        llm_output = self._call_llm(prompt, system_prompt=system_prompt)
        items_raw = self._parse_json_response(llm_output)

        action_items: list[ActionItem] = []
        for item in items_raw:
            p_val = item.get("priority", "Medium").capitalize()
            if p_val not in ["Low", "Medium", "High"]:
                p_val = "Medium"

            action_items.append(
                ActionItem(
                    task=item.get("task", "Untitled Task"),
                    assignee=item.get("assignee", "Unassigned"),
                    priority=PriorityEnum(p_val),
                    context_timestamp=item.get("context_timestamp"),
                    due_date=item.get("due_date")
                )
            )

        # Update cache in data/summaries/{video_id}.json if it exists
        base_dir = data_dir or Path("data")
        summary_cache = base_dir / "summaries" / f"{transcript.video_id}.json"
        if summary_cache.exists():
            try:
                with open(summary_cache, "r", encoding="utf-8") as f:
                    summary_dict = json.load(f)
                summary_obj = SummaryResult(**summary_dict)
                summary_obj.action_items = action_items
                with open(summary_cache, "w", encoding="utf-8") as f:
                    json.dump(summary_obj.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
            except Exception:
                pass

        return action_items
