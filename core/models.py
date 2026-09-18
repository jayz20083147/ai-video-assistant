"""Core data models for AI Video Assistant using Pydantic v2."""

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class PriorityEnum(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class AudioMetadata(BaseModel):
    """Metadata for processed audio files."""
    video_id: str = Field(..., description="SHA-256 hash identifying this media source")
    title: str = Field(..., description="Extracted video title or uploaded file name")
    duration_seconds: float = Field(..., description="Total duration of audio in seconds")
    file_path: Path = Field(..., description="Local path to 16kHz mono WAV file")
    source_url: Optional[str] = Field(None, description="Original YouTube URL if applicable")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def formatted_duration(self) -> str:
        """Returns duration formatted as HH:MM:SS or MM:SS."""
        total_secs = int(self.duration_seconds)
        hours, remainder = divmod(total_secs, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"


class TranscriptSegment(BaseModel):
    """Timestamped segment of a transcription."""
    id: int = Field(..., description="Zero-indexed segment order")
    start: float = Field(..., description="Start timestamp in seconds")
    end: float = Field(..., description="End timestamp in seconds")
    text: str = Field(..., description="Transcribed text content")
    speaker: Optional[str] = Field(None, description="Speaker identifier if diarized")

    @property
    def timestamp_str(self) -> str:
        """Formatted string representation [MM:SS - MM:SS]."""
        start_min, start_sec = divmod(int(self.start), 60)
        end_min, end_sec = divmod(int(self.end), 60)
        return f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"

    @property
    def start_timestamp_str(self) -> str:
        """Start timestamp [MM:SS]."""
        start_min, start_sec = divmod(int(self.start), 60)
        return f"{start_min:02d}:{start_sec:02d}"


class TranscriptionResult(BaseModel):
    """Full transcription result with metadata and segments."""
    video_id: str = Field(...)
    title: str = Field(...)
    duration_seconds: float = Field(...)
    language: str = Field(..., description="Detected or configured ISO language code")
    engine_used: str = Field(..., description="'whisper-local' or 'sarvam-api'")
    segments: list[TranscriptSegment] = Field(default_factory=list)
    full_text: str = Field(..., description="Concatenated raw transcript text")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ActionItem(BaseModel):
    """Structured action item extracted from meeting/video."""
    task: str = Field(..., description="Concrete, actionable task description")
    assignee: str = Field(default="Unassigned", description="Name of person responsible or role")
    priority: PriorityEnum = Field(default=PriorityEnum.MEDIUM, description="Task urgency/impact")
    context_timestamp: Optional[str] = Field(None, description="Timestamp where this was discussed, e.g. '14:20'")
    due_date: Optional[str] = Field(None, description="Extracted deadline if mentioned, or null")


class ChapterBreakdown(BaseModel):
    """Timestamped chapter summary."""
    title: str = Field(..., description="Chapter or topic heading")
    start_time: float = Field(..., description="Chapter start time in seconds")
    end_time: float = Field(..., description="Chapter end time in seconds")
    summary: str = Field(..., description="Concise synopsis of discussion in this time window")

    @property
    def time_range_str(self) -> str:
        start_min, start_sec = divmod(int(self.start_time), 60)
        end_min, end_sec = divmod(int(self.end_time), 60)
        return f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"


class SummaryResult(BaseModel):
    """Executive summary and analytical insights from transcript."""
    video_id: str = Field(...)
    executive_summary: str = Field(..., description="High-level 2-3 paragraph synthesis")
    key_takeaways: list[str] = Field(default_factory=list, description="Bullet points of key outcomes")
    chapters: list[ChapterBreakdown] = Field(default_factory=list, description="Timestamped sections")
    key_decisions: list[str] = Field(default_factory=list, description="List of decisions agreed upon")
    action_items: list[ActionItem] = Field(default_factory=list, description="Extracted actionable tasks")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChunkMetadata(BaseModel):
    """Metadata associated with vector store embeddings."""
    chunk_id: str = Field(..., description="Unique chunk identifier: {video_id}_c{index}")
    video_id: str = Field(...)
    start_time: float = Field(...)
    end_time: float = Field(...)
    start_timestamp_str: str = Field(...)
    end_timestamp_str: str = Field(...)
    segment_indices: list[int] = Field(..., description="List of Whisper segment IDs included in chunk")


class Citation(BaseModel):
    """Citation linking an assertion in RAG answer to video time range."""
    chunk_id: str = Field(...)
    text: str = Field(...)
    start_time: float = Field(...)
    end_time: float = Field(...)
    timestamp_str: str = Field(...)
    relevance_score: Optional[float] = Field(None)


class RAGResponse(BaseModel):
    """Structured response from the conversational RAG engine."""
    query: str = Field(...)
    answer: str = Field(..., description="Generated answer with embedded citation tags")
    citations: list[Citation] = Field(default_factory=list)
