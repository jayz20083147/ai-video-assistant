import pytest
from unittest.mock import patch, MagicMock
from core.models import TranscriptionResult, TranscriptSegment, SummaryResult, ActionItem
from core.summarise import TranscriptSummarizer, SummarizationError
from core.extractor import ActionItemExtractor
from pathlib import Path
import json

@pytest.fixture
def dummy_transcript():
    return TranscriptionResult(
        video_id="dummy_video_001",
        title="Quarterly Review",
        duration_seconds=30.0,
        language="en",
        engine_used="whisper-local",
        segments=[
            TranscriptSegment(id=0, start=0.0, end=10.0, text="Hello and welcome to the quarterly review."),
            TranscriptSegment(id=1, start=10.0, end=20.0, text="Alice, can you prepare the slides for next week?"),
            TranscriptSegment(id=2, start=20.0, end=30.0, text="Sure thing. Also, we decided to migrate to Python 3.12."),
        ],
        full_text="Hello and welcome to the quarterly review. Alice, can you prepare the slides for next week? Sure thing. Also, we decided to migrate to Python 3.12."
    )

@patch("core.summarise.TranscriptSummarizer._call_llm")
def test_summarization_parsing(mock_call_llm, dummy_transcript, tmp_path):
    mock_response = {
        "executive_summary": "Quarterly review kickoff.",
        "key_takeaways": ["Python 3.12 migration"],
        "key_decisions": ["Migrate to Python 3.12"],
        "chapters": [
            {"title": "Intro", "start_time": 0.0, "end_time": 30.0, "summary": "Intro and decisions."}
        ]
    }
    mock_call_llm.return_value = "```json\n" + json.dumps(mock_response) + "\n```"

    summarizer = TranscriptSummarizer(mistral_api_key="mock_key")
    result = summarizer.generate_summary(dummy_transcript, data_dir=tmp_path)

    assert result.video_id == "dummy_video_001"
    assert result.executive_summary == "Quarterly review kickoff."
    assert "Python 3.12 migration" in result.key_takeaways
    assert len(result.chapters) == 1
    assert result.chapters[0].title == "Intro"

    # Check cache was created
    cache_file = tmp_path / "summaries" / "dummy_video_001.json"
    assert cache_file.exists()

@patch("core.extractor.ActionItemExtractor._call_llm")
def test_action_item_extraction(mock_call_llm, dummy_transcript, tmp_path):
    mock_response = {
        "action_items": [
            {
                "task": "Prepare the slides",
                "assignee": "Alice",
                "due_date": "next week",
                "context": "Quarterly review prep"
            }
        ]
    }
    mock_call_llm.return_value = json.dumps(mock_response)

    extractor = ActionItemExtractor(mistral_api_key="mock_key")
    items = extractor.extract(dummy_transcript, data_dir=tmp_path)

    assert len(items) == 1
    assert items[0].task == "Prepare the slides"
    assert items[0].assignee == "Alice"
    assert items[0].due_date == "next week"
