import pytest
from unittest.mock import patch
from core.models import TranscriptionResult, TranscriptSegment
from core.translator import ContentTranslator
from pathlib import Path

@pytest.fixture
def dummy_transcript():
    return TranscriptionResult(
        video_id="dummy_translate",
        title="Testing",
        duration_seconds=5.0,
        language="es",
        engine_used="whisper-local",
        segments=[
            TranscriptSegment(id=0, start=0.0, end=5.0, text="Hello mundo.")
        ],
        full_text="Hello mundo."
    )

@patch("core.translator.ContentTranslator._get_client")
def test_translator(mock_get_client, dummy_transcript, tmp_path):
    mock_client = mock_get_client.return_value
    mock_response = mock_client.chat.complete.return_value
    mock_response.choices[0].message.content = "[0] Hello world."
    
    translator = ContentTranslator(mistral_api_key="mock_key")
    res = translator.translate_transcript(dummy_transcript, target_lang="English")
    
    assert "Hello world." in res.full_text
    assert res.video_id == "dummy_translate"
    assert "English" in res.title
