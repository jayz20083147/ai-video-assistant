import pytest
from unittest.mock import patch, MagicMock
from core.models import TranscriptionResult, TranscriptSegment
from core.transcriber import UnifiedTranscriber
from pathlib import Path
import json

@patch("core.transcriber.WhisperTranscriber.transcribe")
def test_unified_transcriber_whisper(mock_whisper, tmp_path):
    mock_whisper.return_value = TranscriptionResult(
        video_id="vid_w1",
        title="Unknown",
        segments=[TranscriptSegment(start_time=0.0, end_time=1.0, text="Whisper test")],
        full_text="Whisper test"
    )
    
    transcriber = UnifiedTranscriber(engine="whisper")
    res = transcriber.transcribe("dummy.wav", video_id="vid_w1", data_dir=tmp_path)
    
    assert res.full_text == "Whisper test"
    assert res.video_id == "vid_w1"
    
    # Check cache was created
    cache_file = tmp_path / "transcripts" / "vid_w1.json"
    assert cache_file.exists()
    
@patch("core.transcriber.SarvamTranscriber.transcribe")
def test_unified_transcriber_sarvam(mock_sarvam, tmp_path):
    mock_sarvam.return_value = TranscriptionResult(
        video_id="vid_s1",
        title="Unknown",
        segments=[TranscriptSegment(start_time=0.0, end_time=1.0, text="Sarvam test")],
        full_text="Sarvam test"
    )
    
    transcriber = UnifiedTranscriber(engine="sarvam", sarvam_api_key="mock_key")
    res = transcriber.transcribe("dummy.wav", video_id="vid_s1", data_dir=tmp_path)
    
    assert res.full_text == "Sarvam test"

