import pytest
from unittest.mock import patch, MagicMock
from core.models import TranscriptionResult, TranscriptSegment, Citation
from core.vector_store import VectorStoreManager
from core.rag_engine import VideoRAGEngine
from pathlib import Path

@pytest.fixture
def dummy_transcript():
    return TranscriptionResult(
        video_id="dummy_rag",
        title="RAG Testing",
        duration_seconds=15.0,
        language="en",
        engine_used="whisper-local",
        segments=[
            TranscriptSegment(id=0, start=0.0, end=5.0, text="First sentence."),
            TranscriptSegment(id=1, start=5.0, end=10.0, text="Second sentence."),
            TranscriptSegment(id=2, start=10.0, end=15.0, text="Third sentence.")
        ],
        full_text="First sentence. Second sentence. Third sentence."
    )

@patch("core.vector_store.VectorStoreManager._get_chroma_client")
def test_vector_store_indexing(mock_get_chroma, dummy_transcript, tmp_path):
    mock_collection = MagicMock()
    mock_client_instance = MagicMock()
    mock_client_instance.create_collection.return_value = mock_collection
    mock_get_chroma.return_value = mock_client_instance
    
    vs = VectorStoreManager(persist_dir=tmp_path / "chroma")
    chunks = vs.create_chunks(dummy_transcript, target_word_count=4)
    assert len(chunks) == 2 or len(chunks) == 1

@patch("core.rag_engine.VideoRAGEngine._call_llm")
def test_rag_engine_query(mock_call_llm):
    mock_call_llm.return_value = "It means first sentence. [00:00 - 00:05]"
    
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = [
        Citation(chunk_id="c1", text="First sentence.", start_time=0.0, end_time=5.0, timestamp_str="00:00 - 00:05", relevance_score=0.9)
    ]
    
    rag = VideoRAGEngine(vector_store_manager=mock_vs, mistral_api_key="mock")
    response = rag.ask("What is first?", video_id="dummy")
    
    assert "It means first sentence." in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].text == "First sentence."
