import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Add to path
sys.path.append(os.getcwd())

from utils.audio_processor import AudioProcessor
from core.transcriber import UnifiedTranscriber
from core.summarise import TranscriptSummarizer
from core.extractor import ActionItemExtractor
from core.vector_store import VectorStoreManager

def main():
    print("Starting pipeline debug...")
    
    # 1. Audio Processing
    test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw" 
    try:
        print("Processing audio...")
        meta = AudioProcessor.process_input(
            source=test_url,
            is_url=True,
            data_dir=Path("data")
        )
        print(f"Success! Audio meta: {meta.video_id}, {meta.duration_seconds}s")
    except Exception as e:
        print(f"FAILED AT AUDIO PROCESSOR: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Transcription
    try:
        print("Transcribing...")
        transcript = UnifiedTranscriber.transcribe(
            audio_path=meta.file_path,
            video_id=meta.video_id,
            title=meta.title,
            routing_mode="whisper",
            whisper_model_size="tiny"  # tiny for faster testing
        )
        print(f"Success! Transcript length: {len(transcript.segments)} segments")
    except Exception as e:
        print(f"FAILED AT TRANSCRIBER: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Summarization
    try:
        print("Summarizing...")
        summarizer = TranscriptSummarizer(mistral_api_key=os.getenv("MISTRAL_API_KEY", "DUMMY"))
        summary = summarizer.generate_summary(transcript)
        print(f"Success! Summary: {summary.executive_summary[:50]}...")
    except Exception as e:
        print(f"FAILED AT SUMMARIZER: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
