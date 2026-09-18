"""
End-to-End Test Suite for AI Video Assistant Pipeline.
"""

import math
from pathlib import Path
import struct
import unittest
import wave

from core.models import (
    ActionItem,
    AudioMetadata,
    ChapterBreakdown,
    PriorityEnum,
    SummaryResult,
    TranscriptSegment,
    TranscriptionResult
)
from utils.audio_processor import AudioProcessor
from core.vector_store import VectorStoreManager
from utils.export_utils import MarkdownExporter, SRTExporter


class TestAIVideoAssistantPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("data/test_run")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.test_wav = self.test_dir / "synth_stereo_44k.wav"

        # Generate 2 seconds of 44.1kHz stereo audio
        with wave.open(str(self.test_wav), "w") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            for i in range(44100 * 2):
                sample = int(30000.0 * math.sin(2 * math.pi * 440.0 * i / 44100))
                wf.writeframes(struct.pack("<hh", sample, sample))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_audio_normalization(self):
        """Verify audio conversion to 16kHz mono 16-bit PCM WAV."""
        meta = AudioProcessor.process_input(
            source=self.test_wav,
            is_url=False,
            data_dir=self.test_dir
        )
        self.assertTrue(meta.file_path.exists())
        self.assertAlmostEqual(meta.duration_seconds, 2.0, delta=0.2)

        with wave.open(str(meta.file_path), "r") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getframerate(), 16000)
            self.assertEqual(wf.getsampwidth(), 2)

    def test_02_vector_chunking_integrity(self):
        """Verify segment-aware time windowing preserves timestamps and boundaries."""
        segments = [
            TranscriptSegment(id=0, start=0.0, end=10.0, text="Introductory remarks on architecture."),
            TranscriptSegment(id=1, start=10.5, end=25.0, text="Deep dive into ChromaDB vector indexing."),
            TranscriptSegment(id=2, start=25.5, end=40.0, text="Evaluating Mistral AI summarization latency."),
            TranscriptSegment(id=3, start=40.5, end=60.0, text="Concluding thoughts and action item delegation.")
        ]
        transcript = TranscriptionResult(
            video_id="test_suite_vid",
            title="Pipeline Test",
            duration_seconds=60.0,
            language="en",
            engine_used="whisper-test",
            segments=segments,
            full_text=" ".join(s.text for s in segments)
        )

        vsm = VectorStoreManager(persist_dir=self.test_dir / "chroma")
        chunks = vsm.create_chunks(transcript, target_word_count=10, overlap_segments=1)
        self.assertGreater(len(chunks), 1)

        # Check first chunk
        text0, meta0 = chunks[0]
        self.assertEqual(meta0.start_time, 0.0)
        self.assertEqual(meta0.start_timestamp_str, "00:00")
        self.assertTrue(len(text0) > 0)

    def test_03_export_utilities(self):
        """Verify Markdown and SRT export formats."""
        segments = [
            TranscriptSegment(id=0, start=0.0, end=12.34, text="Subtitle line one."),
            TranscriptSegment(id=1, start=12.5, end=25.67, text="Subtitle line two.")
        ]
        transcript = TranscriptionResult(
            video_id="test_suite_vid",
            title="Pipeline Test",
            duration_seconds=26.0,
            language="en",
            engine_used="whisper-test",
            segments=segments,
            full_text="Subtitle line one. Subtitle line two."
        )
        summary = SummaryResult(
            video_id="test_suite_vid",
            executive_summary="Test executive summary.",
            key_takeaways=["Key point 1"],
            chapters=[ChapterBreakdown(title="Ch 1", start_time=0.0, end_time=25.0, summary="Chapter overview")],
            key_decisions=["Decided to proceed"],
            action_items=[ActionItem(task="Run integration tests", assignee="Jay", priority=PriorityEnum.HIGH)]
        )

        srt_out = SRTExporter.export(transcript, output_path=self.test_dir / "test.srt")
        self.assertTrue(srt_out.exists())
        with open(srt_out, "r", encoding="utf-8") as f:
            srt_content = f.read()
        self.assertIn("00:00:00,000 --> 00:00:12,340", srt_content)
        self.assertIn("Subtitle line one.", srt_content)

        md_out = MarkdownExporter.export(summary, output_path=self.test_dir / "test.md")
        self.assertTrue(md_out.exists())
        with open(md_out, "r", encoding="utf-8") as f:
            md_content = f.read()
        self.assertIn("# Video Analysis Report", md_content)
        self.assertIn("Test executive summary.", md_content)
        self.assertIn("Run integration tests", md_content)


if __name__ == "__main__":
    unittest.main()
