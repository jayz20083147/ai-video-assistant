"""
Unit and Integration Tests for Phase 2: Audio Preprocessing Pipeline.
Tests canonical URL normalization, SHA-256 fingerprinting, FFmpeg conversion to 16kHz mono WAV,
duration probing, chunk splitting, and caching behavior.
"""

from io import BytesIO
import math
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
import wave

from utils.audio_processor import AudioProcessor, AudioProcessingError


class TestAudioProcessorPhase2(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_audio_p2_"))
        self.data_dir = self.temp_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_synthetic_wav(
        self,
        output_path: Path,
        duration_sec: float = 3.0,
        channels: int = 2,
        framerate: int = 44100
    ) -> Path:
        """Helper to create a valid sine-wave WAV file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        total_frames = int(framerate * duration_sec)

        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(framerate)

            frames = bytearray()
            for i in range(total_frames):
                val = int(28000.0 * math.sin(2.0 * math.pi * 440.0 * i / framerate))
                packed_sample = struct.pack("<h", val)
                for _ in range(channels):
                    frames.extend(packed_sample)
            wf.writeframes(frames)

        return output_path

    # -------------------------------------------------------------------------
    # 1. YouTube ID Extraction & URL Hashing Normalization
    # -------------------------------------------------------------------------
    def test_extract_youtube_id(self):
        """Test extraction of 11-char YouTube ID from various URL flavors."""
        sample_id = "dQw4w9WgXcQ"
        urls = [
            f"https://www.youtube.com/watch?v={sample_id}",
            f"https://youtube.com/watch?v={sample_id}&t=45s&feature=shared",
            f"https://youtu.be/{sample_id}",
            f"https://youtu.be/{sample_id}?t=10",
            f"https://www.youtube.com/shorts/{sample_id}",
            f"https://www.youtube.com/embed/{sample_id}"
        ]
        for url in urls:
            extracted = AudioProcessor.extract_youtube_id(url)
            self.assertEqual(extracted, sample_id, f"Failed to extract ID from {url}")

    def test_youtube_url_hashing_idempotency(self):
        """Verify that identical video with different query params maps to same hash."""
        url_watch = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        url_share = "https://youtu.be/dQw4w9WgXcQ?feature=share"
        url_time = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120s"

        hash1 = AudioProcessor.compute_hash(url_watch)
        hash2 = AudioProcessor.compute_hash(url_share)
        hash3 = AudioProcessor.compute_hash(url_time)

        self.assertEqual(hash1, hash2)
        self.assertEqual(hash2, hash3)
        self.assertEqual(len(hash1), 64)  # Valid SHA-256 length

    # -------------------------------------------------------------------------
    # 2. File & Stream Hashing
    # -------------------------------------------------------------------------
    def test_file_and_stream_hashing(self):
        """Test that hashing a file on disk equals hashing its BytesIO buffer."""
        wav_path = self.temp_dir / "sample.wav"
        self._create_synthetic_wav(wav_path, duration_sec=1.0)

        file_hash = AudioProcessor.compute_hash(wav_path)

        with open(wav_path, "rb") as f:
            raw_bytes = f.read()

        bytes_hash = AudioProcessor.compute_hash(raw_bytes)
        stream_hash = AudioProcessor.compute_hash(BytesIO(raw_bytes))

        self.assertEqual(file_hash, bytes_hash)
        self.assertEqual(bytes_hash, stream_hash)

    def test_empty_input_hashing_raises(self):
        """Empty files or byte buffers must raise AudioProcessingError."""
        empty_path = self.temp_dir / "empty.wav"
        empty_path.touch()

        with self.assertRaises(AudioProcessingError):
            AudioProcessor.compute_hash(empty_path)

        with self.assertRaises(AudioProcessingError):
            AudioProcessor.compute_hash(b"")

        with self.assertRaises(AudioProcessingError):
            AudioProcessor.compute_hash(BytesIO(b""))

    # -------------------------------------------------------------------------
    # 3. Canonical WAV Conversion & FFmpeg Validation
    # -------------------------------------------------------------------------
    def test_convert_to_wav_format_properties(self):
        """
        Ensure convert_to_wav outputs strict canonical format:
        16,000 Hz, 1-channel (mono), 16-bit PCM WAV.
        """
        stereo_48k = self.temp_dir / "stereo_48k.wav"
        self._create_synthetic_wav(stereo_48k, duration_sec=2.5, channels=2, framerate=48000)

        canonical_out = self.temp_dir / "canonical_out.wav"
        result_path = AudioProcessor.convert_to_wav(stereo_48k, canonical_out, sample_rate=16000)

        self.assertTrue(result_path.exists())
        self.assertEqual(result_path, canonical_out)

        # Inspect resulting WAV file headers
        with wave.open(str(result_path), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1, "Must be mono (1 channel)")
            self.assertEqual(wf.getframerate(), 16000, "Must be resampled to 16,000 Hz")
            self.assertEqual(wf.getsampwidth(), 2, "Must be 16-bit PCM (2 bytes per sample)")

    def test_convert_to_wav_missing_file_raises(self):
        """Converting a non-existent file must raise AudioProcessingError."""
        non_existent = self.temp_dir / "ghost_file.mp4"
        out_target = self.temp_dir / "ghost_out.wav"

        with self.assertRaises(AudioProcessingError):
            AudioProcessor.convert_to_wav(non_existent, out_target)

    # -------------------------------------------------------------------------
    # 4. Audio Duration Probing
    # -------------------------------------------------------------------------
    def test_get_audio_duration_accuracy(self):
        """Verify audio duration calculation matches expected length."""
        test_wav = self.temp_dir / "duration_test.wav"
        target_duration = 3.25
        self._create_synthetic_wav(test_wav, duration_sec=target_duration, framerate=16000, channels=1)

        probed_duration = AudioProcessor.get_audio_duration(test_wav)
        self.assertAlmostEqual(probed_duration, target_duration, delta=0.05)

    # -------------------------------------------------------------------------
    # 5. Audio Chunk Splitting
    # -------------------------------------------------------------------------
    def test_split_audio_chunks(self):
        """Test splitting long audio into fixed-duration chunks."""
        long_wav = self.temp_dir / "long_audio.wav"
        total_duration = 6.0
        self._create_synthetic_wav(long_wav, duration_sec=total_duration, framerate=16000, channels=1)

        # Split into 2-second chunks
        chunks = AudioProcessor.split_audio_chunks(long_wav, chunk_duration_sec=2)
        self.assertEqual(len(chunks), 3)

        # Verify time offsets
        self.assertAlmostEqual(chunks[0][1], 0.0, delta=0.1)
        self.assertAlmostEqual(chunks[0][2], 2.0, delta=0.1)
        self.assertAlmostEqual(chunks[1][1], 2.0, delta=0.1)
        self.assertAlmostEqual(chunks[1][2], 4.0, delta=0.1)
        self.assertAlmostEqual(chunks[2][1], 4.0, delta=0.1)
        self.assertAlmostEqual(chunks[2][2], 6.0, delta=0.1)

        for chunk_path, _, _ in chunks:
            self.assertTrue(chunk_path.exists())

    # -------------------------------------------------------------------------
    # 6. Ingestion & Cache Behavior
    # -------------------------------------------------------------------------
    def test_process_input_and_caching(self):
        """
        Verify process_input:
        1. Ingests local file and creates canonical WAV.
        2. Creates metadata JSON file in data/audio/.
        3. Subsequent invocation retrieves from cache without re-conversion.
        """
        source_wav = self.temp_dir / "source_meeting.wav"
        self._create_synthetic_wav(source_wav, duration_sec=2.0, channels=2, framerate=44100)

        meta1 = AudioProcessor.process_input(source_wav, is_url=False, data_dir=self.data_dir)
        self.assertTrue(meta1.file_path.exists())
        self.assertEqual(meta1.title, "source_meeting")
        self.assertAlmostEqual(meta1.duration_seconds, 2.0, delta=0.1)

        meta_cache_file = self.data_dir / "audio" / f"{meta1.video_id}_meta.json"
        self.assertTrue(meta_cache_file.exists())

        # Second invocation: should load from cache
        meta2 = AudioProcessor.process_input(source_wav, is_url=False, data_dir=self.data_dir)
        self.assertEqual(meta1.video_id, meta2.video_id)
        self.assertEqual(meta1.file_path, meta2.file_path)


if __name__ == "__main__":
    unittest.main()
