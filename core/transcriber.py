"""
Multilingual Speech-to-Text Transcription Engine.
Supports local faster-whisper (with standard whisper fallback) and Sarvam AI REST API (Hindi/Hinglish),
featuring intelligent language routing and content-hashed transcript disk caching.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
import time
from typing import Optional
import requests
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError:
    def retry(*args, **kwargs):
        def decorator(func):
            def wrapper(*a, **kw):
                return func(*a, **kw)
            return wrapper
        return decorator
    def stop_after_attempt(n): return n
    def wait_exponential(*args, **kwargs): return None
    def retry_if_exception_type(*args): return None

from core.models import TranscriptSegment, TranscriptionResult
from utils.audio_processor import AudioProcessor


class TranscriptionError(Exception):
    """Raised when audio transcription encounters a failure."""
    pass


class BaseTranscriber(ABC):
    """Abstract interface for transcription backends."""

    @abstractmethod
    def transcribe(self, audio_path: Path, **kwargs) -> TranscriptionResult:
        """Transcribes audio file to standardized TranscriptionResult."""
        pass


class WhisperTranscriber(BaseTranscriber):
    """
    Local transcription engine using faster-whisper (CTranslate2) or openai-whisper.
    Automatically chooses optimal hardware device (CUDA -> MPS/CPU) and quantization.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "default"
    ) -> None:
        self.model_size = model_size
        self.device = self._resolve_device(device)
        self.compute_type = self._resolve_compute_type(compute_type)
        self._model = None

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            # Note: faster-whisper CTranslate2 executes on CPU / Apple Silicon via optimized CPU threads
            return "cpu"
        except ImportError:
            return "cpu"

    def _resolve_compute_type(self, compute_type: str) -> str:
        if compute_type != "default":
            return compute_type
        if self.device == "cuda":
            return "float16"
        return "int8"  # 8-bit quantized execution on CPU / Apple Silicon for 3x speed and lower RAM

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
            self._backend = "faster-whisper"
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=os.getenv("WHISPER_CACHE_DIR", None)
            )
            return self._model
        except ImportError:
            try:
                import whisper
                self._backend = "openai-whisper"
                self._model = whisper.load_model(self.model_size)
                return self._model
            except ImportError:
                raise TranscriptionError(
                    "Neither faster-whisper nor openai-whisper is installed. "
                    "Install with: pip install faster-whisper"
                )

    def detect_language(self, audio_path: Path) -> tuple[str, float]:
        """
        Probes the first 30 seconds of audio to detect language code and probability.
        """
        model = self._load_model()
        if self._backend == "faster-whisper":
            segments, info = model.transcribe(str(audio_path), beam_size=1, max_new_tokens=20)
            return info.language, info.language_probability
        else:
            import whisper
            audio = whisper.load_audio(str(audio_path))
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            _, probs = model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            return detected_lang, probs[detected_lang]

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        video_id: str = "unknown",
        title: str = "Audio Recording"
    ) -> TranscriptionResult:
        """
        Executes local transcription on 16kHz mono WAV file.
        """
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")

        duration = AudioProcessor.get_audio_duration(audio_path)
        model = self._load_model()
        segments_list: list[TranscriptSegment] = []
        full_text_parts: list[str] = []
        detected_language = language or "en"

        try:
            if self._backend == "faster-whisper":
                segments_iter, info = model.transcribe(
                    str(audio_path),
                    language=language,
                    beam_size=5,
                    word_timestamps=False,
                    no_speech_threshold=0.6,
                    log_prob_threshold=-1.0,
                    condition_on_previous_text=False
                )
                detected_language = info.language

                for i, seg in enumerate(segments_iter):
                    text_clean = seg.text.strip()
                    if text_clean:
                        segments_list.append(
                            TranscriptSegment(
                                id=i,
                                start=round(seg.start, 2),
                                end=round(seg.end, 2),
                                text=text_clean
                            )
                        )
                        full_text_parts.append(text_clean)
            else:
                # openai-whisper fallback
                result = model.transcribe(
                    str(audio_path),
                    language=language,
                    no_speech_threshold=0.6,
                    logprob_threshold=-1.0,
                    condition_on_previous_text=False
                )
                detected_language = result.get("language", "en")
                for i, seg in enumerate(result.get("segments", [])):
                    text_clean = seg.get("text", "").strip()
                    if text_clean:
                        segments_list.append(
                            TranscriptSegment(
                                id=i,
                                start=round(seg.get("start", 0.0), 2),
                                end=round(seg.get("end", 0.0), 2),
                                text=text_clean
                            )
                        )
                        full_text_parts.append(text_clean)

            return TranscriptionResult(
                video_id=video_id,
                title=title,
                duration_seconds=duration,
                language=detected_language,
                engine_used=f"{self._backend}-{self.model_size}",
                segments=segments_list,
                full_text=" ".join(full_text_parts)
            )

        except Exception as e:
            raise TranscriptionError(f"Whisper transcription failed: {e}")


class SarvamTranscriber(BaseTranscriber):
    """
    Transcription engine for Indic languages (Hindi/Hinglish) using Sarvam AI REST API.
    Handles payload limits by auto-chunking long audio files.
    """

    API_URL = "https://api.sarvam.ai/speech-to-text"

    def __init__(self, api_key: str, model: str = "saaras:v1") -> None:
        if not api_key:
            raise TranscriptionError("Sarvam AI API key is required.")
        self.api_key = api_key
        self.model = model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type(requests.RequestException)
    )
    def _transcribe_chunk(self, chunk_path: Path, language_code: str) -> dict:
        headers = {
            "api-subscription-key": self.api_key
        }
        data = {
            "model": self.model,
            "language_code": language_code
        }

        with open(chunk_path, "rb") as audio_file:
            files = {"file": (chunk_path.name, audio_file, "audio/wav")}
            response = requests.post(self.API_URL, headers=headers, data=data, files=files, timeout=60)

        if response.status_code != 200:
            raise TranscriptionError(
                f"Sarvam AI API Error [{response.status_code}]: {response.text}"
            )
        return response.json()

    def transcribe(
        self,
        audio_path: Path,
        language_code: str = "hi-IN",
        video_id: str = "unknown",
        title: str = "Audio Recording"
    ) -> TranscriptionResult:
        """
        Transcribes audio with Sarvam AI API, handling long audio by splitting into 5-minute chunks.
        """
        duration = AudioProcessor.get_audio_duration(audio_path)
        chunks = AudioProcessor.split_audio_chunks(audio_path, chunk_duration_sec=300)

        segments_list: list[TranscriptSegment] = []
        full_text_parts: list[str] = []
        segment_id = 0

        for chunk_file, start_offset, end_offset in chunks:
            chunk_result = self._transcribe_chunk(chunk_file, language_code)
            transcript_text = chunk_result.get("transcript", "").strip()

            # Sarvam returns timestamps or block transcript
            sub_segments = chunk_result.get("timestamps", [])
            if sub_segments:
                for sub in sub_segments:
                    seg_text = sub.get("text", "").strip()
                    if seg_text:
                        s_time = start_offset + float(sub.get("start_time_seconds", 0.0))
                        e_time = start_offset + float(sub.get("end_time_seconds", end_offset - start_offset))
                        segments_list.append(
                            TranscriptSegment(
                                id=segment_id,
                                start=round(s_time, 2),
                                end=round(e_time, 2),
                                text=seg_text
                            )
                        )
                        full_text_parts.append(seg_text)
                        segment_id += 1
            else:
                # Chunk-level segment
                if transcript_text:
                    segments_list.append(
                        TranscriptSegment(
                            id=segment_id,
                            start=round(start_offset, 2),
                            end=round(end_offset, 2),
                            text=transcript_text
                        )
                    )
                    full_text_parts.append(transcript_text)
                    segment_id += 1

            # Clean up temporary chunk if created
            if chunk_file != audio_path and chunk_file.exists():
                chunk_file.unlink(missing_ok=True)

        return TranscriptionResult(
            video_id=video_id,
            title=title,
            duration_seconds=duration,
            language=language_code,
            engine_used=f"sarvam-{self.model}",
            segments=segments_list,
            full_text=" ".join(full_text_parts)
        )


class UnifiedTranscriber:
    """
    Coordinates transcription routing and local filesystem caching.
    """

    @classmethod
    def transcribe(
        cls,
        audio_path: Path,
        video_id: str,
        title: str,
        routing_mode: str = "auto",  # 'auto' | 'whisper' | 'sarvam'
        whisper_model_size: str = "small",
        sarvam_api_key: Optional[str] = None,
        data_dir: Optional[Path] = None
    ) -> TranscriptionResult:
        """
        Executes transcription with disk caching and intelligent language routing.
        """
        base_dir = data_dir or Path("data")
        transcripts_dir = base_dir / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        cache_file = transcripts_dir / f"{video_id}.json"

        # Check Cache
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    return TranscriptionResult(**cached_data)
            except Exception:
                pass

        whisper_engine = WhisperTranscriber(model_size=whisper_model_size)
        target_backend = routing_mode.lower()

        if target_backend == "auto":
            # Probe first 30 seconds for language identification
            try:
                detected_lang, prob = whisper_engine.detect_language(audio_path)
                if detected_lang in ["hi", "mr", "gu", "pa"] and sarvam_api_key:
                    target_backend = "sarvam"
                else:
                    target_backend = "whisper"
            except Exception:
                target_backend = "whisper"

        if target_backend == "sarvam":
            if not sarvam_api_key:
                # Fallback to local Whisper if Sarvam API key not supplied
                result = whisper_engine.transcribe(audio_path, video_id=video_id, title=title)
            else:
                sarvam_engine = SarvamTranscriber(api_key=sarvam_api_key)
                result = sarvam_engine.transcribe(audio_path, language_code="hi-IN", video_id=video_id, title=title)
        else:
            result = whisper_engine.transcribe(audio_path, video_id=video_id, title=title)

        # Write to disk cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

        return result
