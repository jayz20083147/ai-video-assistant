"""
Audio Ingestion, Extraction, and Normalization Utility (Phase 2).
Handles YouTube downloads via yt-dlp, canonical URL fingerprinting,
FFmpeg audio conversion to 16kHz mono WAV, SHA-256 caching, and audio chunking.
"""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import BinaryIO, Optional, Union
import wave

from core.models import AudioMetadata


class AudioProcessingError(Exception):
    """Raised when media download or FFmpeg processing fails."""
    pass


class AudioProcessor:
    """Manages audio extraction, normalization, and local file caching."""

    @staticmethod
    def extract_youtube_id(url: str) -> Optional[str]:
        """
        Extracts canonical 11-character YouTube video ID from various URL patterns:
        - https://www.youtube.com/watch?v=ID
        - https://youtu.be/ID
        - https://www.youtube.com/shorts/ID
        - https://www.youtube.com/embed/ID
        """
        if not isinstance(url, str):
            return None

        patterns = [
            r"(?:v=|\/v\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})",
            r"youtu\.be\/([0-9A-Za-z_-]{11})"
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    @classmethod
    def compute_hash(cls, source: Union[str, Path, bytes, BinaryIO]) -> str:
        """
        Computes deterministic SHA-256 hash for a URL string, local path, or raw file bytes.
        Normalizes YouTube URLs so different query parameters for the same video share the same hash.
        """
        hasher = hashlib.sha256()

        if isinstance(source, str):
            clean_str = source.strip()
            # If it's a YouTube URL, extract canonical video ID
            yt_id = cls.extract_youtube_id(clean_str)
            if yt_id:
                hasher.update(f"youtube:{yt_id}".encode("utf-8"))
                return hasher.hexdigest()

            if clean_str.startswith(("http://", "https://", "www.")):
                hasher.update(clean_str.encode("utf-8"))
                return hasher.hexdigest()
            elif os.path.exists(clean_str):
                source = Path(clean_str)
            else:
                hasher.update(clean_str.encode("utf-8"))
                return hasher.hexdigest()

        if isinstance(source, Path):
            if not source.exists():
                raise AudioProcessingError(f"Cannot compute hash: file does not exist: {source}")
            if source.stat().st_size == 0:
                raise AudioProcessingError(f"Cannot compute hash: file is empty (0 bytes): {source}")

            with open(source, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()

        if isinstance(source, bytes):
            if len(source) == 0:
                raise AudioProcessingError("Cannot compute hash: byte buffer is empty (0 bytes).")
            hasher.update(source)
            return hasher.hexdigest()

        # Handle file-like objects (e.g. Streamlit UploadedFile or BytesIO)
        if hasattr(source, "read") and hasattr(source, "seek"):
            current_pos = source.tell()
            source.seek(0, os.SEEK_END)
            size = source.tell()
            source.seek(0)
            if size == 0:
                raise AudioProcessingError("Cannot compute hash: file stream is empty (0 bytes).")

            for chunk in iter(lambda: source.read(65536), b""):
                hasher.update(chunk)
            source.seek(current_pos)
            return hasher.hexdigest()

        raise AudioProcessingError(f"Unsupported source type for hashing: {type(source)}")

    @classmethod
    def get_audio_duration(cls, file_path: Path) -> float:
        """
        Extracts duration of an audio/video file in seconds using ffprobe.
        Falls back to python wave module if ffprobe is unavailable and file is WAV.
        """
        if not file_path.exists():
            raise AudioProcessingError(f"Cannot get audio duration: file not found: {file_path}")

        ffprobe_bin = shutil.which("ffprobe")
        if ffprobe_bin:
            cmd = [
                ffprobe_bin,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(file_path)
            ]
            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                duration_str = result.stdout.strip()
                if duration_str and duration_str != "N/A":
                    return float(duration_str)
            except Exception:
                pass

        # Fallback for standard WAV files
        try:
            with wave.open(str(file_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return float(frames) / float(rate)
        except Exception:
            pass

        raise AudioProcessingError(f"Failed to probe audio duration for {file_path}")

    @classmethod
    def convert_to_wav(cls, input_path: Path, output_path: Path, sample_rate: int = 16000) -> Path:
        """
        Converts arbitrary media file into canonical 16kHz mono 16-bit PCM WAV using FFmpeg.
        Flags:
          -vn: strip video track
          -ac 1: convert to 1 channel (mono)
          -ar 16000: resample to 16,000 Hz
          -c:a pcm_s16le: signed 16-bit little-endian PCM
          -y: overwrite output without prompting
        """
        if not input_path.exists():
            raise AudioProcessingError(f"Source media file not found: {input_path}")
        if input_path.stat().st_size == 0:
            raise AudioProcessingError(f"Source media file is empty (0 bytes): {input_path}")

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise AudioProcessingError("ffmpeg is not installed or not found in system PATH.")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            ffmpeg_bin,
            "-i", str(input_path),
            "-vn",                       # Strip video
            "-ac", "1",                  # Mono channel
            "-ar", str(sample_rate),     # 16kHz sample rate
            "-c:a", "pcm_s16le",         # 16-bit PCM codec
            "-y",                        # Overwrite without asking
            str(output_path)
        ]

        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                raise AudioProcessingError(f"FFmpeg conversion failed: {res.stderr.strip()[:300]}")

            if not output_path.exists() or output_path.stat().st_size == 0:
                raise AudioProcessingError(f"FFmpeg completed but output file is missing or empty: {output_path}")

            return output_path
        except Exception as e:
            if isinstance(e, AudioProcessingError):
                raise e
            raise AudioProcessingError(f"Error during FFmpeg execution: {e}")

    @classmethod
    def download_youtube_audio(cls, url: str, output_dir: Path) -> tuple[Path, str, float]:
        """
        Downloads the best audio stream from a YouTube URL via yt-dlp.
        Uses anti-bot headers and format resilience.
        Returns: (downloaded_temp_path, video_title, duration_seconds)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check for python yt_dlp library first
        try:
            import yt_dlp
            has_py_module = True
        except ImportError:
            has_py_module = False

        template = str(output_dir / "yt_raw_%(id)s.%(ext)s")

        if has_py_module:
            import yt_dlp
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": template,
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                "socket_timeout": 30,
                "http_headers": {
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
                    )
                }
            }

            # Optional cookie file support
            cookie_candidates = [
                Path("cookies.txt"),
                Path.home() / "cookies.txt",
                output_dir / "cookies.txt"
            ]
            for c_path in cookie_candidates:
                if c_path.exists():
                    ydl_opts["cookiefile"] = str(c_path)
                    break

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    if not info:
                        raise AudioProcessingError(f"Could not extract video metadata for: {url}")

                    title = info.get("title", "YouTube Video")
                    duration = float(info.get("duration", 0.0))
                    v_id = info.get("id", "")
                    downloaded_file = Path(ydl.prepare_filename(info))

                    if not downloaded_file.exists():
                        matches = list(output_dir.glob(f"yt_raw_{v_id}*"))
                        if matches:
                            downloaded_file = matches[0]
                        else:
                            raise AudioProcessingError(f"Downloaded audio file not found on disk for video ID: {v_id}")

                    return downloaded_file, title, duration
            except Exception as e:
                raise AudioProcessingError(f"Failed to download YouTube audio via yt-dlp: {e}")

        # Fallback to system yt-dlp binary if available
        yt_bin = shutil.which("yt-dlp")
        if yt_bin:
            cmd = [
                yt_bin,
                "-f", "bestaudio/best",
                "-o", template,
                "--no-playlist",
                "--print", "%(title)s",
                "--print", "%(duration)s",
                "--print", "%(id)s",
                url
            ]
            try:
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                lines = [l.strip() for l in res.stdout.strip().splitlines() if l.strip()]
                title = lines[0] if len(lines) > 0 else "YouTube Video"
                duration = float(lines[1]) if len(lines) > 1 and lines[1].replace(".", "").isdigit() else 0.0
                v_id = lines[2] if len(lines) > 2 else ""

                matches = list(output_dir.glob(f"yt_raw_{v_id}*"))
                if matches:
                    return matches[0], title, duration
                raise AudioProcessingError(f"Downloaded audio file not found on disk for {url}")
            except Exception as e:
                raise AudioProcessingError(f"Failed to download YouTube audio via CLI: {e}")

        raise AudioProcessingError(
            "yt-dlp is required for YouTube audio extraction. Please install with: pip install yt-dlp"
        )

    @classmethod
    def split_audio_chunks(
        cls,
        audio_path: Path,
        chunk_duration_sec: int = 300,
        output_dir: Optional[Path] = None
    ) -> list[tuple[Path, float, float]]:
        """
        Splits long audio into smaller segments (e.g. 5 minutes) for API payload limits (e.g. Sarvam).
        Returns list of tuples: (chunk_file_path, start_time_sec, end_time_sec)
        """
        if not audio_path.exists():
            raise AudioProcessingError(f"Audio file to split not found: {audio_path}")

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise AudioProcessingError("ffmpeg is required for audio splitting.")

        total_duration = cls.get_audio_duration(audio_path)
        if total_duration <= chunk_duration_sec:
            return [(audio_path, 0.0, total_duration)]

        out_dir = output_dir or audio_path.parent / f"{audio_path.stem}_chunks"
        out_dir.mkdir(parents=True, exist_ok=True)

        chunks: list[tuple[Path, float, float]] = []
        start_time = 0.0
        index = 0

        while start_time < total_duration:
            duration = min(chunk_duration_sec, total_duration - start_time)
            chunk_file = out_dir / f"chunk_{index:03d}.wav"

            cmd = [
                ffmpeg_bin,
                "-ss", str(start_time),
                "-t", str(duration),
                "-i", str(audio_path),
                "-c", "copy",
                "-y",
                str(chunk_file)
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            chunks.append((chunk_file, start_time, start_time + duration))
            start_time += duration
            index += 1

        return chunks

    @classmethod
    def process_input(
        cls,
        source: Union[str, Path, BinaryIO],
        is_url: bool,
        data_dir: Path
    ) -> AudioMetadata:
        """
        High-level orchestrator that checks local cache, downloads/converts audio,
        and returns normalized AudioMetadata.
        """
        audio_dir = data_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        video_id = cls.compute_hash(source)
        canonical_wav = audio_dir / f"{video_id}.wav"
        meta_cache_path = audio_dir / f"{video_id}_meta.json"

        # Cache Hit Check
        if canonical_wav.exists() and meta_cache_path.exists():
            try:
                with open(meta_cache_path, "r", encoding="utf-8") as f:
                    cached_dict = json.load(f)
                    return AudioMetadata(**cached_dict)
            except Exception:
                # If cached metadata is corrupted, recompute
                pass

        title = "Uploaded Media"
        duration = 0.0
        source_url = None

        if is_url:
            source_url = str(source)
            temp_raw, title, duration = cls.download_youtube_audio(source_url, audio_dir)
            cls.convert_to_wav(temp_raw, canonical_wav)
            if temp_raw != canonical_wav and temp_raw.exists():
                temp_raw.unlink(missing_ok=True)
        else:
            # Local file path or file-like buffer
            if isinstance(source, (str, Path)) and Path(source).exists():
                local_path = Path(source)
                title = local_path.stem
                cls.convert_to_wav(local_path, canonical_wav)
            elif hasattr(source, "read"):
                title = getattr(source, "name", "Uploaded Audio")
                title = Path(title).stem
                temp_upload = audio_dir / f"temp_upload_{video_id[:8]}"
                with open(temp_upload, "wb") as f:
                    if hasattr(source, "seek"):
                        source.seek(0)
                    shutil.copyfileobj(source, f)
                try:
                    cls.convert_to_wav(temp_upload, canonical_wav)
                finally:
                    temp_upload.unlink(missing_ok=True)
            else:
                raise AudioProcessingError(f"Unprocessable source type: {type(source)}")

        duration = cls.get_audio_duration(canonical_wav)

        metadata = AudioMetadata(
            video_id=video_id,
            title=title,
            duration_seconds=duration,
            file_path=canonical_wav.resolve(),
            source_url=source_url
        )

        # Write metadata cache
        with open(meta_cache_path, "w", encoding="utf-8") as f:
            json.dump(metadata.model_dump(mode="json"), f, indent=2)

        return metadata
