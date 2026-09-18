# 🎙️ AI Video Assistant

> A free, locally-runnable alternative to **Otter.ai** and **Fireflies.ai** that processes YouTube videos or local media files into transcriptions, executive summaries, structured action items, and an interactive RAG-based chat interface with verified timestamp citations.

---

## ⚡ Tech Stack & Architecture

- **Language:** Python 3.10+
- **Media Ingestion & Normalization:** `yt-dlp` + FFmpeg (16kHz mono 16-bit PCM WAV)
- **Transcription (STT):**
  - **English / General:** OpenAI Whisper (`faster-whisper` CTranslate2 backend, 8-bit quantized)
  - **Hindi / Hinglish / Indic:** Sarvam AI REST API (`saaras:v1`)
- **Intelligence Layer:** Mistral AI (`mistral-small-latest`) for executive summarization and Pydantic-validated action item extraction
- **Embeddings:** HuggingFace local models (`sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`)
- **Vector Database:** ChromaDB (persistent local storage with metadata filtering)
- **UI & Dashboard:** Streamlit
- **Exports:** Pure Python PDF generation (`fpdf2`), SubRip subtitles (`.srt`), and GitHub-flavored Markdown

---

## 📁 Project Structure

```
ai-video-assistant/
├── .env.example                     # Sample environment variable template
├── .gitignore                       # Ignored storage, venv, and cache paths
├── requirements.txt                 # Pinned dependencies
├── README.md                        # Documentation & Quickstart
├── app.py                           # Streamlit UI application dashboard
├── core/
│   ├── __init__.py
│   ├── models.py                    # Pydantic v2 data models
│   ├── transcriber.py               # faster-whisper & Sarvam AI engine
│   ├── translator.py                # Mistral translation utilities
│   ├── summarise.py                 # Executive summarizer (Map-Reduce & direct)
│   ├── extractor.py                 # Structured action item & decision extractor
│   ├── vector_store.py              # ChromaDB + HuggingFace indexing
│   └── rag_engine.py                # Conversational RAG with timestamp citations
├── utils/
│   ├── __init__.py
│   ├── audio_processor.py           # yt-dlp & FFmpeg 16kHz audio converter
│   └── export_utils.py              # fpdf2 PDF report builder & SRT exporter
├── data/                            # Persistent local storage (gitignored)
│   ├── audio/                       # Normalized WAV files & audio cache
│   ├── transcripts/                 # Cached JSON transcript files
│   └── chroma_db/                   # Persistent Chroma vector store
└── exports/                         # Generated PDFs, Markdown, and SRT files
```

---

## 🚀 Quickstart Guide

### 1. Install System FFmpeg
Ensure `ffmpeg` and `ffprobe` are installed on your machine:

```bash
# macOS (Homebrew)
brew install ffmpeg

# Ubuntu / Debian Linux
sudo apt update && sudo apt install -y ffmpeg
```

Verify installation:
```bash
python3 scripts/verify_environment.py
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Credentials
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and fill in your API keys:
- `MISTRAL_API_KEY`: Free key from [Mistral AI Console](https://console.mistral.ai/)
- `SARVAM_API_KEY`: (Optional) Key from [Sarvam AI](https://www.sarvam.ai/) for Hindi/Hinglish speech recognition

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```

---

## 💡 Key Features & Workflow

1. **Deterministic Local Caching:**
   Every media file or YouTube link is fingerprinted via SHA-256. Re-running the same video skips downloading and transcription, loading summaries and vector indices instantly (< 1.5s).
2. **Segment-Aware Time Windowing:**
   Instead of splitting text arbitrarily, contiguous Whisper segments are grouped up to a token budget. This guarantees sentence integrity and produces exact `[MM:SS]` timestamp citations.
3. **Dual Transcription Backends:**
   - Automatically probes the first 30 seconds of audio.
   - If Hindi or Indic language is detected, routes to Sarvam AI (`saaras:v1`).
   - For English and all other languages, runs local `faster-whisper` using 8-bit quantization.
4. **Interactive RAG Chat:**
   Ask questions in natural language. The assistant retrieves relevant transcript excerpts and cites verifiable timestamps.
5. **Multi-Format Exporting:**
   - Executive PDF Meeting Reports via `fpdf2`.
   - Markdown meeting notes.
   - SubRip (`.srt`) subtitle tracks.
   - Raw JSON transcript with segment timestamps.
