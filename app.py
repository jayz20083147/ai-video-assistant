"""
AI Video Assistant — Free, Local Alternative to Otter.ai and Fireflies.ai
Streamlit Application Dashboard.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv(PROJECT_ROOT / ".env")

from core.models import AudioMetadata, SummaryResult, TranscriptionResult
from utils.audio_processor import AudioProcessor, AudioProcessingError
from core.transcriber import UnifiedTranscriber, TranscriptionError
from core.summarise import TranscriptSummarizer, SummarizationError
from core.extractor import ActionItemExtractor, ExtractionError
from core.vector_store import VectorStoreManager
from core.rag_engine import VideoRAGEngine, RAGQueryError
from utils.export_utils import PDFReportGenerator, MarkdownExporter, SRTExporter

# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        background-color: #F1F5F9;
        border-radius: 8px;
        padding: 8px 14px;
        display: inline-block;
        margin-right: 10px;
        font-size: 0.9rem;
        font-weight: 500;
        color: #334155;
    }
    .timestamp-badge {
        background-color: #E0E7FF;
        color: #3730A3;
        font-family: monospace;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .chapter-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
    .priority-high {
        color: #DC2626;
        font-weight: 600;
    }
    .priority-med {
        color: #D97706;
        font-weight: 600;
    }
    .priority-low {
        color: #2563EB;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "metadata" not in st.session_state:
    st.session_state.metadata = None
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None


# -----------------------------------------------------------------------------
# Cached Singletons
# -----------------------------------------------------------------------------
@st.cache_resource
def get_vector_store_manager() -> VectorStoreManager:
    persist_dir = PROJECT_ROOT / "data" / "chroma_db"
    return VectorStoreManager(persist_dir=persist_dir)


# -----------------------------------------------------------------------------
# Sidebar: Configuration & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎙️ AI Video Assistant")
    st.caption("Free, local meeting & video intelligence alternative to Otter.ai & Fireflies.")
    st.divider()

    # Ingestion Source Selector
    st.markdown("#### 1. Media Source")
    input_mode = st.radio("Choose Input Type:", ["YouTube URL", "Upload Media File"], index=0)

    url_input = ""
    uploaded_file = None

    if input_mode == "YouTube URL":
        url_input = st.text_input(
            "YouTube Link:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Supports YouTube videos, podcasts, and shorts."
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload audio or video:",
            type=["mp4", "mp3", "wav", "m4a", "mov", "webm", "ogg"],
            help="File will be normalized to 16kHz mono WAV locally."
        )

    st.divider()

    # Engine Settings
    st.markdown("#### 2. Engine & Model Settings")
    routing_mode = st.selectbox(
        "Language & STT Routing:",
        ["Auto-Detect (Hybrid)", "Local Whisper (English / Global)", "Sarvam AI (Hindi / Hinglish)"],
        index=0,
        help="Auto-Detect routes Hindi audio to Sarvam AI if configured, otherwise uses local Whisper."
    )

    whisper_model = st.selectbox(
        "Whisper Model Size:",
        ["small (Recommended)", "base (Faster)", "tiny (Fastest)"],
        index=0,
        help="'small' delivers high accuracy on technical terminology with low memory."
    )
    whisper_size_code = whisper_model.split()[0]

    # API Keys & Health
    with st.expander("🔑 API Key Settings", expanded=False):
        env_mistral = os.getenv("MISTRAL_API_KEY", "")
        env_sarvam = os.getenv("SARVAM_API_KEY", "")

        mistral_key_input = st.text_input("Mistral AI Key:", value=env_mistral, type="password")
        sarvam_key_input = st.text_input("Sarvam AI Key (Optional):", value=env_sarvam, type="password")

        if mistral_key_input:
            os.environ["MISTRAL_API_KEY"] = mistral_key_input
        if sarvam_key_input:
            os.environ["SARVAM_API_KEY"] = sarvam_key_input

    st.divider()

    # Processing Trigger Button
    process_btn = st.button("🚀 Process Video", type="primary", use_container_width=True)

    # Cache & Storage Utilities
    with st.expander("💾 Storage & Cache Info", expanded=False):
        audio_count = len(list((PROJECT_ROOT / "data" / "audio").glob("*.wav")))
        transcript_count = len(list((PROJECT_ROOT / "data" / "transcripts").glob("*.json")))
        st.write(f"- Indexed Videos: **{transcript_count}**")
        st.write(f"- Cached Audio Files: **{audio_count}**")

        if st.button("🧹 Clear Current Video Cache", use_container_width=True):
            if st.session_state.video_id:
                v_id = st.session_state.video_id
                (PROJECT_ROOT / "data" / "audio" / f"{v_id}.wav").unlink(missing_ok=True)
                (PROJECT_ROOT / "data" / "transcripts" / f"{v_id}.json").unlink(missing_ok=True)
                (PROJECT_ROOT / "data" / "summaries" / f"{v_id}.json").unlink(missing_ok=True)
                st.session_state.video_id = None
                st.session_state.metadata = None
                st.session_state.transcript = None
                st.session_state.summary = None
                st.session_state.chat_history = []
                st.success("Cache cleared for current video.")
                st.rerun()


# -----------------------------------------------------------------------------
# Main Panel: Header & Execution Flow
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🎙️ AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Transcribe, summarize, extract action items, and interrogate videos with verifiable timestamp citations.</div>',
    unsafe_allow_html=True
)

# Execution Logic when "Process Video" is clicked
if process_btn:
    has_source = bool(url_input.strip()) if input_mode == "YouTube URL" else (uploaded_file is not None)

    if not has_source:
        st.warning("⚠️ Please provide a YouTube link or upload a media file.")
    elif not os.getenv("MISTRAL_API_KEY"):
        st.error("❌ Mistral AI API key is missing. Please set MISTRAL_API_KEY in the sidebar or in your .env file.")
    else:
        source_target = url_input.strip() if input_mode == "YouTube URL" else uploaded_file
        is_url = (input_mode == "YouTube URL")

        # Map routing dropdown to code
        route_code = "auto"
        if "Sarvam" in routing_mode:
            route_code = "sarvam"
        elif "Whisper" in routing_mode:
            route_code = "whisper"

        with st.status("Processing media pipeline...", expanded=True) as status_box:
            try:
                # Stage 1: Audio Extraction & Normalization
                status_box.update(label="Step 1/4: Ingesting & normalizing audio (16kHz mono WAV)...", state="running")
                data_dir = PROJECT_ROOT / "data"
                metadata: AudioMetadata = AudioProcessor.process_input(
                    source=source_target,
                    is_url=is_url,
                    data_dir=data_dir
                )
                st.session_state.metadata = metadata
                st.session_state.video_id = metadata.video_id

                # Stage 2: Multilingual Speech-to-Text
                status_box.update(label="Step 2/4: Transcribing speech with timestamps...", state="running")
                transcript: TranscriptionResult = UnifiedTranscriber.transcribe(
                    audio_path=metadata.file_path,
                    video_id=metadata.video_id,
                    title=metadata.title,
                    routing_mode=route_code,
                    whisper_model_size=whisper_size_code,
                    sarvam_api_key=os.getenv("SARVAM_API_KEY"),
                    data_dir=data_dir
                )
                st.session_state.transcript = transcript

                # Stage 3: Intelligence Processing (Summarization & Extraction)
                status_box.update(label="Step 3/4: Generating executive summary & action items via Mistral AI...", state="running")
                summarizer = TranscriptSummarizer(mistral_api_key=os.getenv("MISTRAL_API_KEY"))
                summary_res: SummaryResult = summarizer.generate_summary(transcript=transcript, data_dir=data_dir)

                extractor = ActionItemExtractor(mistral_api_key=os.getenv("MISTRAL_API_KEY"))
                action_items = extractor.extract(transcript=transcript, data_dir=data_dir)
                summary_res.action_items = action_items
                st.session_state.summary = summary_res

                # Stage 4: Vector Indexing & RAG Preparation
                status_box.update(label="Step 4/4: Chunking and indexing timestamped segments in ChromaDB...", state="running")
                vsm = get_vector_store_manager()
                vsm.index_transcript(transcript)

                rag_engine = VideoRAGEngine(
                    vector_store_manager=vsm,
                    mistral_api_key=os.getenv("MISTRAL_API_KEY")
                )
                st.session_state.rag_engine = rag_engine

                status_box.update(label="✅ Processing Complete! All insights and RAG index are ready.", state="complete")
                st.session_state.chat_history = []
                time.sleep(0.5)
                st.rerun()

            except AudioProcessingError as ape:
                status_box.update(label="Audio Processing Failed", state="error")
                st.error(f"Audio Pipeline Error: {ape}")
            except TranscriptionError as te:
                status_box.update(label="Transcription Failed", state="error")
                st.error(f"Transcription Error: {te}")
            except SummarizationError as se:
                status_box.update(label="Summarization Failed", state="error")
                st.error(f"Summarization Error: {se}")
            except Exception as ex:
                status_box.update(label="Pipeline Failed", state="error")
                st.error(f"Unexpected Pipeline Error: {ex}")


# -----------------------------------------------------------------------------
# Results Presentation (Tabs)
# -----------------------------------------------------------------------------
if st.session_state.metadata and st.session_state.summary:
    meta = st.session_state.metadata
    summary = st.session_state.summary
    transcript = st.session_state.transcript

    # Video Metadata Card
    with st.container():
        col_m1, col_m2 = st.columns([3, 1])
        with col_m1:
            st.markdown(f"### 📹 {meta.title}")
            st.markdown(
                f'<span class="metric-badge">⏱️ Duration: <b>{meta.formatted_duration}</b></span>'
                f'<span class="metric-badge">🌐 Language: <b>{transcript.language.upper()}</b></span>'
                f'<span class="metric-badge">⚡ Engine: <b>{transcript.engine_used}</b></span>',
                unsafe_allow_html=True
            )
        with col_m2:
            if meta.file_path.exists():
                st.audio(str(meta.file_path), format="audio/wav")

    st.write("")

    # Main Tabs Interface
    tab_summary, tab_tasks, tab_transcript, tab_chat, tab_export = st.tabs([
        "📋 Executive Summary",
        "✅ Action Items & Decisions",
        "📝 Interactive Transcript",
        "💬 Ask Video (RAG)",
        "📤 Export & Reports"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: EXECUTIVE SUMMARY
    # -------------------------------------------------------------------------
    with tab_summary:
        st.markdown("#### Executive Brief")
        st.write(summary.executive_summary)
        st.divider()

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("#### 💡 Key Takeaways")
            if summary.key_takeaways:
                for item in summary.key_takeaways:
                    st.markdown(f"- {item}")
            else:
                st.info("No specific takeaways generated.")

        with col_t2:
            st.markdown("#### ⚖️ Key Decisions Made")
            if summary.key_decisions:
                for dec in summary.key_decisions:
                    st.markdown(f"- **{dec}**")
            else:
                st.info("No explicit consensus decisions flagged.")

        st.divider()
        st.markdown("#### 📑 Timestamped Chapter Breakdown")
        if summary.chapters:
            for ch in summary.chapters:
                st.markdown(f"""
                <div class="chapter-card">
                    <span class="timestamp-badge">{ch.time_range_str}</span> <b>{ch.title}</b>
                    <p style="margin-top: 6px; margin-bottom: 0; color: #475569;">{ch.summary}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No chapter breakdown available.")

    # -------------------------------------------------------------------------
    # TAB 2: ACTION ITEMS & DECISIONS
    # -------------------------------------------------------------------------
    with tab_tasks:
        st.markdown("#### 📌 Action Items & Next Steps")
        if summary.action_items:
            # Render as clean table / card list
            for i, act in enumerate(summary.action_items, 1):
                p_class = "priority-med"
                if act.priority.value == "High":
                    p_class = "priority-high"
                elif act.priority.value == "Low":
                    p_class = "priority-low"

                time_tag = f'<span class="timestamp-badge">⏱️ {act.context_timestamp}</span>' if act.context_timestamp else ''
                due_tag = f" <i>(Due: {act.due_date})</i>" if act.due_date else ""

                st.markdown(f"""
                <div style="background:#FFFFFF; border:1px solid #E2E8F0; padding:12px; border-radius:8px; margin-bottom:8px;">
                    <b>{i}. {act.task}</b>{due_tag}<br>
                    <span style="font-size:0.88rem; color:#64748B;">
                        Assignee: <b>{act.assignee}</b> | Priority: <span class="{p_class}">{act.priority.value}</span> {time_tag}
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No actionable tasks were detected in this transcript.")

    # -------------------------------------------------------------------------
    # TAB 3: INTERACTIVE TRANSCRIPT
    # -------------------------------------------------------------------------
    with tab_transcript:
        col_s1, col_s2 = st.columns([3, 1])
        with col_s1:
            search_query = st.text_input("🔍 Search transcript:", placeholder="Type to filter spoken text...")
        with col_s2:
            st.write("")
            st.write("")
            st.download_button(
                "📥 Download Raw (.txt)",
                data=transcript.full_text,
                file_name=f"{meta.video_id}_transcript.txt",
                mime="text/plain",
                use_container_width=True
            )

        st.markdown("---")
        filtered_segments = transcript.segments
        if search_query.strip():
            q_lower = search_query.strip().lower()
            filtered_segments = [s for s in filtered_segments if q_lower in s.text.lower()]
            st.caption(f"Showing {len(filtered_segments)} matches for '{search_query}'")

        # Scrollable container for segments
        transcript_container = st.container(height=500)
        with transcript_container:
            for seg in filtered_segments:
                st.markdown(
                    f'<span class="timestamp-badge">[{seg.start_timestamp_str}]</span> {seg.text}',
                    unsafe_allow_html=True
                )

    # -------------------------------------------------------------------------
    # TAB 4: ASK VIDEO (CONVERSATIONAL RAG)
    # -------------------------------------------------------------------------
    with tab_chat:
        st.markdown("#### 💬 Ask Questions About This Video")
        st.caption("Answers are strictly grounded in indexed transcript segments and cite exact timestamp ranges.")

        # Quick Suggestion Pills
        col_p1, col_p2, col_p3 = st.columns(3)
        sample_q = None
        if col_p1.button("📌 What were the key decisions?", use_container_width=True):
            sample_q = "What were the key decisions made in this video?"
        if col_p2.button("📋 Summarize main action items", use_container_width=True):
            sample_q = "Summarize the primary action items and assignees."
        if col_p3.button("❓ What was the conclusion?", use_container_width=True):
            sample_q = "What was the final conclusion or summary statement?"

        # Display Chat History
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "citations" in msg and msg["citations"]:
                    with st.expander("🔎 View Source Video Citations"):
                        for c in msg["citations"]:
                            st.markdown(f"**[{c.get('timestamp_str', '')}]**: *\"{c.get('text', '')}\"*")

        # User Question Input
        user_prompt = st.chat_input("Ask anything about this video...")
        active_query = sample_q or user_prompt

        if active_query:
            # Display user message
            st.session_state.chat_history.append({"role": "user", "content": active_query})
            with st.chat_message("user"):
                st.markdown(active_query)

            # Generate RAG response
            with st.chat_message("assistant"):
                with st.spinner("Searching video transcript and generating cited answer..."):
                    if not st.session_state.rag_engine:
                        vsm = get_vector_store_manager()
                        st.session_state.rag_engine = VideoRAGEngine(
                            vector_store_manager=vsm,
                            mistral_api_key=os.getenv("MISTRAL_API_KEY")
                        )

                    try:
                        rag_response = st.session_state.rag_engine.ask(
                            question=active_query,
                            video_id=st.session_state.video_id,
                            chat_history=st.session_state.chat_history
                        )
                        st.markdown(rag_response.answer)

                        # Citations expander
                        if rag_response.citations:
                            with st.expander("🔎 View Source Video Citations"):
                                for c in rag_response.citations:
                                    st.markdown(f"**[{c.timestamp_str}]**: *\"{c.text}\"*")

                        # Append to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": rag_response.answer,
                            "citations": [c.model_dump() for c in rag_response.citations]
                        })

                    except Exception as e:
                        st.error(f"RAG Retrieval Error: {e}")

    # -------------------------------------------------------------------------
    # TAB 5: EXPORT & REPORTS
    # -------------------------------------------------------------------------
    with tab_export:
        st.markdown("#### 📤 Export Meeting Notes & Transcripts")
        st.write("Generate branded PDF reports, SubRip subtitles (.srt) for video editing, or clean Markdown files.")

        col_e1, col_e2, col_e3 = st.columns(3)

        with col_e1:
            st.markdown("##### 📄 PDF Meeting Report")
            st.caption("Formatted briefing with summary, action items table, and chapters.")
            if st.button("Generate PDF", use_container_width=True):
                with st.spinner("Compiling PDF report..."):
                    pdf_file = PDFReportGenerator.generate(summary, meta)
                    with open(pdf_file, "rb") as f:
                        st.download_button(
                            "⬇️ Download PDF",
                            data=f.read(),
                            file_name=f"{meta.video_id}_report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

        with col_e2:
            st.markdown("##### 📝 Markdown Notes")
            st.caption("Clean GitHub-flavored Markdown for Obsidian, Notion, or GitHub.")
            md_file = MarkdownExporter.export(summary, meta)
            with open(md_file, "r", encoding="utf-8") as f:
                md_content = f.read()
            st.download_button(
                "⬇️ Download Markdown",
                data=md_content,
                file_name=f"{meta.video_id}_notes.md",
                mime="text/markdown",
                use_container_width=True
            )

        with col_e3:
            st.markdown("##### 🎬 Subtitle File (.SRT)")
            st.caption("Standard SubRip subtitle track with exact timestamp synchronization.")
            srt_file = SRTExporter.export(transcript)
            with open(srt_file, "r", encoding="utf-8") as f:
                srt_content = f.read()
            st.download_button(
                "⬇️ Download .SRT Subtitles",
                data=srt_content,
                file_name=f"{meta.video_id}.srt",
                mime="text/plain",
                use_container_width=True
            )

else:
    # Empty State Hero
    st.info("👈 Paste a YouTube link or upload an audio/video file in the sidebar to begin.")
    st.markdown("""
    ### How It Works:
    1. **Ingest & Normalize:** Downloads YouTube video or ingests local media, standardizing audio to 16kHz mono WAV.
    2. **Transcribe:** Powered by local `faster-whisper` (English / Global) or Sarvam AI API (Hindi / Hinglish).
    3. **Synthesize:** Mistral AI analyzes the transcript to build executive summaries, timestamped chapters, and action items.
    4. **Chat & Retrieve:** Ask questions in natural language with verified `[MM:SS]` timestamp citations.
    5. **Export:** Download professional PDF reports, Markdown notes, and SRT subtitles.
    """)
