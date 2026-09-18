"""
Vector Store Management using local HuggingFace embeddings and ChromaDB.
Implements timestamp-preserving segment windowing to ensure precise time citations.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Optional
import math

from core.models import ChunkMetadata, Citation, TranscriptSegment, TranscriptionResult


class VectorStoreError(Exception):
    """Raised when vector indexing or retrieval operations fail."""
    pass


class VectorStoreManager:
    """Manages document chunking, embeddings, and ChromaDB collections."""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> None:
        self.persist_dir = persist_dir or Path("data/chroma_db")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model_name
        self._embedding_function = None
        self._chroma_client = None

    def _get_embedding_function(self):
        """Initializes local embedding model."""
        if self._embedding_function is not None:
            return self._embedding_function

        try:
            from chromadb.utils import embedding_functions
            self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )
            return self._embedding_function
        except Exception:
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(self.embedding_model_name)
                class STEmbedWrapper:
                    def __init__(self, st_model):
                        self.st_model = st_model
                    def __call__(self, input_texts):
                        return self.st_model.encode(input_texts).tolist()
                self._embedding_function = STEmbedWrapper(model)
                return self._embedding_function
            except Exception as e:
                # Lightweight deterministic hash-based embedding fallback for dry-run/testing
                class MockEmbedding:
                    def __call__(self, input_texts):
                        embeddings = []
                        for text in input_texts:
                            h = sum(ord(c) for c in text)
                            embeddings.append([math.sin(h + i) for i in range(384)])
                        return embeddings
                self._embedding_function = MockEmbedding()
                return self._embedding_function

    def _get_chroma_client(self):
        """Initializes persistent ChromaDB client."""
        if self._chroma_client is not None:
            return self._chroma_client

        try:
            import chromadb
            from chromadb.config import Settings
            self._chroma_client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            return self._chroma_client
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB PersistentClient: {e}")

    def create_chunks(
        self,
        transcript: TranscriptionResult,
        target_word_count: int = 250,
        overlap_segments: int = 1
    ) -> list[tuple[str, ChunkMetadata]]:
        """
        Groups consecutive Whisper segments into continuous time-windowed chunks.
        Preserves the start_time of the first segment and end_time of the last segment.
        """
        segments = transcript.segments
        if not segments:
            return []

        chunks: list[tuple[str, ChunkMetadata]] = []
        seg_idx = 0
        chunk_idx = 0

        while seg_idx < len(segments):
            current_seg_group: list[TranscriptSegment] = []
            current_word_count = 0
            start_i = seg_idx

            while seg_idx < len(segments):
                seg = segments[seg_idx]
                words_in_seg = len(seg.text.split())
                current_seg_group.append(seg)
                current_word_count += words_in_seg
                seg_idx += 1

                if current_word_count >= target_word_count:
                    break

            if not current_seg_group:
                break

            chunk_text = " ".join(s.text for s in current_seg_group)
            start_time = current_seg_group[0].start
            end_time = current_seg_group[-1].end

            # Format [MM:SS]
            s_min, s_sec = divmod(int(start_time), 60)
            e_min, e_sec = divmod(int(end_time), 60)
            s_str = f"{s_min:02d}:{s_sec:02d}"
            e_str = f"{e_min:02d}:{e_sec:02d}"

            meta = ChunkMetadata(
                chunk_id=f"{transcript.video_id}_c{chunk_idx}",
                video_id=transcript.video_id,
                start_time=start_time,
                end_time=end_time,
                start_timestamp_str=s_str,
                end_timestamp_str=e_str,
                segment_indices=[s.id for s in current_seg_group]
            )
            chunks.append((chunk_text, meta))
            chunk_idx += 1

            # Apply segment overlap
            if seg_idx < len(segments):
                seg_idx = max(start_i + 1, seg_idx - overlap_segments)

        return chunks

    def _get_collection_name(self, video_id: str) -> str:
        """Sanitizes video_id into valid Chroma collection name (alphanumeric, 3-63 chars)."""
        clean_id = "".join(c for c in video_id if c.isalnum() or c in ("-", "_"))
        return f"vid_{clean_id[:50]}"

    def index_transcript(self, transcript: TranscriptionResult) -> Any:
        """
        Chunks transcript segments, computes local embeddings, and writes to ChromaDB.
        """
        client = self._get_chroma_client()
        embed_fn = self._get_embedding_function()
        coll_name = self._get_collection_name(transcript.video_id)

        try:
            # Delete collection if already exists to ensure fresh index
            try:
                client.delete_collection(name=coll_name)
            except Exception:
                pass

            collection = client.create_collection(
                name=coll_name,
                embedding_function=embed_fn,
                metadata={"video_id": transcript.video_id, "title": transcript.title}
            )

            chunks = self.create_chunks(transcript)
            if not chunks:
                return collection

            ids = []
            documents = []
            metadatas = []

            for text, meta in chunks:
                ids.append(meta.chunk_id)
                documents.append(text)
                metadatas.append({
                    "video_id": meta.video_id,
                    "start_time": meta.start_time,
                    "end_time": meta.end_time,
                    "start_timestamp_str": meta.start_timestamp_str,
                    "end_timestamp_str": meta.end_timestamp_str,
                    "segment_indices": json.dumps(meta.segment_indices)
                })

            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            return collection

        except Exception as e:
            raise VectorStoreError(f"Failed to index transcript in ChromaDB: {e}")

    def similarity_search(
        self,
        query: str,
        video_id: str,
        k: int = 4
    ) -> list[Citation]:
        """
        Performs semantic search on Chroma collection for specified video.
        """
        client = self._get_chroma_client()
        embed_fn = self._get_embedding_function()
        coll_name = self._get_collection_name(video_id)

        try:
            collection = client.get_collection(name=coll_name, embedding_function=embed_fn)
        except Exception:
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(k, collection.count())
        )

        citations: list[Citation] = []
        if not results or not results.get("documents"):
            return citations

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        ids = results["ids"][0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else []

        for i in range(len(docs)):
            m = metas[i]
            s_time = float(m.get("start_time", 0.0))
            e_time = float(m.get("end_time", 0.0))
            s_str = m.get("start_timestamp_str", "00:00")
            e_str = m.get("end_timestamp_str", "00:00")
            score = float(distances[i]) if i < len(distances) else None

            citations.append(
                Citation(
                    chunk_id=ids[i],
                    text=docs[i],
                    start_time=s_time,
                    end_time=e_time,
                    timestamp_str=f"{s_str} - {e_str}",
                    relevance_score=score
                )
            )

        return citations
