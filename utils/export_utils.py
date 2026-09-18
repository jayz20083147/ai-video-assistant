"""
Export Utilities: Generates branded PDF meeting reports (via fpdf2),
SubRip (.srt) subtitle files, and clean Markdown notes.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

from core.models import AudioMetadata, SummaryResult, TranscriptionResult


class ExportError(Exception):
    """Raised when document export fails."""
    pass


class PDFReportGenerator:
    """Creates formatted PDF reports of video summaries and action items."""

    @classmethod
    def generate(
        cls,
        summary: SummaryResult,
        metadata: Optional[AudioMetadata] = None,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Builds a PDF document with executive brief, takeaways, decisions, and action items.
        """
        out_file = output_path or Path(f"exports/{summary.video_id}_summary.pdf")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            from fpdf import FPDF

            class MeetingReportPDF(FPDF):
                def header(self):
                    self.set_font("Helvetica", "B", 10)
                    self.set_text_color(120, 120, 120)
                    self.cell(0, 10, "AI Video Assistant — Meeting & Video Intelligence Report", border=0, ln=1, align="R")
                    self.line(10, 20, 200, 20)
                    self.ln(5)

                def footer(self):
                    self.set_y(-15)
                    self.set_font("Helvetica", "I", 8)
                    self.set_text_color(150, 150, 150)
                    self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

            pdf = MeetingReportPDF()
            pdf.alias_nb_pages()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)

            # Document Title
            pdf.set_font("Helvetica", "B", 18)
            pdf.set_text_color(20, 30, 60)
            title = metadata.title if metadata else "Video Analysis Report"
            pdf.multi_cell(0, 8, title)
            pdf.ln(3)

            # Metadata Strip
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(100, 100, 100)
            meta_parts = []
            if metadata:
                meta_parts.append(f"Duration: {metadata.formatted_duration}")
                if metadata.source_url:
                    meta_parts.append(f"Source: {metadata.source_url}")
            meta_parts.append(f"Generated: {summary.created_at.strftime('%Y-%m-%d %H:%M UTC')}")
            pdf.cell(0, 6, " | ".join(meta_parts), ln=1)
            pdf.ln(5)

            # 1. Executive Summary
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(30, 40, 80)
            pdf.cell(0, 8, "1. Executive Summary", ln=1)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(0, 6, summary.executive_summary)
            pdf.ln(5)

            # 2. Key Takeaways
            if summary.key_takeaways:
                pdf.set_font("Helvetica", "B", 14)
                pdf.set_text_color(30, 40, 80)
                pdf.cell(0, 8, "2. Key Takeaways", ln=1)
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(40, 40, 40)
                for item in summary.key_takeaways:
                    pdf.multi_cell(0, 6, f"-  {item}")
                pdf.ln(5)

            # 3. Key Decisions
            if summary.key_decisions:
                pdf.set_font("Helvetica", "B", 14)
                pdf.set_text_color(30, 40, 80)
                pdf.cell(0, 8, "3. Key Decisions", ln=1)
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(40, 40, 40)
                for dec in summary.key_decisions:
                    pdf.multi_cell(0, 6, f"-  {dec}")
                pdf.ln(5)

            # 4. Action Items Table
            if summary.action_items:
                pdf.set_font("Helvetica", "B", 14)
                pdf.set_text_color(30, 40, 80)
                pdf.cell(0, 8, "4. Action Items & Next Steps", ln=1)
                pdf.ln(2)

                # Table Header
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_fill_color(230, 235, 245)
                pdf.set_text_color(30, 30, 30)
                pdf.cell(90, 7, "Task", border=1, fill=True)
                pdf.cell(35, 7, "Assignee", border=1, fill=True)
                pdf.cell(25, 7, "Priority", border=1, fill=True)
                pdf.cell(40, 7, "Timestamp / Due", border=1, fill=True, ln=1)

                pdf.set_font("Helvetica", "", 8)
                for item in summary.action_items:
                    t_info = item.context_timestamp or item.due_date or "-"
                    pdf.cell(90, 6, item.task[:55], border=1)
                    pdf.cell(35, 6, item.assignee[:20], border=1)
                    pdf.cell(25, 6, item.priority.value, border=1)
                    pdf.cell(40, 6, str(t_info)[:25], border=1, ln=1)
                pdf.ln(5)

            # 5. Chapters
            if summary.chapters:
                pdf.set_font("Helvetica", "B", 14)
                pdf.set_text_color(30, 40, 80)
                pdf.cell(0, 8, "5. Timestamped Chapters", ln=1)
                pdf.set_font("Helvetica", "", 9)
                for ch in summary.chapters:
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.set_text_color(50, 70, 120)
                    pdf.cell(0, 6, f"[{ch.time_range_str}] {ch.title}", ln=1)
                    pdf.set_font("Helvetica", "", 9)
                    pdf.set_text_color(40, 40, 40)
                    pdf.multi_cell(0, 5, ch.summary)
                    pdf.ln(2)

            pdf.output(str(out_file))
            return out_file

        except ImportError:
            # Fallback if fpdf2 not installed: generate clean Markdown summary file
            fallback_md = out_file.with_suffix(".md")
            MarkdownExporter.export(summary, metadata, fallback_md)
            return fallback_md


class SRTExporter:
    """Exports transcript to SubRip (.srt) subtitle format."""

    @classmethod
    def export(cls, transcript: TranscriptionResult, output_path: Optional[Path] = None) -> Path:
        out_file = output_path or Path(f"exports/{transcript.video_id}.srt")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        def _to_srt_time(seconds: float) -> str:
            hrs, rem = divmod(int(seconds), 3600)
            mins, secs = divmod(rem, 60)
            millis = int(round((seconds - int(seconds)) * 1000))
            return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

        lines = []
        for i, seg in enumerate(transcript.segments, 1):
            s_str = _to_srt_time(seg.start)
            e_str = _to_srt_time(seg.end)
            lines.append(f"{i}")
            lines.append(f"{s_str} --> {e_str}")
            lines.append(seg.text)
            lines.append("")

        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return out_file


class MarkdownExporter:
    """Exports meeting intelligence to structured Markdown."""

    @classmethod
    def export(
        cls,
        summary: SummaryResult,
        metadata: Optional[AudioMetadata] = None,
        output_path: Optional[Path] = None
    ) -> Path:
        out_file = output_path or Path(f"exports/{summary.video_id}_notes.md")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        title = metadata.title if metadata else "Video Analysis Report"
        lines = [f"# {title}\n"]

        if metadata:
            lines.append(f"- **Duration:** {metadata.formatted_duration}")
            if metadata.source_url:
                lines.append(f"- **Source:** [{metadata.source_url}]({metadata.source_url})")
            lines.append(f"- **Date:** {summary.created_at.strftime('%Y-%m-%d %H:%M UTC')}\n")

        lines.append("## Executive Summary")
        lines.append(summary.executive_summary + "\n")

        if summary.key_takeaways:
            lines.append("## Key Takeaways")
            for item in summary.key_takeaways:
                lines.append(f"- {item}")
            lines.append("")

        if summary.key_decisions:
            lines.append("## Key Decisions")
            for dec in summary.key_decisions:
                lines.append(f"- {dec}")
            lines.append("")

        if summary.action_items:
            lines.append("## Action Items")
            lines.append("| Task | Assignee | Priority | Timestamp / Deadline |")
            lines.append("|---|---|---|---|")
            for act in summary.action_items:
                t_stamp = act.context_timestamp or act.due_date or "-"
                lines.append(f"| {act.task} | {act.assignee} | {act.priority.value} | {t_stamp} |")
            lines.append("")

        if summary.chapters:
            lines.append("## Chapters")
            for ch in summary.chapters:
                lines.append(f"### `[{ch.time_range_str}]` {ch.title}")
                lines.append(ch.summary + "\n")

        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return out_file
