"""
Delivery Layer - Summary Generation

Handles summary formatting and output generation.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from ..models.transcript import TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class SummaryConfig:
    """Configuration for summary generation."""

    include_key_points: bool = True
    include_timestamps: bool = False
    include_speakers: bool = True
    format: str = "text"  # text, json, html
    max_length: int = 500  # Maximum summary length in words


class SummaryGenerator:
    """
    Generates formatted summaries from transcripts and processing results.

    Supports multiple output formats and summary types.
    """

    def __init__(self, config: SummaryConfig):
        self.config = config

    def generate_summary(
        self,
        segments: List[TranscriptSegment],
        summary_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate formatted summary.

        Args:
            segments: List of transcript segments
            summary_text: Pre-generated summary text (from summarization pipeline)
            metadata: Optional metadata

        Returns:
            Formatted summary as string
        """
        logger.info("Generating formatted summary")

        if self.config.format == "json":
            return self._generate_json_summary(segments, summary_text, metadata)
        elif self.config.format == "html":
            return self._generate_html_summary(segments, summary_text, metadata)
        else:  # text format
            return self._generate_text_summary(segments, summary_text, metadata)

    def save_summary(
        self,
        segments: List[TranscriptSegment],
        output_path: Path,
        summary_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save summary to file.

        Args:
            segments: List of transcript segments
            output_path: Path to save summary
            summary_text: Pre-generated summary text
            metadata: Optional metadata

        Returns:
            Path to saved file
        """
        logger.info(f"Saving summary to {output_path}")

        content = self.generate_summary(segments, summary_text, metadata)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return output_path

    def _generate_text_summary(
        self,
        segments: List[TranscriptSegment],
        summary_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate plain text summary."""
        lines = []

        # Add header
        lines.append("MEETING SUMMARY")
        lines.append("=" * 50)
        lines.append("")

        # Add metadata
        if metadata:
            lines.extend(self._format_metadata_text(metadata))
            lines.append("")

        # Add main summary
        if summary_text:
            lines.append("SUMMARY:")
            lines.append("-" * 20)
            lines.append(summary_text)
            lines.append("")
        else:
            # Generate basic summary from segments
            summary_text = self._generate_basic_summary(segments)
            lines.append("SUMMARY:")
            lines.append("-" * 20)
            lines.append(summary_text)
            lines.append("")

        # Add key points if requested
        if self.config.include_key_points:
            lines.append("KEY POINTS:")
            lines.append("-" * 20)
            key_points = self._extract_key_points(segments)
            for point in key_points:
                lines.append(f"• {point}")
            lines.append("")

        # Add speaker summary if requested
        if self.config.include_speakers:
            lines.append("PARTICIPANTS:")
            lines.append("-" * 20)
            speaker_summary = self._generate_speaker_summary(segments)
            lines.extend(speaker_summary)
            lines.append("")

        return "\n".join(lines)

    def _generate_json_summary(
        self,
        segments: List[TranscriptSegment],
        summary_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate JSON summary."""
        import json

        summary_data = {
            "summary": summary_text or self._generate_basic_summary(segments),
            "key_points": (
                self._extract_key_points(segments)
                if self.config.include_key_points
                else []
            ),
            "participants": (
                self._generate_speaker_summary(segments)
                if self.config.include_speakers
                else []
            ),
            "metadata": metadata or {},
        }

        return json.dumps(summary_data, indent=2, ensure_ascii=False)

    def _generate_html_summary(
        self,
        segments: List[TranscriptSegment],
        summary_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate HTML summary."""
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>Meeting Summary</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; "
            "max-width: 800px; }",
            "h1 { color: #2c3e50; border-bottom: 2px solid #3498db; "
            "padding-bottom: 10px; }",
            "h2 { color: #34495e; margin-top: 30px; }",
            ".summary { background: #f8f9fa; padding: 20px; border-radius: 5px; "
            "margin: 20px 0; }",
            ".key-points { margin: 20px 0; }",
            ".key-points ul { padding-left: 20px; }",
            ".participants { margin: 20px 0; }",
            ".participant { margin: 10px 0; padding: 10px; background: #ecf0f1; "
            "border-radius: 3px; }",
            ".metadata { background: #e8f4f8; padding: 15px; border-radius: 5px; "
            "margin-bottom: 30px; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Meeting Summary</h1>",
        ]

        # Add metadata
        if metadata:
            html_parts.append("<div class='metadata'>")
            html_parts.append("<h2>Metadata</h2>")
            for key, value in metadata.items():
                html_parts.append(f"<p><strong>{key}:</strong> {value}</p>")
            html_parts.append("</div>")

        # Add main summary
        summary = summary_text or self._generate_basic_summary(segments)
        html_parts.extend(
            ["<h2>Summary</h2>", "<div class='summary'>", f"<p>{summary}</p>", "</div>"]
        )

        # Add key points
        if self.config.include_key_points:
            key_points = self._extract_key_points(segments)
            html_parts.extend(
                ["<h2>Key Points</h2>", "<div class='key-points'>", "<ul>"]
            )
            for point in key_points:
                html_parts.append(f"<li>{point}</li>")
            html_parts.extend(["</ul>", "</div>"])

        # Add participants
        if self.config.include_speakers:
            speaker_summary = self._generate_speaker_summary(segments)
            html_parts.extend(["<h2>Participants</h2>", "<div class='participants'>"])
            for participant in speaker_summary:
                html_parts.append(f"<div class='participant'>{participant}</div>")
            html_parts.append("</div>")

        html_parts.extend(["</body>", "</html>"])

        return "\n".join(html_parts)

    def _generate_basic_summary(self, segments: List[TranscriptSegment]) -> str:
        """Generate a basic summary from segments."""
        if not segments:
            return "No content available for summary."

        # Simple extractive summary: take first few segments
        text_segments = [s.text for s in segments[:5] if s.text]  # First 5 segments
        summary = " ".join(text_segments)

        # Truncate to max length
        words = summary.split()
        if len(words) > self.config.max_length:
            summary = " ".join(words[: self.config.max_length]) + "..."

        return summary

    def _extract_key_points(self, segments: List[TranscriptSegment]) -> List[str]:
        """Extract key points from segments."""
        # Simple implementation: look for sentences with keywords
        key_indicators = [
            "important",
            "key",
            "summary",
            "conclusion",
            "decision",
            "action",
        ]

        key_points = []
        for segment in segments:
            if not segment.text:
                continue

            sentences = segment.text.split(".")
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if any(indicator in sentence.lower() for indicator in key_indicators):
                    key_points.append(sentence)
                    if len(key_points) >= 10:  # Limit key points
                        break

            if len(key_points) >= 10:
                break

        return key_points[:10]  # Return up to 10 key points

    def _generate_speaker_summary(self, segments: List[TranscriptSegment]) -> List[str]:
        """Generate summary of speaker participation."""
        from collections import defaultdict

        speaker_stats = defaultdict(
            lambda: {"segments": 0, "words": 0, "duration": 0.0}
        )

        for segment in segments:
            if not segment.speaker_id or not segment.text:
                continue

            speaker_stats[segment.speaker_id]["segments"] += 1
            speaker_stats[segment.speaker_id]["words"] += len(segment.text.split())
            speaker_stats[segment.speaker_id]["duration"] += (
                segment.end_time - segment.start_time
            )

        # Format summary
        summary = []
        for speaker_id, stats in speaker_stats.items():
            speaker_label = self._format_speaker_label(speaker_id)
            summary.append(
                f"{speaker_label}: {stats['segments']} segments, "
                f"{stats['words']} words, {stats['duration']:.1f}s duration"
            )

        return summary

    def _format_metadata_text(self, metadata: Dict[str, Any]) -> List[str]:
        """Format metadata as text lines."""
        lines = []
        for key, value in metadata.items():
            lines.append(f"{key}: {value}")
        return lines

    def _format_speaker_label(self, speaker_id: str) -> str:
        """Format speaker ID for display."""
        if speaker_id and speaker_id.startswith("speaker_"):
            try:
                num = int(speaker_id.split("_")[1]) + 1
                return f"Speaker {num}"
            except (ValueError, IndexError):
                pass
        return speaker_id or "Unknown Speaker"
