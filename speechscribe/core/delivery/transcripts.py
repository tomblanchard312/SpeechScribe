"""
Delivery Layer - Transcript Generation

Handles transcript formatting and output generation.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json

from ..models.transcript import TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class TranscriptConfig:
    """Configuration for transcript generation."""

    include_timestamps: bool = True
    include_speakers: bool = True
    include_confidence: bool = False
    format: str = "text"  # text, json, html, pdf
    speaker_labels: bool = True  # Use "Speaker 1" instead of "speaker_0"


class TranscriptGenerator:
    """
    Generates formatted transcripts from transcript segments.

    Supports multiple output formats and customization options.
    """

    def __init__(self, config: TranscriptConfig):
        self.config = config

    def generate_transcript(
        self,
        segments: List[TranscriptSegment],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate formatted transcript from segments.

        Args:
            segments: List of transcript segments
            metadata: Optional metadata to include

        Returns:
            Formatted transcript as string
        """
        logger.info(f"Generating transcript with {len(segments)} segments")

        if self.config.format == "json":
            return self._generate_json(segments, metadata)
        elif self.config.format == "html":
            return self._generate_html(segments, metadata)
        else:  # text format
            return self._generate_text(segments, metadata)

    def save_transcript(
        self,
        segments: List[TranscriptSegment],
        output_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save transcript to file.

        Args:
            segments: List of transcript segments
            output_path: Path to save transcript
            metadata: Optional metadata to include

        Returns:
            Path to saved file
        """
        logger.info(f"Saving transcript to {output_path}")

        content = self.generate_transcript(segments, metadata)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return output_path

    def _generate_text(
        self,
        segments: List[TranscriptSegment],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate plain text transcript."""
        lines = []

        # Add metadata header if provided
        if metadata:
            lines.extend(self._format_metadata_text(metadata))
            lines.append("")  # Empty line

        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda s: s.start_time)

        current_speaker = None
        for segment in sorted_segments:
            if not segment.text:
                continue

            # Speaker header
            if self.config.include_speakers and segment.speaker_id != current_speaker:
                speaker_label = self._format_speaker_label(segment.speaker_id)
                lines.append(f"{speaker_label}:")
                current_speaker = segment.speaker_id

            # Timestamp
            if self.config.include_timestamps:
                timestamp = self._format_timestamp(segment.start_time)
                line = f"[{timestamp}] {segment.text}"
            else:
                line = segment.text

            # Confidence
            if self.config.include_confidence and segment.confidence is not None:
                line += f" ({segment.confidence:.2f})"

            lines.append(line)
            lines.append("")  # Empty line between segments

        return "\n".join(lines)

    def _generate_json(
        self,
        segments: List[TranscriptSegment],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate JSON transcript."""
        # Convert segments to dictionaries
        segment_dicts = []
        for segment in segments:
            seg_dict = {
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "text": segment.text,
                "speaker_id": segment.speaker_id,
                "confidence": segment.confidence,
                "language": segment.language,
            }

            # Add translations if present
            if hasattr(segment, "translations") and segment.translations:
                seg_dict["translations"] = segment.translations

            segment_dicts.append(seg_dict)

        # Create full transcript object
        transcript = {"segments": segment_dicts, "metadata": metadata or {}}

        return json.dumps(transcript, indent=2, ensure_ascii=False)

    def _generate_html(
        self,
        segments: List[TranscriptSegment],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate HTML transcript."""
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>Transcript</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            ".speaker { font-weight: bold; color: #2c3e50; margin-top: 20px; }",
            ".timestamp { color: #7f8c8d; font-size: 0.9em; }",
            ".text { margin-left: 20px; line-height: 1.5; }",
            ".confidence { color: #95a5a6; font-size: 0.8em; }",
            ".metadata { background: #f8f9fa; padding: 20px; border-radius: 5px; "
            "margin-bottom: 30px; }",
            "</style>",
            "</head>",
            "<body>",
        ]

        # Add metadata
        if metadata:
            html_parts.extend(["<div class='metadata'>", "<h2>Metadata</h2>"])
            for key, value in metadata.items():
                html_parts.append(f"<p><strong>{key}:</strong> {value}</p>")
            html_parts.append("</div>")

        # Add transcript
        html_parts.append("<h2>Transcript</h2>")

        sorted_segments = sorted(segments, key=lambda s: s.start_time)
        current_speaker = None

        for segment in sorted_segments:
            if not segment.text:
                continue

            # Speaker header
            if self.config.include_speakers and segment.speaker_id != current_speaker:
                speaker_label = self._format_speaker_label(segment.speaker_id)
                html_parts.append(f"<div class='speaker'>{speaker_label}:</div>")
                current_speaker = segment.speaker_id

            # Segment content
            html_parts.append("<div class='text'>")

            if self.config.include_timestamps:
                timestamp = self._format_timestamp(segment.start_time)
                html_parts.append(f"<span class='timestamp'>[{timestamp}]</span> ")

            html_parts.append(segment.text)

            if self.config.include_confidence and segment.confidence is not None:
                html_parts.append(
                    f" <span class='confidence'>({segment.confidence:.2f})</span>"
                )

            html_parts.append("</div>")

        html_parts.extend(["</body>", "</html>"])

        return "\n".join(html_parts)

    def _format_metadata_text(self, metadata: Dict[str, Any]) -> List[str]:
        """Format metadata as text lines."""
        lines = ["TRANSCRIPT METADATA"]
        lines.append("=" * 20)

        for key, value in metadata.items():
            if isinstance(value, (int, float)):
                lines.append(f"{key}: {value}")
            else:
                lines.append(f"{key}: {str(value)}")

        return lines

    def _format_speaker_label(self, speaker_id: str) -> str:
        """Format speaker ID for display."""
        if not self.config.speaker_labels:
            return speaker_id

        # Convert speaker_0 to Speaker 1, etc.
        if speaker_id and speaker_id.startswith("speaker_"):
            try:
                num = int(speaker_id.split("_")[1]) + 1
                return f"Speaker {num}"
            except (ValueError, IndexError):
                pass
        return speaker_id or "Unknown Speaker"

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS or HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
