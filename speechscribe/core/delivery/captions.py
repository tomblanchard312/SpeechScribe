"""
Delivery Layer - Live Captioning

Handles real-time caption display and formatting.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from ..models.transcript import TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class CaptionConfig:
    """Configuration for caption display."""
    max_lines: int = 3
    max_chars_per_line: int = 40
    display_duration: float = 5.0  # seconds
    show_speaker: bool = True
    font_size: int = 24
    position: str = "bottom"  # bottom, top, overlay


class LiveCaptioning:
    """
    Handles live caption display and formatting.

    Manages real-time caption presentation with proper timing,
    speaker identification, and display formatting.
    """

    def __init__(self, config: CaptionConfig):
        self.config = config
        self.active_captions: List[Dict[str, Any]] = []
        self.caption_history: List[TranscriptSegment] = []

    def add_segment(self, segment: TranscriptSegment) -> List[Dict[str, Any]]:
        """
        Add a new transcript segment for captioning.

        Args:
            segment: New transcript segment

        Returns:
            Current active captions for display
        """
        logger.debug(f"Adding caption segment: {segment.text[:50]}...")

        # Add to history
        self.caption_history.append(segment)

        # Create caption entry
        caption = {
            'text': self._format_caption_text(segment),
            'start_time': segment.start_time,
            'end_time': segment.end_time,
            'speaker': segment.speaker_id,
            'confidence': segment.confidence
        }

        # Add to active captions
        self.active_captions.append(caption)

        # Clean up old captions
        self._cleanup_old_captions()

        # Format for display
        return self._format_display_captions()

    def get_current_captions(self) -> List[Dict[str, Any]]:
        """
        Get currently active captions for display.

        Returns:
            List of caption dictionaries for display
        """
        self._cleanup_old_captions()
        return self._format_display_captions()

    def clear_captions(self):
        """Clear all active captions."""
        self.active_captions.clear()
        logger.debug("Captions cleared")

    def _format_caption_text(self, segment: TranscriptSegment) -> str:
        """Format caption text with speaker info if enabled."""
        text = segment.text or ""

        if self.config.show_speaker and segment.speaker_id:
            speaker_label = self._format_speaker_label(segment.speaker_id)
            text = f"{speaker_label}: {text}"

        return self._wrap_text(text)

    def _format_speaker_label(self, speaker_id: str) -> str:
        """Format speaker ID for display."""
        # Convert speaker_0 to Speaker 1, etc.
        if speaker_id.startswith("speaker_"):
            try:
                num = int(speaker_id.split("_")[1]) + 1
                return f"Speaker {num}"
            except (ValueError, IndexError):
                pass
        return speaker_id

    def _wrap_text(self, text: str) -> str:
        """Wrap text to fit caption display constraints."""
        if len(text) <= self.config.max_chars_per_line:
            return text

        # Simple word wrapping
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line + " " + word) <= self.config.max_chars_per_line:
                current_line += " " + word if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return "\n".join(lines[:self.config.max_lines])

    def _cleanup_old_captions(self):
        """Remove captions that have expired."""
        import time
        current_time = time.time()

        # Keep captions within display duration
        self.active_captions = [
            caption for caption in self.active_captions
            if current_time - caption['start_time'] < self.config.display_duration
        ]

    def _format_display_captions(self) -> List[Dict[str, Any]]:
        """Format captions for display output."""
        # Sort by start time
        sorted_captions = sorted(self.active_captions,
                                 key=lambda x: x['start_time'])

        # Limit to max lines
        display_captions = []
        total_lines = 0

        for caption in reversed(sorted_captions):  # Most recent first
            lines = caption['text'].count('\n') + 1
            if total_lines + lines > self.config.max_lines:
                break

            # Insert at beginning to maintain order
            display_captions.insert(0, caption)
            total_lines += lines

        return display_captions

    def export_captions(self, output_path: Path, format: str = "srt") -> Path:
        """
        Export caption history to file.

        Args:
            output_path: Path to save captions
            format: Export format (srt, vtt, etc.)

        Returns:
            Path to exported file
        """
        logger.info(f"Exporting captions to {output_path} in {format} format")

        if format.lower() == "srt":
            content = self._export_srt()
        elif format.lower() == "vtt":
            content = self._export_vtt()
        else:
            content = self._export_text()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return output_path

    def _export_srt(self) -> str:
        """Export as SRT format."""
        lines = []
        for i, segment in enumerate(self.caption_history, 1):
            start = self._format_timestamp(segment.start_time)
            end = self._format_timestamp(segment.end_time)
            text = self._format_caption_text(segment)

            lines.extend([
                str(i),
                f"{start} --> {end}",
                text,
                ""  # Empty line
            ])

        return "\n".join(lines)

    def _export_vtt(self) -> str:
        """Export as WebVTT format."""
        lines = ["WEBVTT", ""]

        for segment in self.caption_history:
            start = self._format_timestamp(segment.start_time)
            end = self._format_timestamp(segment.end_time)
            text = self._format_caption_text(segment)

            lines.extend([
                f"{start} --> {end}",
                text,
                ""
            ])

        return "\n".join(lines)

    def _export_text(self) -> str:
        """Export as plain text."""
        lines = []
        for segment in self.caption_history:
            timestamp = f"[{self._format_timestamp(segment.start_time)}]"
            text = self._format_caption_text(segment)
            lines.append(f"{timestamp} {text}")

        return "\n".join(lines)

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS.mmm or MM:SS.mmm."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60

        if hours > 0:
            return "02d"
        else:
            return "02d"
