"""
TTS Module - Base Classes

Base classes for Text-to-Speech processing.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ...models.transcript import TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class TTSConfig:
    """Configuration for TTS processing."""

    voice: str = "default"
    language: str = "en"
    speed: float = 1.0
    output_format: str = "wav"


class TTSProcessor(ABC):
    """
    Abstract base class for Text-to-Speech processors.

    Converts text to speech audio.
    """

    def __init__(self, config: TTSConfig):
        self.config = config

    @abstractmethod
    def synthesize(self, text: str, output_path: Optional[Path] = None) -> bytes:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            output_path: Optional path to save audio file

        Returns:
            Audio data as bytes
        """
        pass

    @abstractmethod
    def synthesize_segments(
        self, segments: List[TranscriptSegment], output_dir: Path
    ) -> List[Path]:
        """
        Synthesize speech for multiple segments.

        Args:
            segments: List of transcript segments
            output_dir: Directory to save audio files

        Returns:
            List of paths to generated audio files
        """
        pass

    @abstractmethod
    def get_available_voices(self) -> List[str]:
        """Get list of available voices."""
        pass
