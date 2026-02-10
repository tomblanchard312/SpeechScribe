"""
ASR Module - Base Classes

Base classes for Automatic Speech Recognition processing.
"""

import logging
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional
from dataclasses import dataclass

from ...models.transcript import TranscriptSegment, AudioFrame

logger = logging.getLogger(__name__)


@dataclass
class ASRConfig:
    """Configuration for ASR processing."""
    model_name: str = "whisper-small"
    language: Optional[str] = None
    translate: bool = False
    vad_filter: bool = True
    min_silence_duration_ms: int = 300


class ASRProcessor(ABC):
    """
    Abstract base class for ASR processors.

    Handles speech-to-text conversion with support for streaming and batch processing.
    """

    def __init__(self, config: ASRConfig):
        self.config = config

    @abstractmethod
    def process_stream(self, audio_frames: Iterator[AudioFrame]) -> Iterator[TranscriptSegment]:
        """
        Process streaming audio frames.

        Args:
            audio_frames: Iterator of audio frames

        Yields:
            TranscriptSegment objects as they become available
        """
        pass

    @abstractmethod
    def process_batch(self, audio_frames: List[AudioFrame]) -> List[TranscriptSegment]:
        """
        Process batch of audio frames.

        Args:
            audio_frames: List of audio frames

        Returns:
            List of TranscriptSegment objects
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        pass

    @abstractmethod
    def is_streaming_supported(self) -> bool:
        """Check if streaming processing is supported."""
        pass
