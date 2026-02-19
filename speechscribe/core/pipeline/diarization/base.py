"""
Diarization Module - Base Classes

Base classes for speaker diarization processing.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from ...models.transcript import TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class DiarizationConfig:
    """Configuration for diarization processing."""

    num_speakers: Optional[int] = None  # Auto-detect if None
    min_speakers: int = 1
    max_speakers: int = 10


class DiarizationProcessor(ABC):
    """
    Abstract base class for speaker diarization processors.

    Assigns speaker identities to transcript segments.
    """

    def __init__(self, config: DiarizationConfig):
        self.config = config

    @abstractmethod
    def process(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """
        Process transcript segments and assign speaker IDs.

        Args:
            segments: List of transcript segments without speaker IDs

        Returns:
            List of transcript segments with speaker IDs assigned
        """
        pass

    @abstractmethod
    def get_supported_engines(self) -> List[str]:
        """Get list of supported diarization engines."""
        pass
