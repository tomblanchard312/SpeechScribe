"""
Summarization Module - Base Classes

Base classes for summarization processing.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from ...models.transcript import TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class SummarizationConfig:
    """Configuration for summarization processing."""

    max_length: int = 200  # Maximum summary length in words
    min_length: int = 50  # Minimum summary length in words
    style: str = "extractive"  # extractive or abstractive


class SummarizationProcessor(ABC):
    """
    Abstract base class for summarization processors.

    Generates summaries from transcript segments.
    """

    def __init__(self, config: SummarizationConfig):
        self.config = config

    @abstractmethod
    def process(self, segments: List[TranscriptSegment]) -> str:
        """
        Generate summary from transcript segments.

        Args:
            segments: List of transcript segments

        Returns:
            Summary text
        """
        pass

    @abstractmethod
    def get_supported_styles(self) -> List[str]:
        """Get list of supported summarization styles."""
        pass
