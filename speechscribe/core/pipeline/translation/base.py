"""
Translation Module - Base Classes

Base classes for translation processing.
"""

import logging
from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass

from ...models.transcript import TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class TranslationConfig:
    """Configuration for translation processing."""

    source_language: str = "auto"  # Auto-detect
    target_languages: List[str] = None
    engine: str = "whisper"  # Use Whisper's built-in translation


class TranslationProcessor(ABC):
    """
    Abstract base class for translation processors.

    Translates transcript segments to target languages.
    """

    def __init__(self, config: TranslationConfig):
        self.config = config

    @abstractmethod
    def process(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """
        Process transcript segments and add translations.

        Args:
            segments: List of transcript segments

        Returns:
            List of transcript segments with translations added
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported target languages."""
        pass
