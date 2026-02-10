"""
Ingestion Layer - Base Adapter Interface

This module defines the base interface for audio ingestion adapters.
Adapters are responsible for connecting to audio sources and normalizing
audio to AudioFrame objects for the speech pipeline.
"""

import abc
import logging
from typing import Iterator, Optional, Dict, Any
import uuid

from ..models import AudioFrame

logger = logging.getLogger(__name__)


class IngestionAdapter(abc.ABC):
    """
    Base class for audio ingestion adapters.

    Adapters convert various audio sources into normalized AudioFrame streams
    that can be consumed by the speech processing pipeline.
    """

    def __init__(self, session_id: str, config: Dict[str, Any]):
        self.session_id = session_id
        self.config = config
        self.stream_id = str(uuid.uuid4())

    @abc.abstractmethod
    def connect(self) -> bool:
        """Establish connection to audio source."""
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Clean up connections and resources."""
        pass

    @abc.abstractmethod
    def get_audio_stream(self) -> Iterator[AudioFrame]:
        """Return iterator of normalized AudioFrames."""
        pass

    @abc.abstractmethod
    def is_real_time(self) -> bool:
        """Return True if adapter provides real-time streaming."""
        pass

    def create_frame(
        self,
        data: bytes,
        timestamp_ms: int,
        sample_rate: int = 16000,
        channels: int = 1,
        speaker_hint: Optional[str] = None,
    ) -> AudioFrame:
        """Create a normalized AudioFrame."""
        return AudioFrame(
            session_id=self.session_id,
            stream_id=self.stream_id,
            timestamp_ms=timestamp_ms,
            data=data,
            sample_rate=sample_rate,
            channels=channels,
            speaker_hint=speaker_hint,
        )
