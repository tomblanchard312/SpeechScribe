"""
Ingestion Layer - Audio Source Adapters

This layer handles audio ingestion from various sources
(files, Teams, Zoom, etc.) and normalizes them to AudioFrame
objects for the speech pipeline.

Adapters are responsible for:
- Connecting to audio sources
- Normalizing audio to PCM format
- Providing speaker hints when available
- Streaming or batch delivery of AudioFrames
"""

import abc
import logging
from typing import Iterator, Optional, Dict, Any, List
from pathlib import Path
import uuid

from .models import AudioFrame
from .audio import AudioProcessor

logger = logging.getLogger(__name__)


class AudioAdapter(abc.ABC):
    """
    Base class for audio ingestion adapters.

    Adapters convert various audio sources into normalized AudioFrame streams
    that can be consumed by the speech processing pipeline.
    """

    def __init__(self, session_id: str, config: Dict[str, Any]):
        self.session_id = session_id
        self.config = config
        self.audio_processor = AudioProcessor(config)
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

    def create_frame(self, data: bytes, timestamp_ms: int,
                     sample_rate: int = 16000, channels: int = 1,
                     speaker_hint: Optional[str] = None) -> AudioFrame:
        """Create a normalized AudioFrame."""
        return AudioFrame(
            session_id=self.session_id,
            stream_id=self.stream_id,
            timestamp_ms=timestamp_ms,
            data=data,
            sample_rate=sample_rate,
            channels=channels,
            speaker_hint=speaker_hint
        )


class FileAdapter(AudioAdapter):
    """
    Adapter for file-based audio ingestion.

    Supports batch processing of audio files from disk.
    """

    def __init__(self, session_id: str, config: Dict[str, Any],
                 file_paths: List[Path]):
        super().__init__(session_id, config)
        self.file_paths = file_paths
        self.current_file_index = 0

    def connect(self) -> bool:
        """Validate files exist and are readable."""
        for file_path in self.file_paths:
            is_valid, error = self.audio_processor.validate_audio_file(
                file_path)
            if not is_valid:
                logger.error(f"Invalid audio file {file_path}: {error}")
                return False
        logger.info(f"File adapter connected for {len(self.file_paths)} files")
        return True

    def disconnect(self) -> None:
        """No cleanup needed for files."""
        pass

    def get_audio_stream(self) -> Iterator[AudioFrame]:
        """Yield AudioFrames for each file."""
        for file_path in self.file_paths:
            logger.info(f"Processing file: {file_path}")

            # Convert to WAV if needed
            try:
                prepared_file = self.audio_processor.prepare_audio(file_path)
            except Exception as e:
                logger.error(f"Failed to prepare audio file {file_path}: {e}")
                continue

            # Read audio data
            import wave
            try:
                with wave.open(str(prepared_file), 'rb') as wf:
                    sample_rate = wf.getframerate()
                    channels = wf.getnchannels()
                    frames_data = wf.readframes(wf.getnframes())

                # Create single frame for entire file
                # In a real implementation, this might be chunked
                # for large files
                frame = self.create_frame(
                    data=frames_data,
                    timestamp_ms=0,
                    sample_rate=sample_rate,
                    channels=channels
                )
                yield frame

            except Exception as e:
                logger.error(f"Failed to read audio file {file_path}: {e}")

    def is_real_time(self) -> bool:
        return False


class TeamsAdapter(AudioAdapter):
    """
    Adapter for Microsoft Teams meeting audio.

    TODO: Implement Teams SDK integration for real-time audio capture.
    Requires Azure AD app registration and Teams SDK credentials.
    """

    def __init__(self, session_id: str, config: Dict[str, Any],
                 meeting_url: str, credentials: Dict[str, str]):
        super().__init__(session_id, config)
        self.meeting_url = meeting_url
        self.credentials = credentials
        # TODO: Initialize Teams SDK client

    def connect(self) -> bool:
        """Connect to Teams meeting."""
        # TODO: Implement Teams connection logic
        logger.warning("Teams adapter not yet implemented")
        return False

    def disconnect(self) -> None:
        """Disconnect from Teams meeting."""
        # TODO: Implement cleanup
        pass

    def get_audio_stream(self) -> Iterator[AudioFrame]:
        """Stream audio from Teams meeting."""
        # TODO: Implement real-time audio streaming
        return iter([])

    def is_real_time(self) -> bool:
        return True


class ZoomAdapter(AudioAdapter):
    """
    Adapter for Zoom meeting audio.

    TODO: Implement Zoom SDK integration for real-time audio capture.
    Requires Zoom app credentials and SDK setup.
    """

    def __init__(self, session_id: str, config: Dict[str, Any],
                 meeting_id: str, credentials: Dict[str, str]):
        super().__init__(session_id, config)
        self.meeting_id = meeting_id
        self.credentials = credentials
        # TODO: Initialize Zoom SDK client

    def connect(self) -> bool:
        """Connect to Zoom meeting."""
        # TODO: Implement Zoom connection logic
        logger.warning("Zoom adapter not yet implemented")
        return False

    def disconnect(self) -> None:
        """Disconnect from Zoom meeting."""
        # TODO: Implement cleanup
        pass

    def get_audio_stream(self) -> Iterator[AudioFrame]:
        """Stream audio from Zoom meeting."""
        # TODO: Implement real-time audio streaming
        return iter([])

    def is_real_time(self) -> bool:
        return True


class AdapterFactory:
    """
    Factory for creating audio adapters based on source type.
    """

    @staticmethod
    def create_adapter(source_type: str, session_id: str,
                       config: Dict[str, Any], **kwargs) -> AudioAdapter:
        """Create appropriate adapter for the source type."""

        if source_type == 'file':
            file_paths = kwargs.get('file_paths', [])
            return FileAdapter(session_id, config, file_paths)

        elif source_type == 'teams':
            meeting_url = kwargs.get('meeting_url', '')
            credentials = kwargs.get('credentials', {})
            return TeamsAdapter(session_id, config, meeting_url, credentials)

        elif source_type == 'zoom':
            meeting_id = kwargs.get('meeting_id', '')
            credentials = kwargs.get('credentials', {})
            return ZoomAdapter(session_id, config, meeting_id, credentials)

        else:
            raise ValueError(f"Unsupported source type: {source_type}")
