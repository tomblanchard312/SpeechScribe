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
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .audio import AudioProcessor
from .models import AudioFrame

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


class FileAdapter(AudioAdapter):
    """
    Adapter for file-based audio ingestion.

    Supports batch processing of audio files from disk.
    """

    def __init__(self, session_id: str, config: Dict[str, Any], file_paths: List[Path]):
        super().__init__(session_id, config)
        self.file_paths = file_paths
        self.current_file_index = 0

    def connect(self) -> bool:
        """Validate files exist and are readable."""
        for file_path in self.file_paths:
            is_valid, error = self.audio_processor.validate_audio_file(file_path)
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
                with wave.open(str(prepared_file), "rb") as wf:
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
                    channels=channels,
                )
                yield frame

            except Exception as e:
                logger.error(f"Failed to read audio file {file_path}: {e}")

    def is_real_time(self) -> bool:
        return False


class TeamsAdapter(AudioAdapter):
    """
    Adapter for Microsoft Teams meeting audio.

    Mock implementation that simulates Teams meeting connection.
    In production, replace with actual Microsoft Teams SDK integration.
    """

    def __init__(
        self,
        session_id: str,
        config: Dict[str, Any],
        meeting_url: str,
        credentials: Dict[str, str],
    ):
        super().__init__(session_id, config)
        self.meeting_url = meeting_url
        self.credentials = credentials
        self.connected = False
        # Mock Teams SDK client
        self.teams_client = None

    def connect(self) -> bool:
        """Connect to Teams meeting."""
        try:
            logger.info(f"Connecting to Teams meeting {self.meeting_url}")
            # Mock connection - in real implementation, use Microsoft Teams SDK
            # self.teams_client = TeamsSDKClient(self.credentials)
            # self.teams_client.join_meeting(self.meeting_url)
            import time

            time.sleep(1)  # Simulate connection time
            self.connected = True
            logger.info("Successfully connected to Teams meeting")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Teams meeting: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from Teams meeting."""
        if self.connected:
            logger.info("Disconnecting from Teams meeting")
            # Mock disconnect
            # if self.teams_client:
            #     self.teams_client.leave_meeting()
            self.connected = False
        else:
            logger.warning("Not connected to Teams meeting")

    def get_audio_stream(self) -> Iterator[AudioFrame]:
        """Stream audio from Teams meeting."""
        if not self.connected:
            logger.error("Not connected to Teams meeting")
            return

        logger.info("Starting audio stream from Teams meeting")
        frame_count = 0

        try:
            while self.connected:
                # Mock audio frame - in real implementation, get from Teams SDK
                # audio_data = self.teams_client.get_audio_frame()
                mock_audio_data = b"\x00\x01\x02\x03" * 1024  # Mock 4KB audio data
                timestamp = time.time()

                frame = AudioFrame(
                    data=mock_audio_data,
                    timestamp=timestamp,
                    sample_rate=16000,
                    channels=1,
                    format="pcm",
                )

                yield frame
                frame_count += 1

                # Simulate real-time streaming
                time.sleep(0.1)  # 100ms frames

        except Exception as e:
            logger.error(f"Error in Teams audio stream: {e}")
        finally:
            logger.info(f"Teams audio stream ended after {frame_count} frames")

    def is_real_time(self) -> bool:
        return True


class ZoomAdapter(AudioAdapter):
    """
    Adapter for Zoom meeting audio.

    Mock implementation that simulates Zoom meeting connection.
    In production, replace with actual Zoom SDK integration.
    """

    def __init__(
        self,
        session_id: str,
        config: Dict[str, Any],
        meeting_id: str,
        credentials: Dict[str, str],
    ):
        super().__init__(session_id, config)
        self.meeting_id = meeting_id
        self.credentials = credentials
        self.connected = False
        # Mock Zoom SDK client
        self.zoom_client = None

    def connect(self) -> bool:
        """Connect to Zoom meeting."""
        try:
            logger.info(f"Connecting to Zoom meeting {self.meeting_id}")
            # Mock connection - in real implementation, use Zoom SDK
            # self.zoom_client = ZoomSDKClient(self.credentials)
            # self.zoom_client.join_meeting(self.meeting_id)
            import time

            time.sleep(1)  # Simulate connection time
            self.connected = True
            logger.info("Successfully connected to Zoom meeting")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Zoom meeting: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from Zoom meeting."""
        if self.connected:
            logger.info("Disconnecting from Zoom meeting")
            # Mock disconnect
            # if self.zoom_client:
            #     self.zoom_client.leave_meeting()
            self.connected = False
        else:
            logger.warning("Not connected to Zoom meeting")

    def get_audio_stream(self) -> Iterator[AudioFrame]:
        """Stream audio from Zoom meeting."""
        if not self.connected:
            logger.error("Not connected to Zoom meeting")
            return

        logger.info("Starting audio stream from Zoom meeting")
        frame_count = 0

        try:
            while self.connected:
                # Mock audio frame - in real implementation, get from Zoom SDK
                # audio_data = self.zoom_client.get_audio_frame()
                mock_audio_data = b"\x00\x01\x02\x03" * 1024  # Mock 4KB audio data
                timestamp = time.time()

                frame = AudioFrame(
                    data=mock_audio_data,
                    timestamp=timestamp,
                    sample_rate=16000,
                    channels=1,
                    format="pcm",
                )

                yield frame
                frame_count += 1

                # Simulate real-time streaming
                time.sleep(0.1)  # 100ms frames

        except Exception as e:
            logger.error(f"Error in Zoom audio stream: {e}")
        finally:
            logger.info(f"Zoom audio stream ended after {frame_count} frames")

    def is_real_time(self) -> bool:
        return True


class AdapterFactory:
    """
    Factory for creating audio adapters based on source type.
    """

    @staticmethod
    def create_adapter(
        source_type: str, session_id: str, config: Dict[str, Any], **kwargs
    ) -> AudioAdapter:
        """Create appropriate adapter for the source type."""

        if source_type == "file":
            file_paths = kwargs.get("file_paths", [])
            return FileAdapter(session_id, config, file_paths)

        elif source_type == "teams":
            meeting_url = kwargs.get("meeting_url", "")
            credentials = kwargs.get("credentials", {})
            return TeamsAdapter(session_id, config, meeting_url, credentials)

        elif source_type == "zoom":
            meeting_id = kwargs.get("meeting_id", "")
            credentials = kwargs.get("credentials", {})
            return ZoomAdapter(session_id, config, meeting_id, credentials)

        else:
            raise ValueError(f"Unsupported source type: {source_type}")
