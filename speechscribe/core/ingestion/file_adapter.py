"""
File-based Audio Ingestion Adapter

Supports batch processing of audio files from disk.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List

from ..models import AudioFrame
from ..utils.audio import AudioProcessor
from .base import IngestionAdapter

logger = logging.getLogger(__name__)


class FileAdapter(IngestionAdapter):
    """
    Adapter for file-based audio ingestion.

    Supports batch processing of audio files from disk.
    """

    def __init__(self, session_id: str, config: Dict[str, Any], file_paths: List[Path]):
        super().__init__(session_id, config)
        self.file_paths = file_paths
        self.audio_processor = AudioProcessor(config)

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
                # In a real implementation, this might be chunked for
                # large files
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
