"""
ASR Module - Whisper Processor

Whisper-based ASR implementation.
"""

import logging
from typing import Iterator, List, Optional
from pathlib import Path
import tempfile

from .base import ASRProcessor, ASRConfig
from ...models.transcript import TranscriptSegment, AudioFrame

logger = logging.getLogger(__name__)


class WhisperASRProcessor(ASRProcessor):
    """
    Whisper-based ASR processor.

    Uses faster-whisper for efficient speech recognition.
    """

    def __init__(self, config: ASRConfig):
        super().__init__(config)
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the Whisper model."""
        try:
            from faster_whisper import WhisperModel

            logger.info(f"Loading Whisper model: {self.config.model_name}")

            # Determine device and compute type
            device = "cuda" if self._has_cuda() else "cpu"
            compute_type = "int8" if device == "cpu" else "float16"

            self.model = WhisperModel(
                self.config.model_name,
                device=device,
                compute_type=compute_type
            )

            logger.info(
                f"Whisper model loaded successfully: {self.config.model_name}")

        except ImportError as e:
            raise ImportError(
                "faster-whisper is not installed.\n"
                "Install with: pip install faster-whisper\n\n"
                "On Windows CPU only, you can accelerate installs like:\n"
                "  pip install torch torchvision torchaudio "
                "--index-url https://download.pytorch.org/whl/cpu\n"
                "  pip install faster-whisper"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Whisper model {self.config.model_name}: {e}") from e

    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def process_stream(self, audio_frames: Iterator[AudioFrame]
                       ) -> Iterator[TranscriptSegment]:
        """
        Process streaming audio frames.

        For streaming, we buffer frames and process in chunks.
        """
        logger.info("Starting streaming ASR processing")

        # Buffer for collecting frames
        buffer = []
        buffer_duration = 0.0
        chunk_duration = 30.0  # Process 30-second chunks

        for frame in audio_frames:
            buffer.append(frame)
            buffer_duration += frame.duration

            # Process chunk when buffer is full
            if buffer_duration >= chunk_duration:
                yield from self._process_buffer(buffer)
                buffer = []
                buffer_duration = 0.0

        # Process remaining buffer
        if buffer:
            yield from self._process_buffer(buffer)

    def process_batch(
        self, audio_frames: List[AudioFrame]
    ) -> List[TranscriptSegment]:
        """
        Process batch of audio frames.
        """
        logger.info(
            f"Starting batch ASR processing of {len(audio_frames)} frames")

        return self._process_buffer(audio_frames)

    def _process_buffer(self, frames: List[AudioFrame]) -> List[TranscriptSegment]:
        """Process a buffer of audio frames."""
        if not frames:
            return []

        # Combine frames into a single audio file
        combined_audio = self._combine_frames(frames)
        if not combined_audio:
            return []

        try:
            # Transcribe using Whisper
            task = "translate" if self.config.translate else "transcribe"

            segments_iter, info = self.model.transcribe(
                str(combined_audio),
                task=task,
                language=self.config.language,
                vad_filter=self.config.vad_filter,
                vad_parameters=dict(
                    min_silence_duration_ms=self.config.min_silence_duration_ms)
            )

            # Convert to TranscriptSegments
            segments = []
            for segment in segments_iter:
                transcript_segment = TranscriptSegment(
                    session_id="",  # Will be set by orchestrator
                    start_time=segment.start,
                    end_time=segment.end,
                    text=segment.text.strip(),
                    speaker_id=None,  # Will be set by diarization if needed
                    confidence=segment.confidence if hasattr(
                        segment, 'confidence') else None,
                    language=info.language,
                    translated_text=segment.text if task == "translate" else None
                )
                segments.append(transcript_segment)

            logger.info(f"ASR processing completed: {len(segments)} segments")
            return segments

        finally:
            # Clean up temporary file
            if combined_audio and combined_audio.exists():
                combined_audio.unlink()

    def _combine_frames(self, frames: List[AudioFrame]) -> Optional[Path]:
        """Combine audio frames into a single file."""
        if not frames:
            return None

        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = Path(temp_file.name)

            # For now, assume frames contain raw audio data
            # In a real implementation, you'd need to handle audio concatenation
            # This is a placeholder implementation
            logger.warning("Audio frame combination not fully implemented")

            # Placeholder: just use the first frame's data
            if frames and hasattr(frames[0], 'data'):
                with open(temp_path, 'wb') as f:
                    f.write(frames[0].data)
                return temp_path
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to combine audio frames: {e}")
            return None

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        # Whisper supports many languages
        return [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'zh', 'ko',
            'ar', 'hi', 'nl', 'pl', 'tr', 'sv', 'da', 'no', 'fi', 'he'
        ]

    def is_streaming_supported(self) -> bool:
        """Check if streaming processing is supported."""
        return True  # Whisper can handle streaming with chunking
