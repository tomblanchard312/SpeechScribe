"""
ASR Module - Whisper Processor

Whisper-based ASR implementation.
"""

import logging
import tempfile
import wave
from pathlib import Path
from typing import Iterator, List, Optional

from ...models.transcript import AudioFrame, TranscriptSegment
from .base import ASRConfig, ASRProcessor

logger = logging.getLogger(__name__)


class WhisperASRProcessor(ASRProcessor):
    """
    Whisper-based ASR processor.

    Uses faster-whisper for efficient speech recognition.
    """

    _SAMPLE_WIDTH_BYTES = 2  # Audio ingestion normalizes to 16-bit PCM.

    def __init__(self, config: ASRConfig):
        super().__init__(config)
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the Whisper model."""
        try:
            from faster_whisper import WhisperModel

            logger.info(f"Loading Whisper model: {self.config.model_name}")

            device = "cuda" if self._has_cuda() else "cpu"
            compute_type = "int8" if device == "cpu" else "float16"

            self.model = WhisperModel(
                self.config.model_name, device=device, compute_type=compute_type
            )

            logger.info(f"Whisper model loaded successfully: {self.config.model_name}")

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
                f"Failed to load Whisper model {self.config.model_name}: {e}"
            ) from e

    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def process_stream(
        self, audio_frames: Iterator[AudioFrame]
    ) -> Iterator[TranscriptSegment]:
        """
        Process streaming audio frames.

        Frames are accumulated into approximately 30-second PCM chunks before
        being written as a valid WAV file and passed to faster-whisper.
        """
        logger.info("Starting streaming ASR processing")

        buffer: List[AudioFrame] = []
        buffer_duration = 0.0
        chunk_duration = 30.0

        for frame in audio_frames:
            buffer.append(frame)
            buffer_duration += self._frame_duration_seconds(frame)

            if buffer_duration >= chunk_duration:
                yield from self._process_buffer(buffer)
                buffer = []
                buffer_duration = 0.0

        if buffer:
            yield from self._process_buffer(buffer)

    def process_batch(self, audio_frames: List[AudioFrame]) -> List[TranscriptSegment]:
        """Process a batch of audio frames."""
        logger.info(f"Starting batch ASR processing of {len(audio_frames)} frames")
        return self._process_buffer(audio_frames)

    def _process_buffer(self, frames: List[AudioFrame]) -> List[TranscriptSegment]:
        """Process a buffer of audio frames."""
        if not frames:
            return []

        combined_audio = self._combine_frames(frames)
        if not combined_audio:
            return []

        try:
            task = "translate" if self.config.translate else "transcribe"

            segments_iter, info = self.model.transcribe(
                str(combined_audio),
                task=task,
                language=self.config.language,
                vad_filter=self.config.vad_filter,
                vad_parameters=dict(
                    min_silence_duration_ms=self.config.min_silence_duration_ms
                ),
            )

            segments = []
            for segment in segments_iter:
                transcript_segment = TranscriptSegment(
                    session_id="",  # Will be set by orchestrator
                    start_time=segment.start,
                    end_time=segment.end,
                    text=segment.text.strip(),
                    speaker_id=None,  # Will be set by diarization if needed
                    confidence=(
                        segment.confidence if hasattr(segment, "confidence") else None
                    ),
                    language=info.language,
                    translated_text=segment.text if task == "translate" else None,
                )
                segments.append(transcript_segment)

            logger.info(f"ASR processing completed: {len(segments)} segments")
            return segments

        finally:
            if combined_audio and combined_audio.exists():
                combined_audio.unlink()

    def _frame_duration_seconds(self, frame: AudioFrame) -> float:
        """Calculate PCM frame duration without relying on a synthetic field."""
        bytes_per_second = (
            frame.sample_rate * frame.channels * self._SAMPLE_WIDTH_BYTES
        )
        if bytes_per_second <= 0:
            raise ValueError(
                "Audio frame must have a positive sample rate and channel count"
            )
        return len(frame.data) / bytes_per_second

    def _combine_frames(self, frames: List[AudioFrame]) -> Optional[Path]:
        """Combine normalized 16-bit PCM frames into a valid WAV file."""
        if not frames:
            return None

        first = frames[0]
        if not first.data:
            logger.warning("Cannot combine an empty first audio frame")
            return None

        for index, frame in enumerate(frames):
            if frame.sample_rate != first.sample_rate:
                raise ValueError(
                    "All buffered audio frames must use the same sample rate; "
                    f"frame 0={first.sample_rate}, frame {index}={frame.sample_rate}"
                )
            if frame.channels != first.channels:
                raise ValueError(
                    "All buffered audio frames must use the same channel count; "
                    f"frame 0={first.channels}, frame {index}={frame.channels}"
                )

        temp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = Path(temp_file.name)

            with wave.open(str(temp_path), "wb") as wav_file:
                wav_file.setnchannels(first.channels)
                wav_file.setsampwidth(self._SAMPLE_WIDTH_BYTES)
                wav_file.setframerate(first.sample_rate)
                for frame in frames:
                    if frame.data:
                        wav_file.writeframes(frame.data)

            logger.debug(
                "Combined %d audio frames into %s (%.2f seconds)",
                len(frames),
                temp_path,
                sum(self._frame_duration_seconds(frame) for frame in frames),
            )
            return temp_path

        except Exception:
            if temp_path and temp_path.exists():
                temp_path.unlink()
            raise

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return [
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "ru",
            "ja",
            "zh",
            "ko",
            "ar",
            "hi",
            "nl",
            "pl",
            "tr",
            "sv",
            "da",
            "no",
            "fi",
            "he",
        ]

    def is_streaming_supported(self) -> bool:
        """Check if streaming processing is supported."""
        return True
