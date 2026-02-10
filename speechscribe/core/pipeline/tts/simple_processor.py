"""
TTS Module - Simple Processor

Simple TTS processor using pyttsx3.
"""

import logging
from typing import List, Optional
from pathlib import Path
import tempfile

from .base import TTSProcessor, TTSConfig
from ...models.transcript import TranscriptSegment

logger = logging.getLogger(__name__)


class SimpleTTSProcessor(TTSProcessor):
    """
    Simple TTS processor using pyttsx3.

    Uses system TTS engines for speech synthesis.
    """

    def __init__(self, config: TTSConfig):
        super().__init__(config)
        self.engine = None
        self._init_engine()

    def _init_engine(self):
        """Initialize the TTS engine."""
        try:
            import pyttsx3

            self.engine = pyttsx3.init()

            # Configure voice
            voices = self.engine.getProperty("voices")
            if voices:
                # Try to find a voice matching the language
                for voice in voices:
                    if (
                        self.config.language.lower() in voice.languages
                        or self.config.language.lower() in voice.name.lower()
                    ):
                        self.engine.setProperty("voice", voice.id)
                        break

            # Set speech rate
            rate = self.engine.getProperty("rate")
            self.engine.setProperty("rate", int(rate * self.config.speed))

        except ImportError:
            logger.warning("pyttsx3 not installed. TTS functionality will be limited.")
            self.engine = None

    def synthesize(self, text: str, output_path: Optional[Path] = None) -> bytes:
        """
        Synthesize speech from text.
        """
        if not self.engine:
            raise RuntimeError("TTS engine not available")

        logger.info(f"Synthesizing text: {text[:50]}...")

        if output_path:
            # Save to file
            self.engine.save_to_file(text, str(output_path))
            self.engine.runAndWait()

            # Read back the file
            with open(output_path, "rb") as f:
                return f.read()
        else:
            # Generate in memory (not supported by pyttsx3, so use temp file)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = Path(temp_file.name)

            try:
                self.engine.save_to_file(text, str(temp_path))
                self.engine.runAndWait()

                with open(temp_path, "rb") as f:
                    audio_data = f.read()

                return audio_data
            finally:
                if temp_path.exists():
                    temp_path.unlink()

    def synthesize_segments(
        self, segments: List[TranscriptSegment], output_dir: Path
    ) -> List[Path]:
        """
        Synthesize speech for multiple segments.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_files = []

        for i, segment in enumerate(segments):
            if not segment.text:
                continue

            output_path = output_dir / f"segment_{i:03d}.wav"
            try:
                self.synthesize(segment.text, output_path)
                output_files.append(output_path)
                logger.debug(f"Generated TTS for segment {i}")
            except Exception as e:
                logger.error(f"Failed to synthesize segment {i}: {e}")

        logger.info(f"TTS synthesis completed: {len(output_files)} files generated")
        return output_files

    def get_available_voices(self) -> List[str]:
        """Get list of available voices."""
        if not self.engine:
            return []

        voices = self.engine.getProperty("voices")
        return [voice.name for voice in voices] if voices else []
