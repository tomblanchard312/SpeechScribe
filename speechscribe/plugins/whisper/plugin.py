import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WhisperPlugin:
    """Whisper ASR plugin for speech-to-text transcription."""

    def __init__(self, model: str = "base", device: str = "cpu"):
        self.model_name = model
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazy load the Whisper model."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self.model_name, device=self.device, compute_type="float32"
                )
                logger.info("Loaded Whisper model: %s", self.model_name)
            except ImportError:
                logger.error("faster-whisper not installed")
                raise RuntimeError("faster-whisper package is required")

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs,
    ) -> Dict[str, Any]:
        """Transcribe audio file to text."""
        self._load_model()
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        segments, info = self._model.transcribe(
            str(audio_file), language=language, task=task, **kwargs
        )

        result_segments = []
        full_text = []

        for segment in segments:
            result_segments.append(
                {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
            )
            full_text.append(segment.text)

        return {
            "text": " ".join(full_text),
            "segments": result_segments,
            "language": info.language,
            "language_probability": info.language_probability,
        }

    def translate(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Translate audio to English."""
        return self.transcribe(audio_path, task="translate", **kwargs)

    def detect_language(self, audio_path: str) -> Dict[str, float]:
        """Detect the language of the audio."""
        self._load_model()
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        segments, info = self._model.transcribe(str(audio_file), language=None)
        # Consume first segment to get language info
        next(segments, None)

        return {
            "language": info.language,
            "probability": info.language_probability,
        }
