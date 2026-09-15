import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SpeechT5Plugin:
    """SpeechT5 TTS plugin for text-to-speech synthesis."""

    def __init__(self, model_name: str = "microsoft/speecht5_tts"):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._vocoder = None
        self._speaker_embeddings = None

    def _load_model(self):
        """Lazy load the SpeechT5 model and vocoder."""
        if self._model is None:
            try:
                from transformers import (
                    SpeechT5Processor,
                    SpeechT5ForTextToSpeech,
                    SpeechT5HifiGan,
                )
                from datasets import load_dataset
                import torch

                self._processor = SpeechT5Processor.from_pretrained(self.model_name)
                self._model = SpeechT5ForTextToSpeech.from_pretrained(self.model_name)
                self._vocoder = SpeechT5HifiGan.from_pretrained(
                    "microsoft/speecht5_hifigan"
                )

                # Load default speaker embeddings
                embeddings_dataset = load_dataset(
                    "Matthijs/cmu-arctic-xvectors", split="validation"
                )
                self._speaker_embeddings = torch.tensor(
                    embeddings_dataset[7306]["xvector"]
                ).unsqueeze(0)

                logger.info("Loaded SpeechT5 model: %s", self.model_name)
            except ImportError as exc:
                logger.error("Required packages not installed: %s", exc)
                raise RuntimeError("transformers and datasets packages are required")

    def synthesize(
        self,
        text: str,
        output_path: str,
        speaker_embedding: Optional[object] = None,
        sample_rate: int = 16000,
    ) -> Path:
        """Convert text to speech and save to file."""
        self._load_model()
        import torch
        import soundfile as sf

        inputs = self._processor(text=text, return_tensors="pt")
        speaker_emb = (
            speaker_embedding
            if speaker_embedding is not None
            else self._speaker_embeddings
        )

        with torch.no_grad():
            speech = self._model.generate_speech(
                inputs["input_ids"], speaker_emb, vocoder=self._vocoder
            )

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        sf.write(str(output_file), speech.numpy(), samplerate=sample_rate)
        logger.info("Generated speech saved to: %s", output_file)

        return output_file

    def text_to_speech(
        self,
        text: str,
        output_path: str,
        voice_id: Optional[int] = None,
    ) -> Path:
        """Public API for text-to-speech synthesis."""
        self._load_model()

        if voice_id is not None:
            try:
                from datasets import load_dataset
                import torch

                embeddings_dataset = load_dataset(
                    "Matthijs/cmu-arctic-xvectors", split="validation"
                )
                speaker_embedding = torch.tensor(
                    embeddings_dataset[voice_id]["xvector"]
                ).unsqueeze(0)
            except Exception as exc:
                logger.warning(
                    "Failed to load voice %d, using default: %s", voice_id, exc
                )
                speaker_embedding = None
        else:
            speaker_embedding = None

        return self.synthesize(text, output_path, speaker_embedding=speaker_embedding)
