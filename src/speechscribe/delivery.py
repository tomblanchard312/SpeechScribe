"""Delivery adapters for transcripts, captions, webhooks, and optional TTS."""

import abc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .models import SessionMetadata, TranscriptSegment

logger = logging.getLogger(__name__)


class OutputAdapter(abc.ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abc.abstractmethod
    def deliver(self, segments: List[TranscriptSegment], metadata: SessionMetadata) -> bool:
        pass


class FileOutputAdapter(OutputAdapter):
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        super().__init__(config)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def deliver(self, segments: List[TranscriptSegment], metadata: SessionMetadata) -> bool:
        try:
            base_name = f"{metadata.session_id}_transcript"
            self._write_json(segments, metadata, self.output_dir / f"{base_name}.json")
            self._write_srt(segments, self.output_dir / f"{base_name}.srt")
            self._write_vtt(segments, self.output_dir / f"{base_name}.vtt")
            self._write_text(segments, self.output_dir / f"{base_name}.txt")
            logger.info("Delivered transcript to %s", self.output_dir)
            return True
        except Exception as exc:
            logger.exception("Failed to deliver file output: %s", exc)
            return False

    def _write_json(self, segments, metadata, path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "metadata": metadata.to_dict(),
                    "segments": [segment.to_dict() for segment in segments],
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

    def _write_srt(self, segments: List[TranscriptSegment], path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for index, segment in enumerate(segments, 1):
                handle.write(f"{index}\n")
                handle.write(
                    f"{self._format_timestamp(segment.start_ms, ',')} --> "
                    f"{self._format_timestamp(segment.end_ms, ',')}\n"
                )
                prefix = f"{segment.speaker_label}: " if segment.speaker_label else ""
                handle.write(f"{prefix}{segment.text}\n\n")

    def _write_vtt(self, segments: List[TranscriptSegment], path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            handle.write("WEBVTT\n\n")
            for segment in segments:
                handle.write(
                    f"{self._format_timestamp(segment.start_ms, '.')} --> "
                    f"{self._format_timestamp(segment.end_ms, '.')}\n"
                )
                prefix = f"{segment.speaker_label}: " if segment.speaker_label else ""
                handle.write(f"{prefix}{segment.text}\n\n")

    def _write_text(self, segments: List[TranscriptSegment], path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for segment in segments:
                if segment.speaker_label:
                    handle.write(f"{segment.speaker_label}: ")
                handle.write(f"{segment.text}\n")

    @staticmethod
    def _format_timestamp(milliseconds: int, decimal_separator: str) -> str:
        milliseconds = max(0, int(milliseconds))
        hours, remainder = divmod(milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        seconds, millis = divmod(remainder, 1_000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}{decimal_separator}{millis:03d}"


class LiveCaptionAdapter(OutputAdapter):
    def __init__(self, config: Dict[str, Any], target_url: str):
        super().__init__(config)
        self.target_url = target_url

    def deliver(self, segments: List[TranscriptSegment], metadata: SessionMetadata) -> bool:
        logger.warning("Live caption delivery is not yet implemented")
        return False


class WebhookAdapter(OutputAdapter):
    def __init__(self, config: Dict[str, Any], webhook_url: str):
        super().__init__(config)
        self.webhook_url = webhook_url

    def deliver(self, segments: List[TranscriptSegment], metadata: SessionMetadata) -> bool:
        try:
            import requests

            response = requests.post(
                self.webhook_url,
                json={
                    "metadata": metadata.to_dict(),
                    "segments": [segment.to_dict() for segment in segments],
                },
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            response.raise_for_status()
            return True
        except Exception as exc:
            logger.error("Webhook delivery failed: %s", exc)
            return False


class TTSAudioAdapter(OutputAdapter):
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        super().__init__(config)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def deliver(self, segments: List[TranscriptSegment], metadata: SessionMetadata) -> bool:
        full_text = " ".join(segment.text for segment in segments).strip()
        if not full_text:
            logger.warning("TTS skipped because transcript is empty")
            return False

        output_path = self.output_dir / f"{metadata.session_id}_tts.wav"
        engine_name = self.config.get("tts_engine", "coqui_tts")

        try:
            if engine_name == "vibevoice_realtime":
                from .engines import VibeVoiceRealtimeTTSEngine

                engine = VibeVoiceRealtimeTTSEngine(self.config)
                audio_path = engine.synthesize(full_text, output_path)
            else:
                from .voice_synthesis import VoiceSynthesizer

                synthesizer = VoiceSynthesizer(self.config)
                audio_path = synthesizer.text_to_speech(
                    full_text,
                    output_path,
                    voice_name=self.config.get("tts_voice", "default"),
                    engine=engine_name,
                )

            logger.info("Generated TTS audio with %s: %s", engine_name, audio_path)
            return True
        except Exception as exc:
            logger.exception("TTS delivery failed with %s: %s", engine_name, exc)
            return False


class DeliveryManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adapters: List[OutputAdapter] = []

    def add_file_output(self, output_dir: Path):
        self.adapters.append(FileOutputAdapter(self.config, output_dir))

    def add_live_captions(self, target_url: str):
        self.adapters.append(LiveCaptionAdapter(self.config, target_url))

    def add_webhook(self, webhook_url: str):
        self.adapters.append(WebhookAdapter(self.config, webhook_url))

    def add_tts_audio(self, output_dir: Path):
        self.adapters.append(TTSAudioAdapter(self.config, output_dir))

    def deliver_all(self, segments: List[TranscriptSegment], metadata: SessionMetadata) -> bool:
        success = True
        for adapter in self.adapters:
            try:
                if not adapter.deliver(segments, metadata):
                    success = False
            except Exception as exc:
                logger.exception("Adapter delivery failed: %s", exc)
                success = False
        return success
