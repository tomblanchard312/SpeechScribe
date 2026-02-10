"""
Delivery Layer - Output Adapters and Formatters

This layer handles all output delivery:
- Transcript artifacts (JSON, text)
- Subtitle formats (SRT, VTT, TTML)
- Live captions
- Webhooks/event streams
- TTS audio output

No model logic here - only output formatting and delivery.
"""

import abc
import logging
from typing import List, Dict, Any
from pathlib import Path
from datetime import timedelta
import json

from .models import TranscriptSegment, SessionMetadata

logger = logging.getLogger(__name__)


class OutputAdapter(abc.ABC):
    """
    Base class for output delivery adapters.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abc.abstractmethod
    def deliver(self, segments: List[TranscriptSegment],
                metadata: SessionMetadata) -> bool:
        """Deliver the transcript segments."""
        pass


class FileOutputAdapter(OutputAdapter):
    """
    Adapter for file-based output delivery.
    """

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        super().__init__(config)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def deliver(self, segments: List[TranscriptSegment],
                metadata: SessionMetadata) -> bool:
        """Write transcript to files."""
        try:
            base_name = f"{metadata.session_id}_transcript"

            # Write JSON transcript
            json_path = self.output_dir / f"{base_name}.json"
            self._write_json(segments, metadata, json_path)

            # Write SRT subtitles
            srt_path = self.output_dir / f"{base_name}.srt"
            self._write_srt(segments, srt_path)

            # Write plain text
            txt_path = self.output_dir / f"{base_name}.txt"
            self._write_text(segments, txt_path)

            logger.info(f"Delivered transcript to {self.output_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to deliver file output: {e}")
            return False

    def _write_json(self, segments: List[TranscriptSegment],
                    metadata: SessionMetadata, path: Path):
        """Write JSON transcript."""
        data = {
            'metadata': metadata.to_dict(),
            'segments': [s.to_dict() for s in segments]
        }
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _write_srt(self, segments: List[TranscriptSegment], path: Path):
        """Write SRT subtitle file."""
        with path.open('w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                f.write(f"{i}\n")
                start = self._format_timestamp(segment.start_ms)
                end = self._format_timestamp(segment.end_ms)
                f.write(f"{start} --> {end}\n")
                f.write(f"{segment.text}\n\n")

    def _write_text(self, segments: List[TranscriptSegment], path: Path):
        """Write plain text transcript."""
        with path.open('w', encoding='utf-8') as f:
            for segment in segments:
                f.write(f"{segment.text} ")

    def _format_timestamp(self, ms: int) -> str:
        """Format milliseconds to SRT timestamp."""
        td = timedelta(milliseconds=ms)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return "02"


class LiveCaptionAdapter(OutputAdapter):
    """
    Adapter for live caption delivery.

    TODO: Implement real-time caption streaming to displays,
    broadcast systems, or meeting platforms.
    """

    def __init__(self, config: Dict[str, Any], target_url: str):
        super().__init__(config)
        self.target_url = target_url
        # TODO: Initialize streaming connection

    def deliver(self, segments: List[TranscriptSegment],
                metadata: SessionMetadata) -> bool:
        """Stream live captions."""
        # TODO: Implement live streaming
        logger.warning("Live caption delivery not yet implemented")
        return False


class WebhookAdapter(OutputAdapter):
    """
    Adapter for webhook-based delivery.

    Posts transcript updates to configured webhooks.
    """

    def __init__(self, config: Dict[str, Any], webhook_url: str):
        super().__init__(config)
        self.webhook_url = webhook_url

    def deliver(self, segments: List[TranscriptSegment],
                metadata: SessionMetadata) -> bool:
        """Post transcript to webhook."""
        try:
            import requests

            payload = {
                'metadata': metadata.to_dict(),
                'segments': [s.to_dict() for s in segments]
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )

            if response.status_code == 200:
                logger.info(
                    f"Delivered transcript to webhook: {self.webhook_url}")
                return True
            else:
                logger.error(
                    f"Webhook delivery failed: {response.status_code}")
                return False

        except ImportError:
            logger.error("requests library required for webhook delivery")
            return False
        except Exception as e:
            logger.error(f"Webhook delivery failed: {e}")
            return False


class TTSAudioAdapter(OutputAdapter):
    """
    Adapter for Text-to-Speech audio output.

    Generates audio from transcript text.
    """

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        super().__init__(config)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def deliver(self, segments: List[TranscriptSegment],
                metadata: SessionMetadata) -> bool:
        """Generate TTS audio from transcript."""
        try:
            from .voice_synthesis import VoiceSynthesizer

            synthesizer = VoiceSynthesizer(self.config)

            # Combine all text
            full_text = " ".join(s.text for s in segments)

            # Generate audio
            output_path = self.output_dir / f"{metadata.session_id}_tts.wav"
            audio_path = synthesizer.text_to_speech(
                full_text,
                output_path,
                voice_name=self.config.get('tts_voice', 'default'),
                engine=self.config.get('tts_engine', 'coqui_tts')
            )

            logger.info(f"Generated TTS audio: {audio_path}")
            return True

        except Exception as e:
            logger.error(f"TTS delivery failed: {e}")
            return False


class DeliveryManager:
    """
    Manages output delivery to multiple destinations.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adapters: List[OutputAdapter] = []

    def add_file_output(self, output_dir: Path):
        """Add file-based output."""
        self.adapters.append(FileOutputAdapter(self.config, output_dir))

    def add_live_captions(self, target_url: str):
        """Add live caption output."""
        self.adapters.append(LiveCaptionAdapter(self.config, target_url))

    def add_webhook(self, webhook_url: str):
        """Add webhook output."""
        self.adapters.append(WebhookAdapter(self.config, webhook_url))

    def add_tts_audio(self, output_dir: Path):
        """Add TTS audio output."""
        self.adapters.append(TTSAudioAdapter(self.config, output_dir))

    def deliver_all(self, segments: List[TranscriptSegment],
                    metadata: SessionMetadata) -> bool:
        """Deliver to all configured adapters."""
        success = True

        for adapter in self.adapters:
            try:
                if not adapter.deliver(segments, metadata):
                    success = False
            except Exception as e:
                logger.error(f"Adapter delivery failed: {e}")
                success = False

        return success
