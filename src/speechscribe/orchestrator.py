"""
SpeechScribe Platform Orchestrator

Main entry point that coordinates the four layers:
- Ingestion: Audio source adapters
- Pipeline: Speech processing stages
- Delivery: Output adapters
- Control: Profiles and recommendations
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .control import RecommendationEngine
from .delivery import DeliveryManager
from .ingestion import AdapterFactory
from .pipeline import SpeechPipeline

logger = logging.getLogger(__name__)


class SpeechScribeOrchestrator:
    """
    Main orchestrator for the SpeechScribe platform.

    Coordinates ingestion, processing, and delivery based on user profiles.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.recommender = RecommendationEngine()

    def process_session(
        self,
        profile_name: str,
        source_type: str,
        delivery_config: Dict[str, Any],
        **source_kwargs,
    ) -> str:
        """
        Process a complete transcription session.

        Args:
            profile_name: Name of the processing profile
            source_type: Type of audio source ('file', 'teams', 'zoom', etc.)
            delivery_config: Configuration for output delivery
            **source_kwargs: Source-specific parameters

        Returns:
            Session ID for tracking
        """
        session_id = str(uuid.uuid4())
        logger.info(f"Starting session {session_id} with profile {profile_name}")

        try:
            # Get recommendations
            rec_config = self.recommender.recommend_configuration(profile_name)
            logger.info(
                f"Recommended engine: {rec_config['engine']} for "
                f"environment: {rec_config['environment'].value}"
            )

            # Create ingestion adapter
            adapter = AdapterFactory.create_adapter(
                source_type, session_id, self.config, **source_kwargs
            )

            # Connect to audio source
            if not adapter.connect():
                raise RuntimeError(f"Failed to connect to {source_type} source")

            # Get audio frames
            audio_frames = list(adapter.get_audio_stream())
            logger.info(f"Received {len(audio_frames)} audio frames")

            # Process through pipeline
            pipeline = SpeechPipeline(profile_name, self.config)
            segments = pipeline.process_audio_frames(audio_frames)
            logger.info(f"Generated {len(segments)} transcript segments")

            # Create session metadata
            metadata = SessionMetadata(
                session_id=session_id,
                created_at=datetime.now(),
                source_type=source_type,
                profile_name=profile_name,
                engine_name=rec_config["engine"],
                total_segments=len(segments),
                languages_detected=list(
                    set(s.language for s in segments if s.language)
                ),
                speakers_identified=list(
                    set(s.speaker_id for s in segments if s.speaker_id)
                ),
            )

            # Set up delivery
            delivery_manager = DeliveryManager(self.config)
            self._configure_delivery(delivery_manager, delivery_config)

            # Deliver outputs
            if delivery_manager.deliver_all(segments, metadata):
                logger.info(f"Session {session_id} completed successfully")
            else:
                logger.warning(f"Session {session_id} completed with delivery issues")

            # Cleanup
            adapter.disconnect()

            return session_id

        except Exception as e:
            logger.error(f"Session {session_id} failed: {e}")
            raise

    def _configure_delivery(
        self, delivery_manager: DeliveryManager, delivery_config: Dict[str, Any]
    ):
        """Configure delivery adapters based on config."""

        # File output
        if "output_dir" in delivery_config:
            output_dir = Path(delivery_config["output_dir"])
            delivery_manager.add_file_output(output_dir)

        # Live captions
        if "live_caption_url" in delivery_config:
            delivery_manager.add_live_captions(delivery_config["live_caption_url"])

        # Webhooks
        if "webhook_url" in delivery_config:
            delivery_manager.add_webhook(delivery_config["webhook_url"])

        # TTS
        if delivery_config.get("tts_enabled", False):
            tts_dir = Path(delivery_config.get("tts_output_dir", "tts_output"))
            delivery_manager.add_tts_audio(tts_dir)


# Convenience functions for backward compatibility
def transcribe_file(
    file_path: str,
    profile: str = "local_analyst_workbench",
    output_dir: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Convenience function for file transcription.

    Maintains backward compatibility with existing API.
    """
    from .config import Config

    config = Config()
    orchestrator = SpeechScribeOrchestrator(config.__dict__)

    delivery_config = {}
    if output_dir:
        delivery_config["output_dir"] = output_dir

    return orchestrator.process_session(
        profile_name=profile,
        source_type="file",
        delivery_config=delivery_config,
        file_paths=[Path(file_path)],
    )


def batch_transcribe_files(
    file_paths: List[str],
    profile: str = "sovereign_offline_archive",
    output_dir: Optional[str] = None,
    **kwargs,
) -> List[str]:
    """
    Convenience function for batch file transcription.
    """
    from .config import Config

    config = Config()
    orchestrator = SpeechScribeOrchestrator(config.__dict__)

    delivery_config = {}
    if output_dir:
        delivery_config["output_dir"] = output_dir

    session_ids = []
    for file_path in file_paths:
        session_id = orchestrator.process_session(
            profile_name=profile,
            source_type="file",
            delivery_config=delivery_config,
            file_paths=[Path(file_path)],
        )
        session_ids.append(session_id)

    return session_ids
