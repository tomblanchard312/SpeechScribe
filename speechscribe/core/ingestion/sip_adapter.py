"""
SIP/WebRTC Audio Ingestion Adapter

TODO: Implement SIP/WebRTC integration for telecom and VoIP audio capture.
"""

import logging
from typing import Any, Dict, Iterator

from ..models import AudioFrame
from .base import IngestionAdapter

logger = logging.getLogger(__name__)


class SIPAdapter(IngestionAdapter):
    """
    Adapter for SIP/WebRTC audio streams.

    TODO: Implement SIP/WebRTC integration for telecom and VoIP audio capture.
    """

    def __init__(
        self,
        session_id: str,
        config: Dict[str, Any],
        sip_uri: str,
        credentials: Dict[str, str],
    ):
        super().__init__(session_id, config)
        self.sip_uri = sip_uri
        self.credentials = credentials
        # TODO: Initialize SIP/WebRTC client

    def connect(self) -> bool:
        """Connect to SIP/WebRTC endpoint."""
        # TODO: Implement SIP/WebRTC connection logic
        logger.warning("SIP adapter not yet implemented")
        return False

    def disconnect(self) -> None:
        """Disconnect from SIP/WebRTC."""
        # TODO: Implement cleanup
        pass

    def get_audio_stream(self) -> Iterator[AudioFrame]:
        """Stream audio from SIP/WebRTC connection."""
        # TODO: Implement real-time audio streaming
        return iter([])

    def is_real_time(self) -> bool:
        return True
