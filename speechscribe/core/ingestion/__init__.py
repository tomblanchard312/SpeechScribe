"""
SpeechScribe Ingestion Layer

Audio source adapters for various input types.
"""

from .base import IngestionAdapter
from .file_adapter import FileAdapter
from .teams_adapter import TeamsAdapter
from .zoom_adapter import ZoomAdapter
from .discord_adapter import DiscordAdapter
from .sip_adapter import SIPAdapter

__all__ = [
    "IngestionAdapter",
    "FileAdapter",
    "TeamsAdapter",
    "ZoomAdapter",
    "DiscordAdapter",
    "SIPAdapter"
]
