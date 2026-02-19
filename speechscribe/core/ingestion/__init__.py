"""
SpeechScribe Ingestion Layer

Audio source adapters for various input types.
"""

from .base import IngestionAdapter
from .discord_adapter import DiscordAdapter
from .file_adapter import FileAdapter
from .sip_adapter import SIPAdapter
from .teams_adapter import TeamsAdapter
from .zoom_adapter import ZoomAdapter

__all__ = [
    "IngestionAdapter",
    "FileAdapter",
    "TeamsAdapter",
    "ZoomAdapter",
    "DiscordAdapter",
    "SIPAdapter",
]
