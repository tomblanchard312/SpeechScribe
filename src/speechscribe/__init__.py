"""
SpeechScribe - Comprehensive Speech Processing Tool

A powerful, offline-capable speech processing tool built with
OpenAI's Whisper technology and advanced TTS capabilities.
"""

__version__ = "1.0.0"
__author__ = "SpeechScribe"
__license__ = "MIT"

from .config import Config
from .control import EngineRegistry, ProfileRegistry, RecommendationEngine

# Legacy API (backward compatibility)
from .core import batch_transcribe, transcribe_audio
from .models import AudioFrame, SessionMetadata, TranscriptSegment

# New platform architecture
from .orchestrator import (
    SpeechScribeOrchestrator,
    batch_transcribe_files,
    transcribe_file,
)

__all__ = [
    # Legacy
    "transcribe_audio",
    "batch_transcribe",
    "Config",
    # New platform
    "SpeechScribeOrchestrator",
    "transcribe_file",
    "batch_transcribe_files",
    "TranscriptSegment",
    "AudioFrame",
    "SessionMetadata",
    "ProfileRegistry",
    "EngineRegistry",
    "RecommendationEngine",
]
