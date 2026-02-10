"""
SpeechScribe - Comprehensive Speech Processing Tool

A powerful, offline-capable speech processing tool built with
OpenAI's Whisper technology and advanced TTS capabilities.
"""

__version__ = "1.0.0"
__author__ = "SpeechScribe"
__license__ = "MIT"

# Legacy API (backward compatibility)
from .core import transcribe_audio, batch_transcribe
from .config import Config

# New platform architecture
from .orchestrator import (
    SpeechScribeOrchestrator,
    transcribe_file,
    batch_transcribe_files,
)
from .models import TranscriptSegment, AudioFrame, SessionMetadata
from .control import ProfileRegistry, EngineRegistry, RecommendationEngine

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
