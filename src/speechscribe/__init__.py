"""
SpeechScribe - Comprehensive Speech Processing Tool

A powerful, offline-capable speech processing tool built with OpenAI's Whisper technology and advanced TTS capabilities.
"""

__version__ = "1.0.0"
__author__ = "SpeechScribe"
__license__ = "MIT"

from .core import transcribe_audio, batch_transcribe
from .config import Config

__all__ = ["transcribe_audio", "batch_transcribe", "Config"]
