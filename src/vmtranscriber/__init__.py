"""
VMTranscriber - Offline Voice Mail Transcription Tool

A powerful, offline-capable audio transcription tool built with OpenAI's Whisper technology.
"""

__version__ = "1.0.0"
__author__ = "VMTranscriber"
__license__ = "MIT"

from .core import transcribe_audio, batch_transcribe
from .config import Config

__all__ = ["transcribe_audio", "batch_transcribe", "Config"]
