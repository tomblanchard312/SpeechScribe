"""
ASR Module

Automatic Speech Recognition processing.
"""

from .base import ASRConfig, ASRProcessor
from .whisper_processor import WhisperASRProcessor

__all__ = ["ASRProcessor", "ASRConfig", "WhisperASRProcessor"]
