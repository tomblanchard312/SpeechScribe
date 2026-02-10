"""
ASR Module

Automatic Speech Recognition processing.
"""

from .base import ASRProcessor, ASRConfig
from .whisper_processor import WhisperASRProcessor

__all__ = [
    'ASRProcessor',
    'ASRConfig',
    'WhisperASRProcessor'
]
