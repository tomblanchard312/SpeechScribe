"""
TTS Module

Text-to-Speech processing.
"""

from .base import TTSProcessor, TTSConfig
from .simple_processor import SimpleTTSProcessor

__all__ = ["TTSProcessor", "TTSConfig", "SimpleTTSProcessor"]
