"""
Diarization Module

Speaker diarization processing.
"""

from .base import DiarizationConfig, DiarizationProcessor
from .simple_processor import SimpleDiarizationProcessor

__all__ = ["DiarizationProcessor", "DiarizationConfig", "SimpleDiarizationProcessor"]
