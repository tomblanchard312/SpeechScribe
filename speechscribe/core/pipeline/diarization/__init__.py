"""
Diarization Module

Speaker diarization processing.
"""

from .base import DiarizationProcessor, DiarizationConfig
from .simple_processor import SimpleDiarizationProcessor

__all__ = [
    'DiarizationProcessor',
    'DiarizationConfig',
    'SimpleDiarizationProcessor'
]
