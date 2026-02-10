"""
Translation Module

Text translation processing.
"""

from .base import TranslationProcessor, TranslationConfig
from .simple_processor import SimpleTranslationProcessor

__all__ = [
    'TranslationProcessor',
    'TranslationConfig',
    'SimpleTranslationProcessor'
]
