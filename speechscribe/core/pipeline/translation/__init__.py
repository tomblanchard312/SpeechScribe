"""
Translation Module

Text translation processing.
"""

from .base import TranslationConfig, TranslationProcessor
from .simple_processor import SimpleTranslationProcessor

__all__ = ["TranslationProcessor", "TranslationConfig", "SimpleTranslationProcessor"]
