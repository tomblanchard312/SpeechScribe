"""
Summarization Module

Text summarization processing.
"""

from .base import SummarizationConfig, SummarizationProcessor
from .simple_processor import SimpleSummarizationProcessor

__all__ = [
    "SummarizationProcessor",
    "SummarizationConfig",
    "SimpleSummarizationProcessor",
]
