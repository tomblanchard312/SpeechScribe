"""
Summarization Module

Text summarization processing.
"""

from .base import SummarizationProcessor, SummarizationConfig
from .simple_processor import SimpleSummarizationProcessor

__all__ = [
    'SummarizationProcessor',
    'SummarizationConfig',
    'SimpleSummarizationProcessor'
]
