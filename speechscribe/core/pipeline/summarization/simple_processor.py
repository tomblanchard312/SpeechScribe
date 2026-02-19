"""
Summarization Module - Simple Processor

Simple extractive summarization.
"""

import logging
from collections import Counter
from typing import List

from ...models.transcript import TranscriptSegment
from .base import SummarizationConfig, SummarizationProcessor

logger = logging.getLogger(__name__)


class SimpleSummarizationProcessor(SummarizationProcessor):
    """
    Simple extractive summarization processor.

    Uses basic text analysis to extract key sentences.
    """

    def __init__(self, config: SummarizationConfig):
        super().__init__(config)

    def process(self, segments: List[TranscriptSegment]) -> str:
        """
        Generate extractive summary from transcript segments.
        """
        logger.info(f"Generating summary from {len(segments)} segments")

        if not segments:
            return ""

        # Combine all text
        full_text = " ".join(segment.text for segment in segments if segment.text)

        if not full_text.strip():
            return ""

        # Simple extractive summarization
        sentences = self._split_sentences(full_text)
        if not sentences:
            # Rough character limit
            return full_text[: self.config.max_length * 10]

        # Score sentences
        sentence_scores = self._score_sentences(sentences)

        # Select top sentences
        top_sentences = self._select_top_sentences(sentences, sentence_scores)

        # Combine into summary
        summary = " ".join(top_sentences)

        # Trim to length
        summary = self._trim_to_length(summary)

        logger.info(f"Summary generated: {len(summary)} characters")
        return summary

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting."""
        # Basic sentence splitting on periods, question marks, exclamation points
        import re

        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _score_sentences(self, sentences: List[str]) -> List[float]:
        """Score sentences based on word frequency."""
        # Simple TF-IDF like scoring
        all_words = []
        for sentence in sentences:
            words = self._tokenize(sentence.lower())
            all_words.extend(words)

        word_freq = Counter(all_words)

        scores = []
        for sentence in sentences:
            words = self._tokenize(sentence.lower())
            score = sum(word_freq[word] for word in words) / len(words) if words else 0
            scores.append(score)

        return scores

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re

        return re.findall(r"\b\w+\b", text)

    def _select_top_sentences(
        self, sentences: List[str], scores: List[float]
    ) -> List[str]:
        """Select top sentences for summary."""
        # Sort by score and take top N
        sentence_pairs = list(zip(sentences, scores))
        sentence_pairs.sort(key=lambda x: x[1], reverse=True)

        # Take enough sentences to reach target length
        selected = []
        current_length = 0

        for sentence, score in sentence_pairs:
            # Rough word limit
            if current_length + len(sentence) > self.config.max_length * 10:
                break
            selected.append(sentence)
            current_length += len(sentence)

        return selected

    def _trim_to_length(self, text: str) -> str:
        """Trim text to approximate word length."""
        words = text.split()
        if len(words) <= self.config.max_length:
            return text

        return " ".join(words[: self.config.max_length]) + "..."

    def get_supported_styles(self) -> List[str]:
        """Get list of supported summarization styles."""
        return ["extractive"]
