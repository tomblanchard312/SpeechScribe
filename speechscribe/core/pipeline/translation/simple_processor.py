"""
Translation Module - Simple Processor

Simple translation processor.
"""

import logging
from typing import List

from googletrans import Translator

from ...models.transcript import TranscriptSegment
from .base import TranslationConfig, TranslationProcessor

logger = logging.getLogger(__name__)


class SimpleTranslationProcessor(TranslationProcessor):
    """
    Simple translation processor.

    Uses Google Translate API for translation.
    """

    def __init__(self, config: TranslationConfig):
        super().__init__(config)
        self.translator = Translator()
        # Simple language mappings for demonstration
        self.language_names = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
        }

    def process(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """
        Translate segments to target languages.
        """
        logger.info(f"Processing translations for {len(segments)} segments")

        target_langs = self.config.target_languages or ["en"]

        for segment in segments:
            translations = {}

            # Detect source language (placeholder)
            source_lang = segment.language or "en"

            for target_lang in target_langs:
                if target_lang != source_lang:
                    try:
                        # Real translation using Google Translate
                        translated = self.translator.translate(
                            segment.text, src=source_lang, dest=target_lang
                        )
                        translations[target_lang] = translated.text
                    except Exception as e:
                        logger.warning(
                            f"Translation failed for {source_lang} to {target_lang}: {e}"
                        )
                        # Fallback to placeholder
                        translations[target_lang] = self._placeholder_translate(
                            segment.text, source_lang, target_lang
                        )

            # Add translations to segment metadata
            if not hasattr(segment, "translations"):
                segment.translations = {}
            segment.translations.update(translations)

        logger.info("Translation processing completed")
        return segments

    def _placeholder_translate(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        """Placeholder translation function as fallback."""
        # This is just for demonstration - real implementation would translate
        source = self.language_names.get(source_lang, source_lang)
        target = self.language_names.get(target_lang, target_lang)
        translated_text = f"[Translated from {source} to {target}: {text}]"
        return translated_text

    def get_supported_languages(self) -> List[str]:
        """Get list of supported target languages."""
        return [
            "en",
            "es",
            "fr",
            "de",
            "zh",
            "ja",
            "ko",
            "ar",
            "hi",
            "pt",
            "ru",
            "it",
            "nl",
        ]
