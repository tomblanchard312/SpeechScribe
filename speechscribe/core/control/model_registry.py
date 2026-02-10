"""
Control Plane - Model Registry

Registry of available speech processing engines and their capabilities.
"""

import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from .environment import Environment
from .profiles import Profile, LatencyRequirement

logger = logging.getLogger(__name__)


@dataclass
class EngineCapability:
    """
    Capabilities of a speech processing engine.
    """
    streaming_support: bool = False
    batch_support: bool = True
    diarization_support: bool = False
    translation_support: bool = False
    supported_languages: List[str] = field(default_factory=lambda: ['en'])
    latency_ms: Optional[int] = None  # Typical latency in milliseconds
    environment_support: Set[Environment] = field(
        default_factory=lambda: {Environment.OFFLINE})
    tts_support: bool = False
    summarization_support: bool = False

    def supports_profile(self, profile: Profile, environment: Environment) -> bool:
        """Check if this engine can satisfy the profile requirements."""
        if environment not in self.environment_support:
            return False

        if profile.streaming_required and not self.streaming_support:
            return False

        if profile.batch_required and not self.batch_support:
            return False

        if profile.diarization_required and not self.diarization_support:
            return False

        if profile.translation_required and not self.translation_support:
            return False

        if profile.tts_required and not self.tts_support:
            return False

        if profile.summarization_required and not self.summarization_support:
            return False

        # Check latency requirements
        if profile.latency_requirement == LatencyRequirement.REALTIME and self.latency_ms and self.latency_ms > 100:
            return False

        if profile.latency_requirement == LatencyRequirement.NEAR_REALTIME and self.latency_ms and self.latency_ms > 1000:
            return False

        return True


class EngineRegistry:
    """
    Registry of available speech processing engines and their capabilities.
    """

    def __init__(self):
        self.engines: Dict[str, EngineCapability] = {}
        self._initialize_engines()

    def _initialize_engines(self):
        """Initialize known engines with their capabilities."""

        # Whisper (OpenAI)
        self.engines['whisper'] = EngineCapability(
            streaming_support=False,  # Batch only
            batch_support=True,
            diarization_support=False,
            translation_support=True,
            supported_languages=['en', 'es', 'fr', 'de', 'it',
                                 'pt', 'ru', 'ja', 'zh', 'ko'],  # Many languages
            latency_ms=5000,  # ~5 seconds for typical audio
            environment_support={Environment.OFFLINE,
                                 Environment.AZURE, Environment.LOCAL},
            tts_support=False,
            summarization_support=False
        )

        # VibeVoice-ASR (hypothetical advanced engine)
        self.engines['vibevoice_asr'] = EngineCapability(
            streaming_support=True,
            batch_support=True,
            diarization_support=True,
            translation_support=True,
            supported_languages=['en', 'es', 'fr', 'de', 'zh'],
            latency_ms=200,  # Low latency
            environment_support={Environment.OFFLINE, Environment.AZURE},
            tts_support=False,
            summarization_support=False
        )

        # Azure Speech Services
        self.engines['azure_speech'] = EngineCapability(
            streaming_support=True,
            batch_support=True,
            diarization_support=True,
            translation_support=True,
            supported_languages=['en', 'es', 'fr', 'de', 'it', 'pt',
                                 'ru', 'ja', 'zh', 'ko', 'ar', 'hi'],  # Many languages
            latency_ms=100,  # Very low latency
            environment_support={Environment.AZURE},  # Cloud only
            tts_support=True,
            summarization_support=False
        )

    def get_available_engines(self, environment: Environment) -> List[str]:
        """Get engines available in the current environment."""
        return [name for name, cap in self.engines.items()
                if environment in cap.environment_support]

    def find_best_engine(self, profile: Profile, environment: Environment) -> Optional[str]:
        """Find the best engine for a profile in the given environment."""
        candidates = []

        for engine_name, capability in self.engines.items():
            if capability.supports_profile(profile, environment):
                candidates.append((engine_name, capability))

        if not candidates:
            return None

        # Simple ranking: prefer lower latency, then by name for consistency
        candidates.sort(key=lambda x: (x[1].latency_ms or 999999, x[0]))
        return candidates[0][0]
