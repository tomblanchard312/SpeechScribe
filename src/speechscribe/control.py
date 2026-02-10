"""
Control Plane - Profiles, Model Registry, and Recommendation Engine

This layer manages:
- User profiles that define processing requirements
- Model/engine registry with capabilities
- Recommendation engine that selects optimal engines
- Environment awareness (Azure vs offline vs local)
"""

import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import os

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environments."""
    AZURE = "azure"
    OFFLINE = "offline"
    LOCAL = "local"


class LatencyRequirement(Enum):
    """Latency requirements for processing."""
    REALTIME = "realtime"      # < 100ms
    NEAR_REALTIME = "near_realtime"  # < 1s
    BATCH = "batch"           # No latency requirement


@dataclass
class Profile:
    """
    User profile defining processing requirements.

    Profiles abstract away model selection and let users specify
    what they need rather than how to achieve it.
    """
    name: str
    description: str
    latency_requirement: LatencyRequirement
    streaming_required: bool = False
    batch_required: bool = False
    diarization_required: bool = False
    translation_required: bool = False
    translation_languages: List[str] = field(default_factory=list)
    environment_constraints: Set[Environment] = field(default_factory=set)
    tts_required: bool = False
    summarization_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'latency_requirement': self.latency_requirement.value,
            'streaming_required': self.streaming_required,
            'batch_required': self.batch_required,
            'diarization_required': self.diarization_required,
            'translation_required': self.translation_required,
            'translation_languages': self.translation_languages,
            'environment_constraints': [
                e.value for e in self.environment_constraints
            ],
            'tts_required': self.tts_required,
            'summarization_required': self.summarization_required
        }


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

    def supports_profile(
        self,
        profile: Profile,
        environment: Environment
    ) -> bool:
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
        if (profile.latency_requirement == LatencyRequirement.REALTIME
                and self.latency_ms and self.latency_ms > 100):
            return False

        if (profile.latency_requirement == LatencyRequirement.NEAR_REALTIME
                and self.latency_ms and self.latency_ms > 1000):
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
            supported_languages=[
                'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'zh', 'ko'
            ],  # Many languages
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
            supported_languages=[
                'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'zh',
                'ko', 'ar', 'hi'
            ],  # Many languages
            latency_ms=100,  # Very low latency
            environment_support={Environment.AZURE},  # Cloud only
            tts_support=True,
            summarization_support=False
        )

    def get_available_engines(self, environment: Environment) -> List[str]:
        """Get engines available in the current environment."""
        return [name for name, cap in self.engines.items()
                if environment in cap.environment_support]

    def find_best_engine(
        self,
        profile: Profile,
        environment: Environment
    ) -> Optional[str]:
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


class ProfileRegistry:
    """
    Registry of predefined profiles.
    """

    def __init__(self):
        self.profiles: Dict[str, Profile] = {}
        self._initialize_profiles()

    def _initialize_profiles(self):
        """Initialize predefined profiles."""

        # Enterprise meeting profiles
        self.profiles['enterprise_meeting_live'] = Profile(
            name='enterprise_meeting_live',
            description='Real-time transcription for enterprise meetings',
            latency_requirement=LatencyRequirement.REALTIME,
            streaming_required=True,
            diarization_required=True,
            translation_required=False,
            environment_constraints={Environment.AZURE}
        )

        self.profiles['enterprise_meeting_post'] = Profile(
            name='enterprise_meeting_post',
            description='Batch processing of recorded enterprise meetings',
            latency_requirement=LatencyRequirement.BATCH,
            batch_required=True,
            diarization_required=True,
            translation_required=True,
            translation_languages=['en', 'es', 'fr', 'de'],
            environment_constraints=set()  # Any environment
        )

        # Broadcast profiles
        self.profiles['broadcast_captions'] = Profile(
            name='broadcast_captions',
            description='Live captions for broadcast television',
            latency_requirement=LatencyRequirement.NEAR_REALTIME,
            streaming_required=True,
            diarization_required=False,
            translation_required=True,
            translation_languages=['en'],
            environment_constraints={Environment.AZURE}
        )

        # Telco profiles
        self.profiles['telco_call_intelligence'] = Profile(
            name='telco_call_intelligence',
            description='Real-time analysis of telecom calls',
            latency_requirement=LatencyRequirement.REALTIME,
            streaming_required=True,
            diarization_required=True,
            translation_required=False,
            summarization_required=True,
            environment_constraints={Environment.AZURE}
        )

        # Sovereign/offline profiles
        self.profiles['sovereign_offline_archive'] = Profile(
            name='sovereign_offline_archive',
            description='Offline batch processing with no external dependencies',
            latency_requirement=LatencyRequirement.BATCH,
            batch_required=True,
            diarization_required=True,
            translation_required=True,
            environment_constraints={Environment.OFFLINE}
        )

        # Local desktop profiles
        self.profiles['local_analyst_workbench'] = Profile(
            name='local_analyst_workbench',
            description='Local desktop analysis with manual speaker assignment',
            latency_requirement=LatencyRequirement.BATCH,
            batch_required=True,
            diarization_required=True,
            translation_required=True,
            environment_constraints={Environment.LOCAL, Environment.OFFLINE}
        )

    def get_profile(self, name: str) -> Optional[Profile]:
        """Get a profile by name."""
        return self.profiles.get(name)

    def list_profiles(self) -> List[str]:
        """List all available profile names."""
        return list(self.profiles.keys())


class RecommendationEngine:
    """
    Recommends optimal processing configuration based on requirements.
    """

    def __init__(self):
        self.engine_registry = EngineRegistry()
        self.profile_registry = ProfileRegistry()

    def detect_environment(self) -> Environment:
        """Detect the current deployment environment."""
        # Check for Azure environment variables
        if os.getenv('AZURE_ENVIRONMENT') or os.getenv('WEBSITE_INSTANCE_ID'):
            return Environment.AZURE

        # Check for offline indicators (no internet, etc.)
        # For now, assume local unless Azure detected
        return Environment.LOCAL

    def recommend_configuration(self, profile_name: str) -> Dict[str, Any]:
        """
        Recommend engine and configuration for a profile.

        Returns dict with:
        - profile: Profile object
        - engine: recommended engine name
        - environment: detected environment
        - capabilities: engine capabilities
        """
        profile = self.profile_registry.get_profile(profile_name)
        if not profile:
            raise ValueError(f"Unknown profile: {profile_name}")

        environment = self.detect_environment()
        engine = self.engine_registry.find_best_engine(profile, environment)

        if not engine:
            raise ValueError(
                f"No suitable engine found for profile {profile_name} "
                f"in {environment.value}")

        capabilities = self.engine_registry.engines[engine]

        return {
            'profile': profile,
            'engine': engine,
            'environment': environment,
            'capabilities': capabilities
        }
