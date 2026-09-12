"""Control plane: profiles, ASR registry, and recommendation logic."""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class Environment(Enum):
    AZURE = "azure"
    OFFLINE = "offline"
    LOCAL = "local"


class LatencyRequirement(Enum):
    REALTIME = "realtime"
    NEAR_REALTIME = "near_realtime"
    BATCH = "batch"


@dataclass
class Profile:
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
            "name": self.name,
            "description": self.description,
            "latency_requirement": self.latency_requirement.value,
            "streaming_required": self.streaming_required,
            "batch_required": self.batch_required,
            "diarization_required": self.diarization_required,
            "translation_required": self.translation_required,
            "translation_languages": self.translation_languages,
            "environment_constraints": [e.value for e in self.environment_constraints],
            "tts_required": self.tts_required,
            "summarization_required": self.summarization_required,
        }


@dataclass
class EngineCapability:
    """Capabilities native to an ASR engine.

    Downstream SpeechScribe stages such as translation, TTS, and external
    diarization do not determine whether an ASR engine can be selected.
    """

    streaming_support: bool = False
    batch_support: bool = True
    diarization_support: bool = False
    translation_support: bool = False
    supported_languages: List[str] = field(default_factory=lambda: ["en"])
    latency_ms: Optional[int] = None
    environment_support: Set[Environment] = field(
        default_factory=lambda: {Environment.OFFLINE}
    )
    tts_support: bool = False
    summarization_support: bool = False

    def supports_profile(self, profile: Profile, environment: Environment) -> bool:
        if environment not in self.environment_support:
            return False
        if profile.streaming_required and not self.streaming_support:
            return False
        if profile.batch_required and not self.batch_support:
            return False
        if (
            profile.latency_requirement == LatencyRequirement.REALTIME
            and self.latency_ms is not None
            and self.latency_ms > 100
        ):
            return False
        if (
            profile.latency_requirement == LatencyRequirement.NEAR_REALTIME
            and self.latency_ms is not None
            and self.latency_ms > 1000
        ):
            return False
        return True


class EngineRegistry:
    def __init__(self):
        self.engines: Dict[str, EngineCapability] = {}
        self._initialize_engines()

    def _initialize_engines(self):
        self.engines["whisper"] = EngineCapability(
            streaming_support=False,
            batch_support=True,
            diarization_support=False,
            translation_support=True,
            supported_languages=[
                "en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh", "ko"
            ],
            latency_ms=5000,
            environment_support={Environment.OFFLINE, Environment.AZURE, Environment.LOCAL},
        )

        self.engines["vibevoice_asr"] = EngineCapability(
            streaming_support=True,
            batch_support=True,
            diarization_support=True,
            translation_support=False,
            supported_languages=[
                "en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh", "ko"
            ],
            latency_ms=None,
            environment_support={Environment.OFFLINE, Environment.AZURE, Environment.LOCAL},
        )

        self.engines["azure_speech"] = EngineCapability(
            streaming_support=True,
            batch_support=True,
            diarization_support=True,
            translation_support=True,
            supported_languages=[
                "en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh", "ko", "ar", "hi"
            ],
            latency_ms=100,
            environment_support={Environment.AZURE},
            tts_support=True,
        )

    def get_available_engines(self, environment: Environment) -> List[str]:
        return [name for name, cap in self.engines.items() if environment in cap.environment_support]

    def find_best_engine(self, profile: Profile, environment: Environment) -> Optional[str]:
        candidates = [
            (name, cap)
            for name, cap in self.engines.items()
            if cap.supports_profile(profile, environment)
        ]
        if not candidates:
            return None

        # Preserve Whisper as the conservative batch default. Realtime profiles
        # naturally exclude it and can select VibeVoice/Azure as appropriate.
        def rank(item):
            name, cap = item
            latency = cap.latency_ms if cap.latency_ms is not None else 999999
            whisper_preference = 0 if name == "whisper" and not profile.streaming_required else 1
            return (whisper_preference, latency, name)

        candidates.sort(key=rank)
        return candidates[0][0]


class ProfileRegistry:
    def __init__(self):
        self.profiles: Dict[str, Profile] = {}
        self._initialize_profiles()

    def _initialize_profiles(self):
        self.profiles["enterprise_meeting_live"] = Profile(
            name="enterprise_meeting_live",
            description="Real-time transcription for enterprise meetings",
            latency_requirement=LatencyRequirement.REALTIME,
            streaming_required=True,
            diarization_required=True,
            environment_constraints={Environment.AZURE},
        )
        self.profiles["enterprise_meeting_post"] = Profile(
            name="enterprise_meeting_post",
            description="Batch processing of recorded enterprise meetings",
            latency_requirement=LatencyRequirement.BATCH,
            batch_required=True,
            diarization_required=True,
            translation_required=True,
            translation_languages=["en", "es", "fr", "de"],
        )
        self.profiles["broadcast_captions"] = Profile(
            name="broadcast_captions",
            description="Live captions for broadcast television",
            latency_requirement=LatencyRequirement.NEAR_REALTIME,
            streaming_required=True,
            translation_required=True,
            translation_languages=["en"],
            environment_constraints={Environment.AZURE},
        )
        self.profiles["telco_call_intelligence"] = Profile(
            name="telco_call_intelligence",
            description="Real-time analysis of telecom calls",
            latency_requirement=LatencyRequirement.REALTIME,
            streaming_required=True,
            diarization_required=True,
            summarization_required=True,
            environment_constraints={Environment.AZURE},
        )
        self.profiles["sovereign_offline_archive"] = Profile(
            name="sovereign_offline_archive",
            description="Offline batch processing with no external dependencies",
            latency_requirement=LatencyRequirement.BATCH,
            batch_required=True,
            diarization_required=True,
            translation_required=True,
            environment_constraints={Environment.OFFLINE},
        )
        self.profiles["local_analyst_workbench"] = Profile(
            name="local_analyst_workbench",
            description="Local desktop analysis with manual speaker assignment",
            latency_requirement=LatencyRequirement.BATCH,
            batch_required=True,
            diarization_required=True,
            translation_required=True,
            environment_constraints={Environment.LOCAL, Environment.OFFLINE},
        )

    def get_profile(self, name: str) -> Optional[Profile]:
        return self.profiles.get(name)

    def list_profiles(self) -> List[str]:
        return list(self.profiles.keys())


class RecommendationEngine:
    def __init__(self):
        self.engine_registry = EngineRegistry()
        self.profile_registry = ProfileRegistry()

    def detect_environment(self) -> Environment:
        if os.getenv("AZURE_ENVIRONMENT") or os.getenv("WEBSITE_INSTANCE_ID"):
            return Environment.AZURE
        if os.getenv("SPEECHSCRIBE_OFFLINE", "").lower() in {"1", "true", "yes"}:
            return Environment.OFFLINE
        return Environment.LOCAL

    def recommend_configuration(self, profile_name: str) -> Dict[str, Any]:
        profile = self.profile_registry.get_profile(profile_name)
        if not profile:
            raise ValueError(f"Unknown profile: {profile_name}")

        environment = self.detect_environment()
        if profile.environment_constraints and environment not in profile.environment_constraints:
            # Keep recommendation explicit rather than silently violating the profile.
            raise ValueError(
                f"Profile {profile_name} is not allowed in {environment.value} environment"
            )

        engine = self.engine_registry.find_best_engine(profile, environment)
        if not engine:
            raise ValueError(
                f"No suitable ASR engine found for profile {profile_name} in {environment.value}"
            )

        return {
            "profile": profile,
            "engine": engine,
            "environment": environment,
            "capabilities": self.engine_registry.engines[engine],
        }
