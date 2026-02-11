"""
Control Plane - Profile Definitions

User profiles that define processing requirements.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .environment import Environment

logger = logging.getLogger(__name__)


class LatencyRequirement(Enum):
    """Latency requirements for processing."""

    REALTIME = "realtime"  # < 100ms
    NEAR_REALTIME = "near_realtime"  # < 1s
    BATCH = "batch"  # No latency requirement


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
        self.profiles["enterprise_meeting_live"] = Profile(
            name="enterprise_meeting_live",
            description="Real-time transcription for enterprise meetings",
            latency_requirement=LatencyRequirement.REALTIME,
            streaming_required=True,
            diarization_required=True,
            translation_required=False,
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
            environment_constraints=set(),  # Any environment
        )

        # Broadcast profiles
        self.profiles["broadcast_captions"] = Profile(
            name="broadcast_captions",
            description="Live captions for broadcast television",
            latency_requirement=LatencyRequirement.NEAR_REALTIME,
            streaming_required=True,
            diarization_required=False,
            translation_required=True,
            translation_languages=["en"],
            environment_constraints={Environment.AZURE},
        )

        # Telco profiles
        self.profiles["telco_call_intelligence"] = Profile(
            name="telco_call_intelligence",
            description="Real-time analysis of telecom calls",
            latency_requirement=LatencyRequirement.REALTIME,
            streaming_required=True,
            diarization_required=True,
            translation_required=False,
            summarization_required=True,
            environment_constraints={Environment.AZURE},
        )

        # Sovereign/offline profiles
        self.profiles["sovereign_offline_archive"] = Profile(
            name="sovereign_offline_archive",
            description="Offline batch processing with no external dependencies",
            latency_requirement=LatencyRequirement.BATCH,
            batch_required=True,
            diarization_required=True,
            translation_required=True,
            environment_constraints={Environment.OFFLINE},
        )

        # Local desktop profiles
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
        """Get a profile by name."""
        return self.profiles.get(name)

    def list_profiles(self) -> List[str]:
        """List all available profile names."""
        return list(self.profiles.keys())
