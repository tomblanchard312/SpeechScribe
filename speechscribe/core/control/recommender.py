"""
Control Plane - Recommendation Engine

Recommends optimal processing configuration based on requirements.
"""

import logging
from typing import Optional

from .environment import Environment, EnvironmentDetector
from .model_registry import EngineRegistry
from .pipeline_plan import ExecutionMode, PipelinePlan, StageConfig
from .profiles import Profile, ProfileRegistry

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Recommends optimal processing configuration based on requirements.
    """

    def __init__(self):
        self.engine_registry = EngineRegistry()
        self.profile_registry = ProfileRegistry()
        self.environment_detector = EnvironmentDetector()

    def recommend_configuration(self, profile_name: str) -> PipelinePlan:
        """
        Recommend pipeline plan for a profile.

        Returns PipelinePlan with:
        - Profile and environment
        - Execution mode (streaming/batch)
        - Enabled stages with engine selection
        """
        profile = self.profile_registry.get_profile(profile_name)
        if not profile:
            raise ValueError(f"Unknown profile: {profile_name}")

        environment = self.environment_detector.detect()

        # Determine execution mode
        execution_mode = self._determine_execution_mode(profile)

        # Build stage configurations
        stages = self._build_stage_configs(profile, environment)

        return PipelinePlan(
            profile=profile,
            environment=environment,
            execution_mode=execution_mode,
            stages=stages,
        )

    def _determine_execution_mode(self, profile: Profile) -> ExecutionMode:
        """Determine execution mode based on profile requirements."""
        if profile.streaming_required:
            return ExecutionMode.STREAMING
        elif profile.batch_required:
            return ExecutionMode.BATCH
        else:
            # Default to batch for profiles without specific mode requirements
            return ExecutionMode.BATCH

    def _build_stage_configs(
        self, profile: Profile, environment: Environment
    ) -> list[StageConfig]:
        """Build stage configurations with engine selection."""
        stages = []

        # ASR stage - always required
        asr_engine = self._select_engine_for_stage("asr", profile, environment)
        if asr_engine:
            stages.append(
                StageConfig(stage_name="asr", engine_name=asr_engine, enabled=True)
            )
        else:
            raise ValueError(
                f"No suitable ASR engine found for profile {profile.name} in "
                f"{environment.value}"
            )

        # Diarization stage
        if profile.diarization_required:
            diarization_engine = self._select_engine_for_stage(
                "diarization", profile, environment
            )
            if diarization_engine:
                stages.append(
                    StageConfig(
                        stage_name="diarization",
                        engine_name=diarization_engine,
                        enabled=True,
                    )
                )

        # Translation stage
        if profile.translation_required:
            translation_engine = self._select_engine_for_stage(
                "translation", profile, environment
            )
            if translation_engine:
                stages.append(
                    StageConfig(
                        stage_name="translation",
                        engine_name=translation_engine,
                        enabled=True,
                        config={"target_languages": profile.translation_languages},
                    )
                )

        # Summarization stage
        if profile.summarization_required:
            summarization_engine = self._select_engine_for_stage(
                "summarization", profile, environment
            )
            if summarization_engine:
                stages.append(
                    StageConfig(
                        stage_name="summarization",
                        engine_name=summarization_engine,
                        enabled=True,
                    )
                )

        # TTS stage
        if profile.tts_required:
            tts_engine = self._select_engine_for_stage("tts", profile, environment)
            if tts_engine:
                stages.append(
                    StageConfig(stage_name="tts", engine_name=tts_engine, enabled=True)
                )

        return stages

    def _select_engine_for_stage(
        self, stage_name: str, profile: Profile, environment: Environment
    ) -> Optional[str]:
        """
        Select the best engine for a specific pipeline stage.

        This creates a temporary profile-like requirement for the stage
        and finds the best matching engine.
        """
        # Create stage-specific requirements
        stage_requirements = self._get_stage_requirements(stage_name, profile)

        # Find best engine for these requirements
        return self.engine_registry.find_best_engine(stage_requirements, environment)

    def _get_stage_requirements(self, stage_name: str, profile: Profile) -> Profile:
        """
        Get the requirements for a specific stage based on the profile.

        This creates a minimal profile that captures the requirements
        relevant to the specific stage.
        """
        # Base requirements from profile
        base_requirements = {
            "streaming_required": profile.streaming_required,
            "batch_required": profile.batch_required,
            "latency_requirement": profile.latency_requirement,
            "environment_constraints": profile.environment_constraints,
        }

        # Stage-specific requirements
        if stage_name == "asr":
            # ASR engines need basic transcription capability
            pass  # Uses base requirements

        elif stage_name == "diarization":
            # Diarization engines need diarization capability
            base_requirements["diarization_required"] = True

        elif stage_name == "translation":
            # Translation engines need translation capability
            base_requirements["translation_required"] = True

        elif stage_name == "summarization":
            # Summarization engines need summarization capability
            base_requirements["summarization_required"] = True

        elif stage_name == "tts":
            # TTS engines need TTS capability
            base_requirements["tts_required"] = True

        # Create a temporary profile for engine selection
        return Profile(
            name=f"{profile.name}_{stage_name}",
            description=f"Stage requirements for {stage_name}",
            latency_requirement=base_requirements["latency_requirement"],
            streaming_required=base_requirements.get("streaming_required", False),
            batch_required=base_requirements.get("batch_required", False),
            diarization_required=base_requirements.get("diarization_required", False),
            translation_required=base_requirements.get("translation_required", False),
            translation_languages=profile.translation_languages,
            environment_constraints=profile.environment_constraints,
            tts_required=base_requirements.get("tts_required", False),
            summarization_required=base_requirements.get(
                "summarization_required", False
            ),
        )
