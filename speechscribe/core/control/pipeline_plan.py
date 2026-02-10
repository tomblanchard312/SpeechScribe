"""
Control Plane - Pipeline Plan

Represents the execution plan derived from a profile and environment.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from .profiles import Profile
from .environment import Environment

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Pipeline execution mode."""
    STREAMING = "streaming"
    BATCH = "batch"


class FailureMode(Enum):
    """Failure behavior for pipeline stages."""
    REQUIRED = "required"        # Stage must succeed, pipeline fails if it doesn't
    OPTIONAL = "optional"        # Stage can fail, pipeline continues without it
    BEST_EFFORT = "best_effort"  # Stage attempts to run, but failure is tolerated
    # Stage failure degrades output quality but doesn't stop pipeline
    DEGRADES = "degrades"


@dataclass
class StageConfig:
    """Configuration for a pipeline stage."""
    stage_name: str
    engine_name: str
    enabled: bool = True
    config: Optional[Dict] = None
    failure_mode: FailureMode = FailureMode.REQUIRED


@dataclass
class PipelinePlan:
    """
    Execution plan derived from profile and environment.

    This object represents an execution contract.

    Changes must be backward compatible or versioned.

    Orchestrator behavior depends on it.

    Profiles, tests, UI previews, and dry-run validation will all depend on it soon.

    Encapsulates all decisions about how to execute the pipeline:
    - Which stages are enabled
    - Execution mode (streaming vs batch)
    - Which engine to use for each stage
    - Failure semantics for each stage
    """

    profile: Profile
    environment: Environment
    execution_mode: ExecutionMode
    stages: List[StageConfig]

    profile: Profile
    environment: Environment
    execution_mode: ExecutionMode
    stages: List[StageConfig]

    def __post_init__(self):
        """Validate the plan after initialization."""
        self._set_default_failure_modes()
        self._validate_plan()

    def _validate_plan(self):
        """Validate that the plan is consistent."""
        # ASR must always be enabled
        asr_stage = self.get_stage_config('asr')
        if not asr_stage or not asr_stage.enabled:
            raise ValueError("ASR stage must be enabled in pipeline plan")

        # Check that execution mode is compatible with profile
        if (self.execution_mode == ExecutionMode.STREAMING and
                not self.profile.streaming_required):
            logger.warning(
                f"Profile {self.profile.name} doesn't require streaming, "
                f"but plan specifies streaming mode")

        if (self.execution_mode == ExecutionMode.BATCH and
                not self.profile.batch_required):
            logger.warning(
                f"Profile {self.profile.name} doesn't require batch processing, "
                "but plan specifies batch mode")

    def _set_default_failure_modes(self):
        """Set default failure modes for stages based on requirements."""
        for stage in self.stages:
            # Always set default failure modes based on stage type
            if stage.stage_name == 'diarization':
                stage.failure_mode = FailureMode.OPTIONAL
            elif stage.stage_name == 'translation':
                stage.failure_mode = FailureMode.BEST_EFFORT
            elif stage.stage_name == 'tts':
                stage.failure_mode = FailureMode.DEGRADES
            # ASR remains REQUIRED by default
            # summarization remains REQUIRED by default

    def get_enabled_stages(self) -> List[str]:
        """Get list of enabled stage names."""
        return [stage.stage_name for stage in self.stages if stage.enabled]

    def get_stage_config(self, stage_name: str) -> Optional[StageConfig]:
        """Get configuration for a specific stage."""
        for stage in self.stages:
            if stage.stage_name == stage_name:
                return stage
        return None

    def get_engine_for_stage(self, stage_name: str) -> Optional[str]:
        """Get the engine name for a specific stage."""
        config = self.get_stage_config(stage_name)
        return config.engine_name if config else None

    def is_stage_enabled(self, stage_name: str) -> bool:
        """Check if a stage is enabled."""
        config = self.get_stage_config(stage_name)
        return config.enabled if config else False

    def refresh_failure_modes(self):
        """Refresh default failure modes for all stages."""
        self._set_default_failure_modes()

    def to_dict(self) -> Dict:
        """Convert plan to dictionary for serialization."""
        return {
            'profile': self.profile.to_dict(),
            'environment': self.environment.value,
            'execution_mode': self.execution_mode.value,
            'stages': [
                {
                    'stage_name': stage.stage_name,
                    'engine_name': stage.engine_name,
                    'enabled': stage.enabled,
                    'config': stage.config,
                    'failure_mode': stage.failure_mode.value
                }
                for stage in self.stages
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PipelinePlan':
        """Create plan from dictionary."""

        profile_data = data['profile']
        profile = Profile(
            name=profile_data['name'],
            description=profile_data['description'],
            latency_requirement=profile_data['latency_requirement'],
            streaming_required=profile_data.get('streaming_required', False),
            batch_required=profile_data.get('batch_required', False),
            diarization_required=profile_data.get(
                'diarization_required', False),
            translation_required=profile_data.get(
                'translation_required', False),
            translation_languages=profile_data.get(
                'translation_languages', []),
            environment_constraints=set(),
            tts_required=profile_data.get('tts_required', False),
            summarization_required=profile_data.get(
                'summarization_required', False)
        )

        environment = Environment(data['environment'])
        execution_mode = ExecutionMode(data['execution_mode'])

        stages = [
            StageConfig(
                stage_name=stage_data['stage_name'],
                engine_name=stage_data['engine_name'],
                enabled=stage_data.get('enabled', True),
                config=stage_data.get('config'),
                failure_mode=FailureMode(
                    stage_data.get('failure_mode', 'required'))
            )
            for stage_data in data['stages']
        ]

        return cls(
            profile=profile,
            environment=environment,
            execution_mode=execution_mode,
            stages=stages
        )
