"""
Pipeline Orchestrator

Coordinates the speech processing pipeline stages.
"""

import logging
from typing import Dict, Any, List, Iterator
from dataclasses import dataclass

from ..models.transcript import TranscriptSegment, AudioFrame, ProcessingResult
from ..control import PipelinePlan, ExecutionMode, StageConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Context passed through pipeline stages."""
    plan: PipelinePlan
    audio_frames: Iterator[AudioFrame]
    session_metadata: Dict[str, Any]
    processing_options: Dict[str, Any]


class PipelineOrchestrator:
    """
    Orchestrates the speech processing pipeline.

    Coordinates ASR, diarization, translation, summarization, and TTS stages
    based on the pipeline plan.
    """

    def __init__(self, plan: PipelinePlan):
        self.plan = plan

        # Initialize pipeline stages based on plan
        self.enabled_stages = plan.get_enabled_stages()

    def process_stream(
        self, context: PipelineContext
    ) -> Iterator[ProcessingResult]:
        """
        Process audio stream through the pipeline.

        Args:
            context: Pipeline context with audio frames and metadata

        Yields:
            ProcessingResult objects as they become available
        """
        logger.info(
            f"Starting pipeline processing with stages: "
            f"{self.enabled_stages}")

        # Validate execution mode
        if self.plan.execution_mode != ExecutionMode.STREAMING:
            raise ValueError(
                f"Pipeline plan specifies {self.plan.execution_mode.value} mode, "
                "but streaming processing requested")

        # Initialize processing state
        current_segments: List[TranscriptSegment] = []

        # Process through each enabled stage
        for stage_name in self.enabled_stages:
            logger.debug(f"Processing stage: {stage_name}")

            engine_name = self.plan.get_engine_for_stage(stage_name)
            if not engine_name:
                logger.warning(
                    f"No engine specified for stage {stage_name}, skipping")
                continue

            if stage_name == 'asr':
                current_segments = self._process_asr_stream(
                    context, current_segments, engine_name)
            elif stage_name == 'diarization':
                current_segments = self._process_diarization_stream(
                    context, current_segments, engine_name)
            elif stage_name == 'translation':
                current_segments = self._process_translation_stream(
                    context, current_segments, engine_name)
            elif stage_name == 'summarization':
                current_segments = self._process_summarization_stream(
                    context, current_segments, engine_name)
            elif stage_name == 'tts':
                current_segments = self._process_tts_stream(
                    context, current_segments, engine_name)

        # Yield final results
        for segment in current_segments:
            yield ProcessingResult(
                segment=segment,
                stage='final',
                metadata=context.session_metadata
            )

    def process_batch(
        self, context: PipelineContext
    ) -> List[ProcessingResult]:
        """
        Process audio batch through the pipeline.

        Args:
            context: Pipeline context with audio frames and metadata

        Returns:
            List of ProcessingResult objects
        """
        logger.info(
            f"Starting batch processing with stages: "
            f"{self.enabled_stages}")

        # Validate execution mode
        if self.plan.execution_mode != ExecutionMode.BATCH:
            raise ValueError(
                f"Pipeline plan specifies {self.plan.execution_mode.value} mode, "
                "but batch processing requested")

        # Initialize processing state
        current_segments: List[TranscriptSegment] = []

        # Process through each enabled stage
        for stage_name in self.enabled_stages:
            logger.debug(f"Processing stage: {stage_name}")

            engine_name = self.plan.get_engine_for_stage(stage_name)
            if not engine_name:
                logger.warning(
                    f"No engine specified for stage {stage_name}, skipping")
                continue

            if stage_name == 'asr':
                current_segments = self._process_asr_batch(
                    context, current_segments, engine_name)
            elif stage_name == 'diarization':
                current_segments = self._process_diarization_batch(
                    context, current_segments, engine_name)
            elif stage_name == 'translation':
                current_segments = self._process_translation_batch(
                    context, current_segments, engine_name)
            elif stage_name == 'summarization':
                current_segments = self._process_summarization_batch(
                    context, current_segments, engine_name)
            elif stage_name == 'tts':
                current_segments = self._process_tts_batch(
                    context, current_segments, engine_name)

        # Return final results
        return [
            ProcessingResult(
                segment=segment,
                stage='final',
                metadata=context.session_metadata
            )
            for segment in current_segments
        ]

    # Processing methods for each stage with engine selection
    # These will be implemented by importing from the respective stage modules

    def _process_asr_stream(
        self, context: PipelineContext, segments: List[TranscriptSegment],
        engine_name: str
    ) -> List[TranscriptSegment]:
        """Process ASR stage with specified engine."""
        # TODO: Import and instantiate ASR processor for the specified engine
        logger.warning(
            f"ASR streaming processing not yet implemented for engine: "
            f"{engine_name}")
        return segments

    def _process_diarization_stream(
        self, context: PipelineContext, segments: List[TranscriptSegment],
        engine_name: str
    ) -> List[TranscriptSegment]:
        """Process diarization stage with specified engine."""
        # TODO: Import and instantiate diarization processor for the specified engine
        logger.warning(
            f"Diarization streaming processing not yet implemented for engine: "
            f"{engine_name}")
        return segments

    def _process_translation_stream(
        self, context: PipelineContext, segments: List[TranscriptSegment],
        engine_name: str
    ) -> List[TranscriptSegment]:
        """Process translation stage with specified engine."""
        # TODO: Import and instantiate translation processor for the specified engine
        logger.warning(
            f"Translation streaming processing not yet implemented for engine: "
            f"{engine_name}")
        return segments

    def _process_summarization_stream(
        self, context: PipelineContext, segments: List[TranscriptSegment],
        engine_name: str
    ) -> List[TranscriptSegment]:
        """Process summarization stage with specified engine."""
        # TODO: Import and instantiate summarization processor for the specified engine
        logger.warning(
            f"Summarization streaming processing not yet implemented for engine: "
            f"{engine_name}")
        return segments

    def _process_tts_stream(
        self, context: PipelineContext, segments: List[TranscriptSegment],
        engine_name: str
    ) -> List[TranscriptSegment]:
        """Process TTS stage with specified engine."""
        # TODO: Import and instantiate TTS processor for the specified engine
        logger.warning(
            f"TTS streaming processing not yet implemented for engine: "
            f"{engine_name}")
        return segments

    # Batch processing versions
    def _process_asr_batch(
        self, context: PipelineContext, segments: List[TranscriptSegment],
        engine_name: str
    ) -> List[TranscriptSegment]:
        """Process ASR stage for batch with specified engine."""
        # TODO: Import and instantiate ASR processor for the specified engine
        logger.warning(
            f"ASR batch processing not yet implemented for engine: "
            f"{engine_name}")
        return segments

    def _process_diarization_batch(
        self, context: PipelineContext, segments: List[TranscriptSegment],
        engine_name: str
    ) -> List[TranscriptSegment]:
        """Process diarization stage for batch with specified engine."""
        # TODO: Import and instantiate diarization processor for the specified engine
        message = ("Diarization batch processing not yet implemented for engine: "
                   f"{engine_name}")
        logger.warning(message)
        return segments

    def _process_translation_batch(
        self, context: PipelineContext, segments: List[TranscriptSegment],
        engine_name: str
    ) -> List[TranscriptSegment]:
        """Process translation stage for batch with specified engine."""
        # TODO: Import and instantiate translation processor for the specified engine
        message = ("Translation batch processing not yet implemented for engine: "
                   f"{engine_name}")
        logger.warning(message)
        return segments

    def _process_summarization_batch(
        self, context: PipelineContext, segments: List[TranscriptSegment],
        engine_name: str
    ) -> List[TranscriptSegment]:
        """Process summarization stage for batch with specified engine."""
        # TODO: Import and instantiate summarization processor for the specified engine
        message = ("Summarization batch processing not yet implemented for engine: "
                   f"{engine_name}")
        logger.warning(message)
        return segments

    def _process_tts_batch(
        self, context: PipelineContext, segments: List[TranscriptSegment],
        engine_name: str
    ) -> List[TranscriptSegment]:
        """Process TTS stage for batch with specified engine."""
        # TODO: Import and instantiate TTS processor for the specified engine
        message = ("TTS batch processing not yet implemented for engine: "
                   f"{engine_name}")
        logger.warning(message)
        return segments

    def preview_execution(self) -> 'PipelinePreview':
        """
        Generate a dry-run preview of pipeline execution.

        Returns a structured summary of what would happen during execution
        without actually processing any audio.

        Returns:
            PipelinePreview: Structured execution preview
        """
        return PipelinePreview.from_plan(self.plan)

    @classmethod
    def preview(cls, plan: PipelinePlan) -> 'PipelinePreview':
        """
        Generate a dry-run preview for a given pipeline plan.

        This static method answers: "What will happen if I run this profile here?"

        No audio is processed, returns comprehensive execution summary including:
        - Enabled stages and their engines
        - Expected latency class
        - Forbidden stages (if any)
        - Failure semantics for each stage

        Args:
            plan: Pipeline plan to preview

        Returns:
            PipelinePreview: Structured execution preview
        """
        return PipelinePreview.from_plan(plan)


@dataclass
class StagePreview:
    """Preview information for a single pipeline stage."""
    stage_name: str
    engine_name: str
    enabled: bool
    description: str
    expected_inputs: List[str]
    expected_outputs: List[str]
    failure_modes: List[str]
    resource_requirements: Dict[str, Any]


@dataclass
class PipelinePreview:
    """Structured preview of pipeline execution."""
    profile_name: str
    profile_description: str
    environment: str
    execution_mode: str
    enabled_stages: List[str]
    forbidden_stages: List[str]  # Stages that cannot run in this environment
    stage_previews: List[StagePreview]
    execution_flow: List[str]
    failure_behavior: Dict[str, Any]
    resource_summary: Dict[str, Any]
    expected_latency_class: str

    @classmethod
    def from_plan(cls, plan: PipelinePlan) -> 'PipelinePreview':
        """Create preview from pipeline plan."""
        stage_previews = []
        execution_flow = []
        failure_modes = []
        resource_requirements = {}
        forbidden_stages = []

        # Build stage previews
        for stage_config in plan.stages:
            if not stage_config.enabled:
                continue

            stage_preview = cls._create_stage_preview(stage_config, plan)
            stage_previews.append(stage_preview)

            # Add to execution flow
            execution_flow.append(
                f"{stage_config.stage_name} ({stage_config.engine_name})")

            # Collect failure modes and resources
            failure_modes.extend(stage_preview.failure_modes)
            resource_requirements.update(stage_preview.resource_requirements)

            # Check if stage is forbidden in this environment
            if cls._is_stage_forbidden(stage_config, plan):
                forbidden_stages.append(stage_config.stage_name)

        # Determine expected latency class
        expected_latency_class = cls._determine_latency_class(plan)

        # Determine failure behavior
        failure_behavior = {
            'error_handling': 'fail_fast',  # Could be configurable
            'rollback_capable': False,  # For now, no rollback
            'partial_results': plan.execution_mode == ExecutionMode.BATCH,
            'common_failure_modes': list(set(failure_modes)),
            'stage_failure_semantics': {
                stage.stage_name: stage.failure_mode.value
                for stage in plan.stages if stage.enabled
            }
        }

        # Resource summary

        # Resource summary
        resource_summary = {
            'estimated_memory_mb': cls._estimate_memory_usage(stage_previews),
            'estimated_time_per_minute': cls._estimate_processing_time(stage_previews),
            'external_dependencies': cls._identify_dependencies(stage_previews),
            'concurrency_safe': plan.execution_mode == ExecutionMode.BATCH
        }

        return cls(
            profile_name=plan.profile.name,
            profile_description=plan.profile.description,
            environment=plan.environment.value,
            execution_mode=plan.execution_mode.value,
            enabled_stages=plan.get_enabled_stages(),
            forbidden_stages=forbidden_stages,
            stage_previews=stage_previews,
            execution_flow=execution_flow,
            failure_behavior=failure_behavior,
            resource_summary=resource_summary,
            expected_latency_class=expected_latency_class
        )

    @staticmethod
    def _create_stage_preview(stage_config: 'StageConfig', plan: PipelinePlan) -> StagePreview:
        """Create preview for a specific stage."""
        stage_name = stage_config.stage_name
        engine_name = stage_config.engine_name

        # Stage-specific information
        if stage_name == 'asr':
            description = "Automatic Speech Recognition - converts audio to text"
            expected_inputs = ["Audio frames (WAV, MP3, etc.)"]
            expected_outputs = ["Transcript segments with timestamps"]
            failure_modes = [
                "Audio format not supported",
                "Model loading failure",
                "Out of memory",
                "Corrupted audio data"
            ]
            resources = {
                f"{stage_name}_memory_mb": 1024,  # Base estimate
                f"{stage_name}_cpu_cores": 1
            }

        elif stage_name == 'diarization':
            description = "Speaker Diarization - identifies and labels speakers"
            expected_inputs = ["Transcript segments"]
            expected_outputs = ["Transcript segments with speaker labels"]
            failure_modes = [
                "Insufficient audio quality",
                "Too many speakers for model",
                "Model not available in environment"
            ]
            resources = {
                f"{stage_name}_memory_mb": 512,
                f"{stage_name}_cpu_cores": 1
            }

        elif stage_name == 'translation':
            description = "Text Translation - translates transcript to target languages"
            expected_inputs = ["Transcript segments"]
            expected_outputs = ["Transcript segments with translations"]
            failure_modes = [
                "Translation service unavailable",
                "Unsupported language pair",
                "Network connectivity issues"
            ]
            resources = {
                f"{stage_name}_memory_mb": 256,
                f"{stage_name}_network_required": plan.environment.value == 'azure'
            }

        elif stage_name == 'summarization':
            description = "Text Summarization - generates summary of transcript"
            expected_inputs = ["Transcript segments"]
            expected_outputs = ["Summary text"]
            failure_modes = [
                "Model not available",
                "Insufficient text length",
                "Memory constraints"
            ]
            resources = {
                f"{stage_name}_memory_mb": 512,
                f"{stage_name}_cpu_cores": 1
            }

        elif stage_name == 'tts':
            description = "Text-to-Speech - converts text back to audio"
            expected_inputs = ["Transcript segments"]
            expected_outputs = ["Audio files"]
            failure_modes = [
                "TTS engine not available",
                "Unsupported voice/language",
                "Audio output format issues"
            ]
            resources = {
                f"{stage_name}_memory_mb": 256,
                f"{stage_name}_cpu_cores": 1
            }

        else:
            description = f"Unknown stage: {stage_name}"
            expected_inputs = ["Unknown"]
            expected_outputs = ["Unknown"]
            failure_modes = ["Unknown stage"]
            resources = {}

        return StagePreview(
            stage_name=stage_name,
            engine_name=engine_name,
            enabled=stage_config.enabled,
            description=description,
            expected_inputs=expected_inputs,
            expected_outputs=expected_outputs,
            failure_modes=failure_modes,
            resource_requirements=resources
        )

    @staticmethod
    def _estimate_memory_usage(stage_previews: List[StagePreview]) -> int:
        """Estimate total memory usage in MB."""
        total_memory = 256  # Base system memory
        for preview in stage_previews:
            total_memory += preview.resource_requirements.get(
                f"{preview.stage_name}_memory_mb", 0)
        return total_memory

    @staticmethod
    def _estimate_processing_time(stage_previews: List[StagePreview]) -> float:
        """Estimate processing time per minute of audio in seconds."""
        # Rough estimates based on typical processing times
        time_estimates = {
            'asr': 60.0,  # 1:1 processing ratio
            'diarization': 30.0,  # Faster than ASR
            'translation': 10.0,  # Fast text processing
            'summarization': 5.0,  # Very fast
            'tts': 15.0  # Slower than real-time
        }

        total_time = 0.0
        for preview in stage_previews:
            total_time += time_estimates.get(preview.stage_name, 10.0)

        return total_time

    @staticmethod
    def _identify_dependencies(stage_previews: List[StagePreview]) -> List[str]:
        """Identify external dependencies."""
        dependencies = []
        for preview in stage_previews:
            if preview.resource_requirements.get(f"{preview.stage_name}_network_required"):
                dependencies.append(f"Network access for {preview.stage_name}")
            if "azure" in preview.engine_name.lower():
                dependencies.append("Azure cloud services")
            if "whisper" in preview.engine_name.lower():
                dependencies.append("Local Whisper model")

        return list(set(dependencies))

    @staticmethod
    def _is_stage_forbidden(stage_config: StageConfig, plan: PipelinePlan) -> bool:
        """Check if a stage is forbidden in the current environment."""
        # For now, no stages are forbidden, but this could check:
        # - Environment constraints
        # - License restrictions
        # - Hardware requirements
        return False

    @staticmethod
    def _determine_latency_class(plan: PipelinePlan) -> str:
        """Determine expected latency class for the pipeline."""
        if plan.execution_mode == ExecutionMode.STREAMING:
            if plan.profile.latency_requirement.value == 'realtime':
                return "realtime (< 100ms)"
            elif plan.profile.latency_requirement.value == 'near_realtime':
                return "near-realtime (< 1s)"
            else:
                return "streaming (variable)"
        else:
            return "batch (no latency requirement)"

    def to_dict(self) -> Dict[str, Any]:
        """Convert preview to dictionary."""
        return {
            'profile': {
                'name': self.profile_name,
                'description': self.profile_description
            },
            'environment': self.environment,
            'execution_mode': self.execution_mode,
            'expected_latency_class': self.expected_latency_class,
            'enabled_stages': self.enabled_stages,
            'forbidden_stages': self.forbidden_stages,
            'execution_flow': self.execution_flow,
            'stages': [
                {
                    'name': sp.stage_name,
                    'engine': sp.engine_name,
                    'description': sp.description,
                    'inputs': sp.expected_inputs,
                    'outputs': sp.expected_outputs,
                    'failure_modes': sp.failure_modes
                }
                for sp in self.stage_previews
            ],
            'failure_behavior': self.failure_behavior,
            'resource_summary': self.resource_summary
        }

    def format_human_readable(self) -> str:
        """Format preview as human-readable text."""
        lines = []

        # Header
        lines.append("=" * 60)
        lines.append("PIPELINE EXECUTION PREVIEW")
        lines.append("=" * 60)
        lines.append("")

        # Profile and environment
        lines.append("CONFIGURATION:")
        lines.append(f"  Profile: {self.profile_name}")
        lines.append(f"  Description: {self.profile_description}")
        lines.append(f"  Environment: {self.environment}")
        lines.append(f"  Execution Mode: {self.execution_mode}")
        lines.append(f"  Expected Latency: {self.expected_latency_class}")
        lines.append("")

        # Enabled stages
        lines.append("ENABLED STAGES:")
        for stage in self.enabled_stages:
            stage_preview = next(
                (sp for sp in self.stage_previews if sp.stage_name == stage), None)
            if stage_preview:
                lines.append(f"  • {stage} ({stage_preview.engine_name})")
                lines.append(f"    {stage_preview.description}")
        lines.append("")

        # Execution flow
        lines.append("EXECUTION FLOW:")
        for i, step in enumerate(self.execution_flow, 1):
            lines.append(f"  {i}. {step}")
        lines.append("")

        # Forbidden stages
        if self.forbidden_stages:
            lines.append("FORBIDDEN STAGES:")
            for stage in self.forbidden_stages:
                lines.append(f"  • {stage} (cannot run in {self.environment})")
            lines.append("")

        # Stage details
        lines.append("STAGE DETAILS:")
        for preview in self.stage_previews:
            lines.append(
                f"  {preview.stage_name.upper()} ({preview.engine_name}):")
            lines.append(f"    Description: {preview.description}")
            lines.append(f"    Inputs: {', '.join(preview.expected_inputs)}")
            lines.append(f"    Outputs: {', '.join(preview.expected_outputs)}")
            if preview.failure_modes:
                lines.append(
                    f"    Potential Failures: {', '.join(preview.failure_modes)}")
            lines.append("")

        # Failure behavior
        lines.append("FAILURE BEHAVIOR:")
        lines.append(
            f"  Error Handling: {self.failure_behavior['error_handling']}")
        lines.append(
            f"  Rollback Capable: {self.failure_behavior['rollback_capable']}")
        lines.append(
            f"  Partial Results: {self.failure_behavior['partial_results']}")
        if self.failure_behavior['common_failure_modes']:
            lines.append(
                f"  Common Issues: {', '.join(self.failure_behavior['common_failure_modes'])}")
        lines.append("")
        lines.append("STAGE FAILURE SEMANTICS:")
        for stage_name, failure_mode in self.failure_behavior['stage_failure_semantics'].items():
            lines.append(f"  • {stage_name}: {failure_mode}")
        lines.append("")

        # Resource summary
        lines.append("RESOURCE REQUIREMENTS:")
        lines.append(
            f"  Estimated Memory: {self.resource_summary['estimated_memory_mb']} MB")
        lines.append(
            f"  Processing Time: {self.resource_summary['estimated_time_per_minute']:.1f}s per minute of audio")
        lines.append(
            f"  Concurrency Safe: {self.resource_summary['concurrency_safe']}")
        if self.resource_summary['external_dependencies']:
            lines.append(
                f"  Dependencies: {', '.join(self.resource_summary['external_dependencies'])}")
        lines.append("")

        return "\n".join(lines)
