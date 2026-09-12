"""SpeechScribe - modular speech intelligence platform."""

__version__ = "2.0.0"
__author__ = "SpeechScribe"
__license__ = "MIT"

from .config import Config
from .control import EngineRegistry, ProfileRegistry, RecommendationEngine
from .core import batch_transcribe, transcribe_audio
from .models import AudioFrame, SessionMetadata, TranscriptSegment
from .orchestrator import SpeechScribeOrchestrator, batch_transcribe_files, transcribe_file
from .planning import (
    ExecutionMode,
    FailureMode,
    PipelinePlan,
    PlanValidationReport,
    PlanValidator,
    StageConfig,
    ValidationIssue,
    ValidationOutcome,
    ValidationReason,
)

__all__ = [
    "transcribe_audio",
    "batch_transcribe",
    "Config",
    "SpeechScribeOrchestrator",
    "transcribe_file",
    "batch_transcribe_files",
    "TranscriptSegment",
    "AudioFrame",
    "SessionMetadata",
    "ProfileRegistry",
    "EngineRegistry",
    "RecommendationEngine",
    "ExecutionMode",
    "FailureMode",
    "StageConfig",
    "PipelinePlan",
    "PlanValidator",
    "PlanValidationReport",
    "ValidationIssue",
    "ValidationOutcome",
    "ValidationReason",
]
