"""Compatibility re-exports for the old ``speechscribe.core.control`` path."""

from ..control import (
    EngineCapability,
    EngineRegistry,
    Environment,
    LatencyRequirement,
    Profile,
    ProfileRegistry,
    RecommendationEngine,
)
from ..planning import (
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
    "Environment",
    "LatencyRequirement",
    "Profile",
    "EngineCapability",
    "EngineRegistry",
    "ProfileRegistry",
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
