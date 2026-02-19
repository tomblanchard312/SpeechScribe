"""
Control Plane Module

Provides profile management, environment detection, and engine selection.
"""

from .environment import Environment, EnvironmentDetector
from .model_registry import EngineCapability, EngineRegistry
from .pipeline_plan import ExecutionMode, FailureMode, PipelinePlan, StageConfig
from .plan_validator import (
    PlanValidationReport,
    PlanValidator,
    ValidationIssue,
    ValidationOutcome,
    ValidationReason,
)
from .profiles import LatencyRequirement, Profile, ProfileRegistry
from .recommender import RecommendationEngine

__all__ = [
    "EnvironmentDetector",
    "Environment",
    "Profile",
    "LatencyRequirement",
    "ProfileRegistry",
    "EngineCapability",
    "EngineRegistry",
    "RecommendationEngine",
    "PipelinePlan",
    "ExecutionMode",
    "StageConfig",
    "FailureMode",
    "PlanValidator",
    "PlanValidationReport",
    "ValidationOutcome",
    "ValidationReason",
    "ValidationIssue",
]
