"""
Control Plane Module

Provides profile management, environment detection, and engine selection.
"""

from .environment import EnvironmentDetector, Environment
from .profiles import Profile, LatencyRequirement, ProfileRegistry
from .model_registry import EngineCapability, EngineRegistry
from .recommender import RecommendationEngine
from .pipeline_plan import PipelinePlan, ExecutionMode, StageConfig, FailureMode
from .plan_validator import (
    PlanValidator, PlanValidationReport, ValidationOutcome,
    ValidationReason, ValidationIssue
)

__all__ = [
    'EnvironmentDetector',
    'Environment',
    'Profile',
    'LatencyRequirement',
    'ProfileRegistry',
    'EngineCapability',
    'EngineRegistry',
    'RecommendationEngine',
    'PipelinePlan',
    'ExecutionMode',
    'StageConfig',
    'FailureMode',
    'PlanValidator',
    'PlanValidationReport',
    'ValidationOutcome',
    'ValidationReason',
    'ValidationIssue'
]
