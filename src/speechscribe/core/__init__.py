"""Compatibility namespace for pre-consolidation imports.

Implementation lives in the canonical ``speechscribe`` package modules.
"""

from ..control import EngineRegistry, ProfileRegistry, RecommendationEngine
from ..planning import PipelinePlan, PlanValidator

__all__ = [
    "EngineRegistry",
    "ProfileRegistry",
    "RecommendationEngine",
    "PipelinePlan",
    "PlanValidator",
]
