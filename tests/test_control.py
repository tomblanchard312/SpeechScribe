"""
Minimal tests for the control plane (profiles, environment, registry).

Run from repo root: pytest tests/
"""

import pytest


def test_environment_enum():
    """Environment enum has expected values."""
    from speechscribe.control import Environment

    assert Environment.AZURE.value == "azure"
    assert Environment.OFFLINE.value == "offline"
    assert Environment.LOCAL.value == "local"


def test_profile_registry_list_profiles():
    """ProfileRegistry lists at least one profile."""
    from speechscribe.control import ProfileRegistry

    registry = ProfileRegistry()
    profiles = registry.list_profiles()
    assert isinstance(profiles, list)
    assert len(profiles) >= 1
    assert (
        "enterprise_meeting_post" in profiles or "sovereign_offline_archive" in profiles
    )


def test_recommendation_engine_recommend():
    """RecommendationEngine.recommend_configuration returns dict or raises ValueError."""
    from speechscribe.control import RecommendationEngine

    engine = RecommendationEngine()
    # In local environment some profiles may have no suitable engine
    try:
        config = engine.recommend_configuration("enterprise_meeting_post")
        assert "engine" in config
        assert "environment" in config
    except ValueError as e:
        assert "Unknown profile" in str(e) or "No suitable engine" in str(e)
