"""Pipeline planning and governance for SpeechScribe."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .control import Environment, LatencyRequirement, Profile


class ExecutionMode(Enum):
    STREAMING = "streaming"
    BATCH = "batch"


class FailureMode(Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    BEST_EFFORT = "best_effort"
    DEGRADES = "degrades"


@dataclass
class StageConfig:
    stage_name: str
    engine_name: str
    enabled: bool = True
    config: Optional[Dict[str, Any]] = None
    failure_mode: FailureMode = FailureMode.REQUIRED


@dataclass
class PipelinePlan:
    """Serializable execution contract derived from a profile."""

    profile: Profile
    environment: Environment
    execution_mode: ExecutionMode
    stages: List[StageConfig]

    def __post_init__(self) -> None:
        self._set_default_failure_modes()
        self._validate_plan()

    def _set_default_failure_modes(self) -> None:
        for stage in self.stages:
            if stage.failure_mode != FailureMode.REQUIRED:
                continue
            if stage.stage_name == "diarization":
                stage.failure_mode = FailureMode.OPTIONAL
            elif stage.stage_name == "translation":
                stage.failure_mode = FailureMode.BEST_EFFORT
            elif stage.stage_name == "tts":
                stage.failure_mode = FailureMode.DEGRADES

    def _validate_plan(self) -> None:
        asr_stage = self.get_stage_config("asr")
        if not asr_stage or not asr_stage.enabled:
            raise ValueError("ASR stage must be enabled in pipeline plan")

    def get_enabled_stages(self) -> List[str]:
        return [stage.stage_name for stage in self.stages if stage.enabled]

    def get_stage_config(self, stage_name: str) -> Optional[StageConfig]:
        return next((s for s in self.stages if s.stage_name == stage_name), None)

    def get_engine_for_stage(self, stage_name: str) -> Optional[str]:
        stage = self.get_stage_config(stage_name)
        return stage.engine_name if stage else None

    def is_stage_enabled(self, stage_name: str) -> bool:
        stage = self.get_stage_config(stage_name)
        return bool(stage and stage.enabled)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "environment": self.environment.value,
            "execution_mode": self.execution_mode.value,
            "stages": [
                {
                    "stage_name": s.stage_name,
                    "engine_name": s.engine_name,
                    "enabled": s.enabled,
                    "config": s.config,
                    "failure_mode": s.failure_mode.value,
                }
                for s in self.stages
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelinePlan":
        p = data["profile"]
        profile = Profile(
            name=p["name"],
            description=p["description"],
            latency_requirement=LatencyRequirement(p["latency_requirement"]),
            streaming_required=p.get("streaming_required", False),
            batch_required=p.get("batch_required", False),
            diarization_required=p.get("diarization_required", False),
            translation_required=p.get("translation_required", False),
            translation_languages=p.get("translation_languages", []),
            environment_constraints={
                Environment(value) for value in p.get("environment_constraints", [])
            },
            tts_required=p.get("tts_required", False),
            summarization_required=p.get("summarization_required", False),
        )
        stages = [
            StageConfig(
                stage_name=s["stage_name"],
                engine_name=s["engine_name"],
                enabled=s.get("enabled", True),
                config=s.get("config"),
                failure_mode=FailureMode(s.get("failure_mode", "required")),
            )
            for s in data["stages"]
        ]
        return cls(
            profile=profile,
            environment=Environment(data["environment"]),
            execution_mode=ExecutionMode(data["execution_mode"]),
            stages=stages,
        )


class ValidationOutcome(Enum):
    VALID = "valid"
    VALID_WITH_WARNINGS = "valid_with_warnings"
    INVALID = "invalid"


class ValidationReason(Enum):
    FORBIDDEN_ENGINE = "forbidden_engine"
    LATENCY_CLASS_MISMATCH = "latency_class_mismatch"
    REQUIRED_STAGE_UNAVAILABLE = "required_stage_unavailable"
    POLICY_VIOLATION = "policy_violation"
    ENVIRONMENT_CONSTRAINT_VIOLATION = "environment_constraint_violation"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"
    STAGE_DEPENDENCY_MISSING = "stage_dependency_missing"


@dataclass
class ValidationIssue:
    reason: ValidationReason
    severity: str
    stage_name: Optional[str]
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class PlanValidationReport:
    plan: PipelinePlan
    outcome: ValidationOutcome
    issues: List[ValidationIssue]
    summary: str

    @property
    def is_valid(self) -> bool:
        return self.outcome != ValidationOutcome.INVALID

    @property
    def has_warnings(self) -> bool:
        return any(issue.severity == "warning" for issue in self.issues)

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)


class PlanValidator:
    """Validate execution plans against environment and stage policies."""

    def __init__(self) -> None:
        self.forbidden_engines = {
            Environment.AZURE: [],
            Environment.OFFLINE: ["azure_speech"],
            Environment.LOCAL: ["azure_speech"],
        }
        self.resource_limits = {"max_stages": 10}

    def validate_plan(self, plan: PipelinePlan) -> PlanValidationReport:
        issues: List[ValidationIssue] = []
        enabled = [s for s in plan.stages if s.enabled]
        names = [s.stage_name for s in enabled]

        if "asr" not in names:
            issues.append(
                ValidationIssue(
                    ValidationReason.REQUIRED_STAGE_UNAVAILABLE,
                    "error",
                    "asr",
                    "ASR stage is required but not present in plan",
                )
            )
        if len(names) != len(set(names)):
            issues.append(
                ValidationIssue(
                    ValidationReason.POLICY_VIOLATION,
                    "error",
                    None,
                    "Duplicate stages found in pipeline plan",
                )
            )
        if (
            plan.profile.environment_constraints
            and plan.environment not in plan.profile.environment_constraints
        ):
            issues.append(
                ValidationIssue(
                    ValidationReason.ENVIRONMENT_CONSTRAINT_VIOLATION,
                    "error",
                    None,
                    f"Profile {plan.profile.name} is not allowed in {plan.environment.value}",
                )
            )

        forbidden = self.forbidden_engines.get(plan.environment, [])
        for stage in enabled:
            if stage.engine_name in forbidden:
                issues.append(
                    ValidationIssue(
                        ValidationReason.FORBIDDEN_ENGINE,
                        "error",
                        stage.stage_name,
                        f"Engine '{stage.engine_name}' is forbidden in {plan.environment.value}",
                    )
                )

        if len(enabled) > self.resource_limits["max_stages"]:
            issues.append(
                ValidationIssue(
                    ValidationReason.RESOURCE_LIMIT_EXCEEDED,
                    "error",
                    None,
                    f"Too many enabled stages ({len(enabled)})",
                )
            )

        for dependent in ("translation", "diarization"):
            if dependent in names and "asr" not in names:
                issues.append(
                    ValidationIssue(
                        ValidationReason.STAGE_DEPENDENCY_MISSING,
                        "error" if dependent == "translation" else "warning",
                        dependent,
                        f"{dependent.title()} stage requires ASR",
                    )
                )

        errors = sum(i.severity == "error" for i in issues)
        warnings = sum(i.severity == "warning" for i in issues)
        if errors:
            outcome = ValidationOutcome.INVALID
            summary = f"Plan is invalid with {errors} error(s) and {warnings} warning(s)"
        elif warnings:
            outcome = ValidationOutcome.VALID_WITH_WARNINGS
            summary = f"Plan is valid but has {warnings} warning(s)"
        else:
            outcome = ValidationOutcome.VALID
            summary = "Plan is valid"

        return PlanValidationReport(plan, outcome, issues, summary)
