"""
Control Plane - Plan Validation

Validates PipelinePlan objects against governance policies.
"""

import logging
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

from .pipeline_plan import PipelinePlan, StageConfig
from .environment import Environment

logger = logging.getLogger(__name__)


class ValidationOutcome(Enum):
    """Plan validation outcomes."""
    VALID = "valid"
    VALID_WITH_WARNINGS = "valid_with_warnings"
    INVALID = "invalid"


class ValidationReason(Enum):
    """Reasons for validation outcomes."""
    FORBIDDEN_ENGINE = "forbidden_engine"
    LATENCY_CLASS_MISMATCH = "latency_class_mismatch"
    REQUIRED_STAGE_UNAVAILABLE = "required_stage_unavailable"
    POLICY_VIOLATION = "policy_violation"
    ENVIRONMENT_CONSTRAINT_VIOLATION = "environment_constraint_violation"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"
    STAGE_DEPENDENCY_MISSING = "stage_dependency_missing"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    reason: ValidationReason
    severity: str  # "error" or "warning"
    stage_name: Optional[str]
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class PlanValidationReport:
    """
    Structured validation report for PipelinePlan objects.

    Answers: "Is this plan acceptable under current policy?"

    This is governance, not execution logic.
    """

    plan: PipelinePlan
    outcome: ValidationOutcome
    issues: List[ValidationIssue]
    summary: str

    @property
    def is_valid(self) -> bool:
        """Check if the plan is valid (no errors)."""
        return self.outcome != ValidationOutcome.INVALID

    @property
    def has_warnings(self) -> bool:
        """Check if the plan has warnings."""
        return any(issue.severity == "warning" for issue in self.issues)

    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return len([issue for issue in self.issues if issue.severity == "error"])

    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return len([issue for issue in self.issues if issue.severity == "warning"])

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'plan': self.plan.to_dict(),
            'outcome': self.outcome.value,
            'issues': [
                {
                    'reason': issue.reason.value,
                    'severity': issue.severity,
                    'stage_name': issue.stage_name,
                    'message': issue.message,
                    'details': issue.details
                }
                for issue in self.issues
            ],
            'summary': self.summary,
            'is_valid': self.is_valid,
            'has_warnings': self.has_warnings,
            'error_count': self.error_count,
            'warning_count': self.warning_count
        }


class PlanValidator:
    """
    Validates PipelinePlan objects against governance policies.

    This is governance logic, not execution logic.
    """

    def __init__(self):
        # In a real implementation, these would be loaded from config
        self.forbidden_engines = {
            Environment.AZURE: [],  # No forbidden engines in Azure
            Environment.OFFLINE: ['azure_speech'],  # No cloud engines offline
            Environment.LOCAL: ['azure_speech']  # No cloud engines locally
        }

        self.resource_limits = {
            'max_memory_mb': 8192,  # 8GB limit
            'max_stages': 10  # Max pipeline stages
        }

    def validate_plan(self, plan: PipelinePlan) -> PlanValidationReport:
        """
        Validate a pipeline plan against governance policies.

        Args:
            plan: PipelinePlan to validate

        Returns:
            PlanValidationReport with validation results
        """
        issues = []

        # Check basic plan structure
        issues.extend(self._validate_plan_structure(plan))

        # Check environment constraints
        issues.extend(self._validate_environment_constraints(plan))

        # Check engine availability and policies
        issues.extend(self._validate_engine_policies(plan))

        # Check resource limits
        issues.extend(self._validate_resource_limits(plan))

        # Check stage dependencies
        issues.extend(self._validate_stage_dependencies(plan))

        # Determine outcome
        outcome = self._determine_outcome(issues)

        # Generate summary
        summary = self._generate_summary(outcome, issues)

        return PlanValidationReport(
            plan=plan,
            outcome=outcome,
            issues=issues,
            summary=summary
        )

    def _validate_plan_structure(self, plan: PipelinePlan) -> List[ValidationIssue]:
        """Validate basic plan structure."""
        issues = []

        # Must have at least ASR stage
        asr_stages = [s for s in plan.stages if s.stage_name ==
                      'asr' and s.enabled]
        if not asr_stages:
            issues.append(ValidationIssue(
                reason=ValidationReason.REQUIRED_STAGE_UNAVAILABLE,
                severity="error",
                stage_name="asr",
                message="ASR stage is required but not present in plan"
            ))

        # Check for duplicate stages
        stage_names = [s.stage_name for s in plan.stages if s.enabled]
        if len(stage_names) != len(set(stage_names)):
            issues.append(ValidationIssue(
                reason=ValidationReason.POLICY_VIOLATION,
                severity="error",
                stage_name=None,
                message="Duplicate stages found in pipeline plan"
            ))

        return issues

    def _validate_environment_constraints(self, plan: PipelinePlan) -> List[ValidationIssue]:
        """Validate environment constraints."""
        issues = []

        # Check profile environment constraints
        if plan.environment not in plan.profile.environment_constraints:
            issues.append(ValidationIssue(
                reason=ValidationReason.ENVIRONMENT_CONSTRAINT_VIOLATION,
                severity="error",
                stage_name=None,
                message=(f"Profile {plan.profile.name} not allowed in "
                         f"{plan.environment.value} environment")
            ))

        return issues

    def _validate_engine_policies(self, plan: PipelinePlan) -> List[ValidationIssue]:
        """Validate engine policies and availability."""
        issues = []

        forbidden_engines = self.forbidden_engines.get(plan.environment, [])

        for stage_config in plan.stages:
            if not stage_config.enabled:
                continue

            # Check for forbidden engines
            if stage_config.engine_name in forbidden_engines:
                issues.append(ValidationIssue(
                    reason=ValidationReason.FORBIDDEN_ENGINE,
                    severity="error",
                    stage_name=stage_config.stage_name,
                    message=(f"Engine '{stage_config.engine_name}' is forbidden in "
                             f"{plan.environment.value} environment")
                ))

            # Check latency compatibility (warning only)
            if self._check_latency_mismatch(plan, stage_config):
                issues.append(ValidationIssue(
                    reason=ValidationReason.LATENCY_CLASS_MISMATCH,
                    severity="warning",
                    stage_name=stage_config.stage_name,
                    message=(f"Engine '{stage_config.engine_name}' may not meet "
                             f"latency requirements for profile")
                ))

        return issues

    def _validate_resource_limits(self, plan: PipelinePlan) -> List[ValidationIssue]:
        """Validate resource limits."""
        issues = []

        # Check stage count
        enabled_stages = len([s for s in plan.stages if s.enabled])
        if enabled_stages > self.resource_limits['max_stages']:
            issues.append(ValidationIssue(
                reason=ValidationReason.RESOURCE_LIMIT_EXCEEDED,
                severity="error",
                stage_name=None,
                message=(f"Too many stages ({enabled_stages}) exceeds limit "
                         f"({self.resource_limits['max_stages']})")
            ))

        return issues

    def _validate_stage_dependencies(self, plan: PipelinePlan) -> List[ValidationIssue]:
        """Validate stage dependencies."""
        issues = []

        enabled_stages = {s.stage_name for s in plan.stages if s.enabled}

        # Translation requires ASR
        if 'translation' in enabled_stages and 'asr' not in enabled_stages:
            issues.append(ValidationIssue(
                reason=ValidationReason.STAGE_DEPENDENCY_MISSING,
                severity="error",
                stage_name="translation",
                message="Translation stage requires ASR stage to be enabled"
            ))

        # Diarization typically requires ASR
        if 'diarization' in enabled_stages and 'asr' not in enabled_stages:
            issues.append(ValidationIssue(
                reason=ValidationReason.STAGE_DEPENDENCY_MISSING,
                severity="warning",
                stage_name="diarization",
                message="Diarization stage typically requires ASR stage"
            ))

        return issues

    def _check_latency_mismatch(self, plan: PipelinePlan, stage_config: 'StageConfig') -> bool:
        """Check if engine latency might not match profile requirements."""
        # This is a simplified check - in reality would check engine capabilities
        if plan.profile.latency_requirement.value == 'realtime':
            # Real-time profiles shouldn't use batch engines
            if 'batch' in stage_config.engine_name.lower():
                return True

        return False

    def _determine_outcome(self, issues: List[ValidationIssue]) -> ValidationOutcome:
        """Determine validation outcome from issues."""
        error_count = len([i for i in issues if i.severity == "error"])
        warning_count = len([i for i in issues if i.severity == "warning"])

        if error_count > 0:
            return ValidationOutcome.INVALID
        elif warning_count > 0:
            return ValidationOutcome.VALID_WITH_WARNINGS
        else:
            return ValidationOutcome.VALID

    def _generate_summary(self, outcome: ValidationOutcome, issues: List[ValidationIssue]) -> str:
        """Generate human-readable summary."""
        error_count = len([i for i in issues if i.severity == "error"])
        warning_count = len([i for i in issues if i.severity == "warning"])

        if outcome == ValidationOutcome.INVALID:
            return f"Plan is invalid with {error_count} error(s) and {warning_count} warning(s)"
        elif outcome == ValidationOutcome.VALID_WITH_WARNINGS:
            return f"Plan is valid but has {warning_count} warning(s)"
        else:
            return "Plan is valid"
