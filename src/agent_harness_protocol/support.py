"""Capability discovery and cleanup bounds."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from types import MappingProxyType


class Capability(StrEnum):
    CALLER_APPROVAL = "caller_approval"
    QUESTIONS = "questions"
    PERSISTENCE = "persistence"
    WORKSPACE = "workspace"
    EXECUTION_CONSTRAINT = "execution_constraint"
    STRUCTURED_OUTPUT = "structured_output"
    DIAGNOSTICS = "diagnostics"
    CONTEXT_RETENTION = "context_retention"
    USER_HISTORY_VISIBILITY = "user_history_visibility"


class SupportScope(StrEnum):
    PROVIDER = "provider"
    HARNESS = "harness"
    SESSION = "session"


class Support:
    """Base class for capability support information."""


@dataclass(frozen=True, slots=True)
class Supported(Support):
    """Supported by this harness configuration; admission still decides each request."""


@dataclass(frozen=True, slots=True)
class ConditionalSupport(Support):
    scope: SupportScope
    condition: str

    def __post_init__(self) -> None:
        if not self.condition or self.condition.isspace():
            raise ValueError("conditional support must explain its condition")


@dataclass(frozen=True, slots=True)
class Unsupported(Support):
    reason: str

    def __post_init__(self) -> None:
        if not self.reason or self.reason.isspace():
            raise ValueError("unsupported capability must include a reason")


@dataclass(frozen=True, slots=True)
class UnknownSupport(Support):
    reason: str

    def __post_init__(self) -> None:
        if not self.reason or self.reason.isspace():
            raise ValueError("unknown capability support must include a reason")


SUPPORTED = Supported()


@dataclass(frozen=True, slots=True)
class SupportReport:
    """Discovery information, not a promise that a later request will be admitted."""

    entries: Mapping[Capability, Support]

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))

    def __getitem__(self, capability: Capability) -> Support:
        return self.entries.get(
            capability,
            UnknownSupport("adapter provides no information about this capability"),
        )


@dataclass(frozen=True, slots=True)
class CleanupBudget:
    """Published upper bounds for one task and one whole cleanup operation."""

    per_task: timedelta
    total: timedelta
    aggregates_across_resources: bool

    def __post_init__(self) -> None:
        if self.per_task < timedelta(0) or self.total < timedelta(0):
            raise ValueError("cleanup budgets must not be negative")
