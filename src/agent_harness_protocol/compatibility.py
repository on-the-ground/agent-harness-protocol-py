"""Compatibility diagnostics and boundary errors."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from .identity import InteractionId, SessionId, TaskId


class CompatibilityStatus(StrEnum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNCONFIRMED = "unconfirmed"


class CompatibilityIssueKind(StrEnum):
    ADVISORY = "advisory"
    UNSUPPORTED = "unsupported"
    UNCONFIRMED = "unconfirmed"


@dataclass(frozen=True, slots=True)
class CompatibilityIssue:
    path: str
    message: str
    kind: CompatibilityIssueKind = CompatibilityIssueKind.UNSUPPORTED


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """Whether required meaning can be preserved without weakening the request."""

    issues: tuple[CompatibilityIssue, ...] = ()

    def __init__(self, issues: Iterable[CompatibilityIssue] = ()) -> None:
        object.__setattr__(self, "issues", tuple(issues))

    @property
    def status(self) -> CompatibilityStatus:
        if any(issue.kind is CompatibilityIssueKind.UNSUPPORTED for issue in self.issues):
            return CompatibilityStatus.INCOMPATIBLE
        if any(issue.kind is CompatibilityIssueKind.UNCONFIRMED for issue in self.issues):
            return CompatibilityStatus.UNCONFIRMED
        return CompatibilityStatus.COMPATIBLE

    @property
    def is_compatible(self) -> bool:
        return self.status is CompatibilityStatus.COMPATIBLE

    def require_compatible(self) -> None:
        if self.status is CompatibilityStatus.INCOMPATIBLE:
            raise IncompatibleRequirementError(self.issues)
        if self.status is CompatibilityStatus.UNCONFIRMED:
            raise RequirementUnconfirmedError(self.issues)


COMPATIBLE = CompatibilityReport()


class RequirementUnconfirmedError(RuntimeError):
    def __init__(self, issues: Iterable[CompatibilityIssue]) -> None:
        self.issues = tuple(issues)
        detail = "; ".join(f"{issue.path}: {issue.message}" for issue in self.issues)
        super().__init__(f"requirement compatibility could not be confirmed: {detail}")


class IncompatibleRequirementError(RuntimeError):
    def __init__(self, issues: Iterable[CompatibilityIssue]) -> None:
        self.issues = tuple(issues)
        detail = "; ".join(f"{issue.path}: {issue.message}" for issue in self.issues)
        super().__init__(f"requirements cannot be preserved: {detail}")


class HarnessTransportError(RuntimeError):
    """The harness boundary failed; this alone does not prove that the task failed."""


@dataclass(frozen=True, slots=True)
class UnconfirmedStart:
    session_id: SessionId
    request_id: str

    def __post_init__(self) -> None:
        if not self.request_id or self.request_id.isspace():
            raise ValueError("request id must not be blank")


@dataclass(frozen=True, slots=True)
class UnconfirmedResponse:
    task_id: TaskId
    interaction_id: InteractionId


class TaskStartUnconfirmedError(RuntimeError):
    def __init__(self, reference: UnconfirmedStart, message: str) -> None:
        self.reference = reference
        super().__init__(message)


class InteractionResponseUnconfirmedError(RuntimeError):
    def __init__(self, reference: UnconfirmedResponse, message: str) -> None:
        self.reference = reference
        super().__init__(message)


class SessionBlockedError(RuntimeError):
    def __init__(self, session_id: SessionId, message: str) -> None:
        self.session_id = session_id
        super().__init__(message)
