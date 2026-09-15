"""Optional provider diagnostics, isolated from semantic task events."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from .identity import ProviderId, TaskId


class DiagnosticEvent:
    task_id: TaskId


@dataclass(frozen=True, slots=True)
class ProviderDiagnostic(DiagnosticEvent):
    task_id: TaskId
    provider: ProviderId
    name: str
    payload: str


@dataclass(frozen=True, slots=True)
class DiagnosticGap(DiagnosticEvent):
    task_id: TaskId
    dropped_records: int

    def __post_init__(self) -> None:
        if self.dropped_records <= 0:
            raise ValueError("dropped diagnostic count must be positive")


class TaskDiagnostics(ABC):
    """Optional, independently buffered diagnostics for one task."""

    @abstractmethod
    def diagnostics(self) -> AsyncIterator[DiagnosticEvent]:
        """Open a new diagnostic subscription that terminates with the task."""
        raise NotImplementedError
