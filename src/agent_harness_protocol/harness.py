"""Abstract ports implemented by concrete agent-harness adapters."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType
from typing import Self

from .compatibility import CompatibilityReport
from .events import TaskEvent, TaskState
from .identity import InteractionId, PersistentSessionRef, ProviderId, SessionId, TaskId
from .interaction import InteractionRequest, InteractionResponse
from .outcome import TaskOutcome
from .requirements import SessionSpec, TaskRequest
from .support import CleanupBudget, SupportReport


class ContextRetentionDisposition(StrEnum):
    EPHEMERAL = "ephemeral"
    MATERIALIZED = "materialized"
    UNKNOWN = "unknown"


class UserHistoryVisibility(StrEnum):
    VISIBLE = "visible"
    HIDDEN = "hidden"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SessionDisposition:
    retention: ContextRetentionDisposition = ContextRetentionDisposition.UNKNOWN
    history_visibility: UserHistoryVisibility = UserHistoryVisibility.UNKNOWN


class AgentHarness(ABC):
    """Business-facing port for delegating work to an agent harness."""

    @property
    @abstractmethod
    def provider(self) -> ProviderId:
        raise NotImplementedError

    @property
    @abstractmethod
    def support(self) -> SupportReport:
        raise NotImplementedError

    @property
    @abstractmethod
    def cleanup_budget(self) -> CleanupBudget:
        raise NotImplementedError

    @abstractmethod
    def validate(self, spec: SessionSpec) -> CompatibilityReport:
        """Diagnose requirements without starting native work."""
        raise NotImplementedError

    @abstractmethod
    async def create_session(self, spec: SessionSpec) -> "AgentSession":
        """Create a new logical context without waiting for task completion."""
        raise NotImplementedError

    @abstractmethod
    async def aclose(self) -> None:
        """Settle owned handles and resources within ``cleanup_budget.total``."""
        raise NotImplementedError

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


class AgentSession(ABC):
    """A logical context shared by sequential tasks."""

    @property
    @abstractmethod
    def id(self) -> SessionId:
        raise NotImplementedError

    @property
    @abstractmethod
    def spec(self) -> SessionSpec:
        raise NotImplementedError

    @property
    def disposition(self) -> SessionDisposition:
        return SessionDisposition()

    @property
    @abstractmethod
    def persistent_ref(self) -> PersistentSessionRef | None:
        raise NotImplementedError

    @abstractmethod
    def validate(self, request: TaskRequest) -> CompatibilityReport:
        raise NotImplementedError

    @abstractmethod
    async def start_task(self, request: TaskRequest) -> "AgentTask":
        """Submit a task and return after admission, before task completion."""
        raise NotImplementedError

    @abstractmethod
    async def release(self) -> None:
        """Idempotently release this handle within the harness cleanup budget."""
        raise NotImplementedError

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.release()


class PersistentSessions(ABC):
    """Optional contract for reopening persistent context."""

    @abstractmethod
    async def reopen_session(
        self, reference: PersistentSessionRef, spec: SessionSpec
    ) -> AgentSession:
        raise NotImplementedError


class AgentTask(ABC):
    """Observation, intervention, and terminal-result handle for one delegated task."""

    @property
    @abstractmethod
    def id(self) -> TaskId:
        raise NotImplementedError

    @property
    @abstractmethod
    def session_id(self) -> SessionId:
        raise NotImplementedError

    @property
    @abstractmethod
    def state(self) -> TaskState:
        """Current snapshot, independent of event subscribers."""
        raise NotImplementedError

    @abstractmethod
    def events(self) -> AsyncIterator[TaskEvent]:
        """Open a new bounded semantic-event subscription.

        Slow subscribers receive :class:`ObservationGap`; the single terminal event is
        retained and delivered last. Late subscribers need not receive historical events.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def pending_interactions(self) -> tuple[InteractionRequest, ...]:
        """Current response-required snapshot, independent of event delivery."""
        raise NotImplementedError

    @abstractmethod
    async def respond(self, interaction_id: InteractionId, response: InteractionResponse) -> None:
        raise NotImplementedError

    @abstractmethod
    async def request_cancellation(self) -> None:
        """Request interruption; returning does not itself prove termination."""
        raise NotImplementedError

    @abstractmethod
    async def await_outcome(self) -> TaskOutcome:
        """Return the same terminal judgment to every waiter.

        Cancelling this awaitable must not cancel the underlying agent task.
        """
        raise NotImplementedError
