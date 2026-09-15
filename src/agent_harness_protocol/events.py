"""Provider-independent semantic task events."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

from .identity import InteractionId, MessageId, TaskId, WorkId
from .interaction import EffectKind, InteractionRequest, InteractionResolution
from .outcome import AgentUsage, Cancelled, Completed, Failed, TaskOutcome, Unresolved


class MessageRole(StrEnum):
    ANSWER = "answer"
    COMMENTARY = "commentary"
    EXPLANATION = "explanation"
    UNKNOWN = "unknown"


class WorkStatus(StrEnum):
    STARTED = "started"
    UPDATED = "updated"
    COMPLETED = "completed"
    FAILED = "failed"
    DECLINED = "declined"
    CANCELLED = "cancelled"


class WarningKind(StrEnum):
    CONTEXT_PRESSURE = "context_pressure"
    CONFIGURATION = "configuration"
    RECOVERABLE = "recoverable"
    OTHER = "other"


class TaskState(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    AWAITING_RESPONSE = "awaiting_response"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNRESOLVED = "unresolved"

    @property
    def is_terminal(self) -> bool:
        return self in {
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELLED,
            TaskState.UNRESOLVED,
        }


class TaskEvent:
    task_id: TaskId


@dataclass(frozen=True, slots=True)
class TaskStarted(TaskEvent):
    task_id: TaskId


@dataclass(frozen=True, slots=True)
class MessageDelta(TaskEvent):
    task_id: TaskId
    message_id: MessageId
    text: str
    role: MessageRole = MessageRole.UNKNOWN


@dataclass(frozen=True, slots=True)
class MessageCompleted(TaskEvent):
    task_id: TaskId
    message_id: MessageId
    text: str
    role: MessageRole = MessageRole.UNKNOWN


@dataclass(frozen=True, slots=True)
class ToolCallChanged(TaskEvent):
    task_id: TaskId
    work_id: WorkId
    name: str
    status: WorkStatus
    arguments: str | None = None
    result: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EffectChanged(TaskEvent):
    task_id: TaskId
    work_id: WorkId
    kind: EffectKind
    status: WorkStatus
    description: str | None = None
    output: str | None = None
    exit_code: int | None = None
    changed_paths: tuple[str, ...] = ()

    def __init__(
        self,
        task_id: TaskId,
        work_id: WorkId,
        kind: EffectKind,
        status: WorkStatus,
        description: str | None = None,
        output: str | None = None,
        exit_code: int | None = None,
        changed_paths: Iterable[str] = (),
    ) -> None:
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "work_id", work_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "exit_code", exit_code)
        object.__setattr__(self, "changed_paths", tuple(changed_paths))


@dataclass(frozen=True, slots=True)
class InteractionRequested(TaskEvent):
    task_id: TaskId
    request: InteractionRequest


@dataclass(frozen=True, slots=True)
class InteractionResolved(TaskEvent):
    task_id: TaskId
    interaction_id: InteractionId
    resolution: InteractionResolution


@dataclass(frozen=True, slots=True)
class UsageChanged(TaskEvent):
    task_id: TaskId
    task: AgentUsage
    session: AgentUsage | None = None


@dataclass(frozen=True, slots=True)
class Warning(TaskEvent):
    task_id: TaskId
    kind: WarningKind
    message: str


@dataclass(frozen=True, slots=True)
class ObservationGap(TaskEvent):
    task_id: TaskId
    dropped_events: int

    def __post_init__(self) -> None:
        if self.dropped_events <= 0:
            raise ValueError("dropped event count must be positive")


TerminalOutcome = TypeVar("TerminalOutcome", bound=TaskOutcome)


class TerminalEvent(TaskEvent, Generic[TerminalOutcome]):
    outcome: TerminalOutcome


@dataclass(frozen=True, slots=True)
class TaskCompleted(TerminalEvent[Completed]):
    task_id: TaskId
    outcome: Completed


@dataclass(frozen=True, slots=True)
class TaskFailed(TerminalEvent[Failed]):
    task_id: TaskId
    outcome: Failed


@dataclass(frozen=True, slots=True)
class TaskCancelled(TerminalEvent[Cancelled]):
    task_id: TaskId
    outcome: Cancelled


@dataclass(frozen=True, slots=True)
class TaskUnresolved(TerminalEvent[Unresolved]):
    task_id: TaskId
    outcome: Unresolved
