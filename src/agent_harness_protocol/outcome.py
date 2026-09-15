"""Terminal outcomes, delivered output, and honest usage accounting."""

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from .identity import TaskId


class SchemaValidation(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    NOT_VALIDATED = "not_validated"


class TaskOutput:
    complete: bool


@dataclass(frozen=True, slots=True)
class TextOutput(TaskOutput):
    text: str
    complete: bool = True


@dataclass(frozen=True, slots=True)
class StructuredOutput(TaskOutput):
    json: str
    validation: SchemaValidation
    complete: bool = True


@dataclass(frozen=True, slots=True)
class AgentUsage:
    """Cumulative usage; ``None`` means unknown while zero means measured zero."""

    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    cache_write_input_tokens: int | None = None

    UNKNOWN: ClassVar["AgentUsage"]
    ZERO: ClassVar["AgentUsage"]

    def __add__(self, other: "AgentUsage") -> "AgentUsage":
        def add(left: int | None, right: int | None) -> int | None:
            return None if left is None or right is None else left + right

        return AgentUsage(
            input_tokens=add(self.input_tokens, other.input_tokens),
            cached_input_tokens=add(self.cached_input_tokens, other.cached_input_tokens),
            output_tokens=add(self.output_tokens, other.output_tokens),
            reasoning_tokens=add(self.reasoning_tokens, other.reasoning_tokens),
            total_tokens=add(self.total_tokens, other.total_tokens),
            cache_write_input_tokens=add(
                self.cache_write_input_tokens, other.cache_write_input_tokens
            ),
        )

    def __sub__(self, other: "AgentUsage") -> "AgentUsage":
        def subtract(current: int | None, baseline: int | None) -> int | None:
            if current is None or baseline is None or current < baseline:
                return None
            return current - baseline

        return AgentUsage(
            input_tokens=subtract(self.input_tokens, other.input_tokens),
            cached_input_tokens=subtract(self.cached_input_tokens, other.cached_input_tokens),
            output_tokens=subtract(self.output_tokens, other.output_tokens),
            reasoning_tokens=subtract(self.reasoning_tokens, other.reasoning_tokens),
            total_tokens=subtract(self.total_tokens, other.total_tokens),
            cache_write_input_tokens=subtract(
                self.cache_write_input_tokens, other.cache_write_input_tokens
            ),
        )


AgentUsage.UNKNOWN = AgentUsage()
AgentUsage.ZERO = AgentUsage(0, 0, 0, 0, 0, 0)


class UnresolvedReason(StrEnum):
    OBSERVATION_LOST = "observation_lost"
    CLEANUP_BOUND_EXCEEDED = "cleanup_bound_exceeded"
    CANCELLATION_UNCONFIRMED = "cancellation_unconfirmed"
    PARTIAL_EVIDENCE = "partial_evidence"


class StopReason(StrEnum):
    FINISHED = "finished"
    ITERATION_LIMIT = "iteration_limit"
    LOOP_DETECTED = "loop_detected"
    PROVIDER_STOPPED = "provider_stopped"


class FailureKind(StrEnum):
    TRANSIENT = "transient"
    AUTHENTICATION = "authentication"
    POLICY_BLOCKED = "policy_blocked"
    CONTEXT_OVERFLOW = "context_overflow"
    BUDGET_EXCEEDED = "budget_exceeded"
    PROVIDER = "provider"
    TRANSPORT = "transport"
    UNKNOWN = "unknown"


class TaskOutcome:
    task_id: TaskId
    output: TaskOutput | None
    usage: AgentUsage
    session_usage: AgentUsage | None


@dataclass(frozen=True, slots=True)
class Completed(TaskOutcome):
    task_id: TaskId
    stop_reason: StopReason
    output: TaskOutput | None = None
    usage: AgentUsage = AgentUsage.UNKNOWN
    session_usage: AgentUsage | None = None


@dataclass(frozen=True, slots=True)
class Failed(TaskOutcome):
    task_id: TaskId
    kind: FailureKind
    message: str
    cause: BaseException | None = None
    output: TaskOutput | None = None
    usage: AgentUsage = AgentUsage.UNKNOWN
    session_usage: AgentUsage | None = None


@dataclass(frozen=True, slots=True)
class Cancelled(TaskOutcome):
    task_id: TaskId
    output: TaskOutput | None = None
    usage: AgentUsage = AgentUsage.UNKNOWN
    session_usage: AgentUsage | None = None


@dataclass(frozen=True, slots=True)
class Unresolved(TaskOutcome):
    task_id: TaskId
    reason: UnresolvedReason
    known: str
    output: TaskOutput | None = None
    usage: AgentUsage = AgentUsage.UNKNOWN
    session_usage: AgentUsage | None = None
