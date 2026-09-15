"""Small seams through which adapters bind the reusable conformance suites.

Fixtures must report facts observed at a real SDK, process, service, model, storage,
or effect boundary. They must not derive their observations from the public task state
or directly mutate a task merely to satisfy an assertion.
"""

from collections.abc import Callable, Sequence
from collections.abc import Set as AbstractSet
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeAlias, TypeVar

from ..compatibility import CompatibilityStatus, UnconfirmedStart
from ..events import MessageRole
from ..harness import AgentHarness, AgentSession, AgentTask
from ..identity import ProviderId
from ..interaction import InteractionResponse
from ..outcome import AgentUsage, FailureKind, SchemaValidation, StopReason
from ..requirements import SessionSpec, TaskRequest
from ..support import Capability, SupportReport

T = TypeVar("T")
FixtureResource: TypeAlias = AbstractAsyncContextManager[T]


class RuntimeObservation(Protocol):
    @property
    def observed_contexts(self) -> Sequence[str]: ...

    @property
    def observed_text_values(self) -> Sequence[str]: ...

    def hold(self) -> None: ...

    def release(self) -> None: ...


class AcceptanceFixture(Protocol):
    @property
    def harness(self) -> AgentHarness: ...

    @property
    def spec(self) -> SessionSpec: ...

    @property
    def observation(self) -> RuntimeObservation: ...


class StartControl(Protocol):
    def accept(self) -> None: ...

    def reject_before_delivery(self, message: str) -> None: ...

    def lose_acceptance_acknowledgement(self, accepted_by_runtime: bool) -> None: ...

    def observed_submissions(self) -> int: ...

    def observed_accepted_starts(self) -> int: ...


class StartAcceptanceFixture(AcceptanceFixture, Protocol):
    @property
    def start(self) -> StartControl: ...

    @property
    def submitted_starts(self) -> Sequence[UnconfirmedStart]: ...


class ResponseControl(Protocol):
    def accept(self) -> None: ...

    def reject_before_delivery(self, message: str) -> None: ...

    def lose_acceptance_acknowledgement(self, accepted_by_runtime: bool) -> None: ...

    def observed_submissions(self) -> int: ...

    def observed_accepted_responses(self) -> Sequence[InteractionResponse]: ...


class ResponseAcceptanceFixture(AcceptanceFixture, Protocol):
    @property
    def response(self) -> ResponseControl: ...

    def effect_count(self) -> int: ...


class InteractionRaceFixture(ResponseAcceptanceFixture, Protocol):
    def hold_response(self) -> None: ...

    async def await_response_submission(self) -> None: ...

    def release_response(self) -> None: ...


class RepeatedApprovalFixture(ResponseAcceptanceFixture, Protocol):
    def prepare_next_effect(self) -> None: ...


class AccountingFixture(AcceptanceFixture, Protocol):
    @property
    def measurements(self) -> Sequence[AgentUsage]: ...

    @property
    def session_measurements(self) -> Sequence[AgentUsage | None]: ...

    @property
    def completed_message_roles(self) -> AbstractSet[MessageRole]: ...


class ContextFixture(StartAcceptanceFixture, Protocol):
    def recreate_harness(self) -> AgentHarness: ...


class ExecutionCase(StrEnum):
    READ_ONLY = "read_only"
    WORKSPACE_DENIED_NETWORK = "workspace_denied_network"
    WORKSPACE_ALLOWED_NETWORK = "workspace_allowed_network"


class ExecutionAttempt(StrEnum):
    WRITE_WORKSPACE = "write_workspace"
    WRITE_ADDITIONAL = "write_additional"
    WRITE_OUTSIDE = "write_outside"
    USE_NETWORK = "use_network"


class ExecutionConstraintFixture(AcceptanceFixture, Protocol):
    def prepare(self, attempt: ExecutionAttempt) -> TaskRequest: ...

    def written_targets(self) -> AbstractSet[str]: ...

    def network_requests(self) -> int: ...


class MessageKind(StrEnum):
    ANSWER = "answer"
    COMMENTARY = "commentary"
    EXPLANATION = "explanation"


class OutputObservation:
    complete: bool


@dataclass(frozen=True, slots=True)
class TextObservation(OutputObservation):
    text: str
    complete: bool = True


@dataclass(frozen=True, slots=True)
class StructuredObservation(OutputObservation):
    json: str
    complete: bool = True
    reported_validation: SchemaValidation = SchemaValidation.NOT_VALIDATED


class TaskLifecycleControl(Protocol):
    async def report_running(self) -> None: ...

    async def report_message_delta(
        self, message_key: str, text: str, role: MessageKind | None = None
    ) -> None: ...

    async def report_completion(
        self,
        output: OutputObservation | None = None,
        stop_reason: StopReason = StopReason.FINISHED,
    ) -> None: ...

    async def report_failure(self, message: str, kind: FailureKind | None = None) -> None: ...

    async def report_cancelled_termination(self) -> None: ...


class LifecycleFixture(Protocol):
    @property
    def harness(self) -> AgentHarness: ...

    def control(self, task: AgentTask) -> TaskLifecycleControl: ...


class OutputCase(StrEnum):
    PARTIAL = "partial"
    MISSING = "missing"
    EMPTY = "empty"


class OutcomeFixture(Protocol):
    @property
    def harness(self) -> AgentHarness: ...

    @property
    def spec(self) -> SessionSpec: ...

    @property
    def known_partial_usage(self) -> AgentUsage | None: ...

    def begin_model(self) -> None: ...

    def finish_model(self) -> None: ...

    def fail_model(self) -> None: ...

    async def lose_observation(self, session: AgentSession) -> None: ...


class PersistenceFailureFixture(AcceptanceFixture, Protocol):
    @property
    def supports_changed_instructions(self) -> bool: ...

    def recreate_harness(self) -> AgentHarness: ...

    def hide_stored_context(self) -> None: ...

    def restore_stored_context(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RequirementCase:
    id: str
    session_spec: SessionSpec
    request: TaskRequest
    session_validation: CompatibilityStatus
    task_validation: CompatibilityStatus
    create_decision: CompatibilityStatus | None = None
    start_decision: CompatibilityStatus | None = None
    capability: Capability | None = None

    def __post_init__(self) -> None:
        if not self.id or self.id.isspace():
            raise ValueError("requirement case id must not be blank")
        if self.create_decision is None:
            object.__setattr__(self, "create_decision", self.session_validation)
        if self.start_decision is None:
            object.__setattr__(self, "start_decision", self.task_validation)


@dataclass(frozen=True, slots=True)
class FixtureProfile:
    id: str
    description: str
    expected_support: SupportReport
    cases: tuple[RequirementCase, ...]

    def __init__(
        self,
        id: str,
        description: str,
        expected_support: SupportReport,
        cases: Sequence[RequirementCase],
    ) -> None:
        if not id or id.isspace():
            raise ValueError("profile id must not be blank")
        if not description or description.isspace():
            raise ValueError("profile description must not be blank")
        values = tuple(cases)
        if not values:
            raise ValueError("profile must declare requirement cases")
        if len({case.id for case in values}) != len(values):
            raise ValueError("profile requirement case ids must be unique")
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "expected_support", expected_support)
        object.__setattr__(self, "cases", values)


class ProfileFixture(Protocol):
    @property
    def provider(self) -> ProviderId: ...

    def profiles(self) -> Sequence[FixtureProfile]: ...

    def create_harness(self, profile_id: str) -> AgentHarness: ...


class RuntimeRequirementsFixture(ProfileFixture, Protocol):
    @property
    def observation(self) -> RuntimeObservation: ...


class WorkspaceFixture(AcceptanceFixture, Protocol):
    @property
    def working_directory(self) -> str: ...

    @property
    def active_skill_name(self) -> str: ...

    @property
    def active_skill_path(self) -> str: ...

    @property
    def active_skill_body(self) -> str: ...

    @property
    def inactive_skill_name(self) -> str: ...

    @property
    def inactive_skill_path(self) -> str: ...

    @property
    def inactive_skill_body(self) -> str: ...

    @property
    def invalid_spec(self) -> SessionSpec: ...


FixtureFactory: TypeAlias = Callable[[], FixtureResource[T]]
