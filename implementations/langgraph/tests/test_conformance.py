"""Bind the shared AHP conformance suites to the real LangGraph harness.

Every suite drives the actual ``LangGraphHarness``, a compiled ``StateGraph``, and a
LangGraph checkpointer. Only the chat model is replaced by ``ControlledChatModel``, and
observations are read from what that model received.

The runtime suite requires confirmed cancellation. For the controlled model the model
coroutine *is* the whole native work, so coroutine termination does prove that work
stopped, and the fixture selects
``CancellationSemantics.COROUTINE_TERMINATION_CONFIRMS_WORK_STOPPED``. The default
``UNCONFIRMED`` semantics is exercised by the cleanup suite and the adapter tests.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

from agent_harness_protocol import (
    READ_ONLY,
    SUPPORTED,
    AgentHarness,
    ApprovalRequirement,
    Capability,
    CompatibilityStatus,
    ContextRetentionDisposition,
    ContextRetentionRequirement,
    DiagnosticsRequirement,
    ExecutionConstraint,
    NetworkAccess,
    PersistenceRequirement,
    ProviderId,
    QuestionRequirement,
    SessionRequirements,
    SessionSpec,
    StructuredOutputRequirement,
    Support,
    SupportReport,
    TaskRequest,
    TaskRequirements,
    TextInput,
    UnknownSupport,
    Unsupported,
    UserHistoryVisibility,
    UserHistoryVisibilityRequirement,
    WorkspaceRequirement,
)
from agent_harness_protocol.conformance import (
    AcceptanceFixture,
    CleanupBudgetConformanceTests,
    FixtureProfile,
    FixtureResource,
    RequirementCase,
    RequirementsConformanceTests,
    RuntimeObservation,
    RuntimeProfileConformanceTests,
    RuntimeRequirementsFixture,
)
from langgraph.checkpoint.memory import InMemorySaver
from model_boundary import ControlledChatModel, ModelBoundaryObservation

from agent_harness_protocol_langgraph import CancellationSemantics, LangGraphHarness
from agent_harness_protocol_langgraph.harness import PROVIDER

MODEL_ID = "ahp-controlled-model"
CONFIRMED = CancellationSemantics.COROUTINE_TERMINATION_CONFIRMS_WORK_STOPPED
OK = CompatibilityStatus.COMPATIBLE
NO = CompatibilityStatus.INCOMPATIBLE
UNSURE = CompatibilityStatus.UNCONFIRMED
NO_TASK_REQUIREMENTS = TaskRequirements()

# --------------------------------------------------------------------------------------
# Runtime


@asynccontextmanager
async def _boundary(
    ignore_cancellation: bool = False,
) -> AsyncGenerator[ModelBoundaryObservation, None]:
    observation = ModelBoundaryObservation(
        ControlledChatModel(ignore_cancellation=ignore_cancellation)
    )
    try:
        yield observation
    finally:
        await observation.settle()


def _runtime_harness(boundary: RuntimeObservation) -> AgentHarness:
    assert isinstance(boundary, ModelBoundaryObservation)
    return LangGraphHarness(boundary.model, model_id=MODEL_ID, cancellation_semantics=CONFIRMED)


class TestLangGraphRuntime(RuntimeProfileConformanceTests):
    """Runtime suite against the real graph."""

    def boundary(self) -> FixtureResource[RuntimeObservation]:
        """Provide a fresh model boundary."""
        return _boundary()

    def harness(self, boundary: RuntimeObservation) -> AgentHarness:
        """Build the harness under test."""
        return _runtime_harness(boundary)


# --------------------------------------------------------------------------------------
# Cleanup


@dataclass(frozen=True, slots=True)
class _AcceptanceFixture:
    harness: AgentHarness
    spec: SessionSpec
    observation: ModelBoundaryObservation


@asynccontextmanager
async def _cleanup_fixture(
    semantics: CancellationSemantics, ignore_cancellation: bool
) -> AsyncGenerator[AcceptanceFixture, None]:
    async with _boundary(ignore_cancellation) as observation:
        harness = LangGraphHarness(observation.model, cancellation_semantics=semantics)
        try:
            yield _AcceptanceFixture(harness, SessionSpec(instructions="AHP_CLEANUP"), observation)
        finally:
            observation.release()
            await harness.aclose()


class TestLangGraphCleanupConfirmedCancellation(CleanupBudgetConformanceTests):
    """Cleanup suite with confirmed coroutine cancellation."""

    def cleanup_fixture(self) -> FixtureResource[AcceptanceFixture]:
        """Provide the cleanup fixture for this configuration."""
        return _cleanup_fixture(CONFIRMED, ignore_cancellation=False)


class TestLangGraphCleanupUnconfirmedCancellation(CleanupBudgetConformanceTests):
    """Cleanup suite with the default unconfirmed cancellation."""

    def cleanup_fixture(self) -> FixtureResource[AcceptanceFixture]:
        """Provide the cleanup fixture for this configuration."""
        return _cleanup_fixture(CancellationSemantics.UNCONFIRMED, ignore_cancellation=False)


class TestLangGraphCleanupWithModelIgnoringCancellation(CleanupBudgetConformanceTests):
    """Cleanup suite with a model that keeps running after cancellation."""

    def cleanup_fixture(self) -> FixtureResource[AcceptanceFixture]:
        """Provide the cleanup fixture for this configuration."""
        return _cleanup_fixture(CONFIRMED, ignore_cancellation=True)


# --------------------------------------------------------------------------------------
# Requirements


def _spec(requirements: SessionRequirements) -> SessionSpec:
    return SessionSpec(requirements=requirements)


def _request(case_id: str, requirements: TaskRequirements = NO_TASK_REQUIREMENTS) -> TaskRequest:
    return TaskRequest(TextInput(f"requirement-case-{case_id}"), requirements)


def _case(
    case_id: str,
    session: CompatibilityStatus,
    spec: SessionSpec,
    capability: Capability | None = None,
    task: CompatibilityStatus = OK,
    requirements: TaskRequirements = NO_TASK_REQUIREMENTS,
) -> RequirementCase:
    return RequirementCase(
        case_id,
        spec,
        _request(case_id, requirements),
        session_validation=session,
        task_validation=task,
        capability=capability,
    )


def _common_cases() -> list[RequirementCase]:
    """Cases whose decision does not depend on the declared disposition."""
    structured = TaskRequirements(StructuredOutputRequirement('{"type":"object"}'))
    return [
        _case("default", OK, SessionSpec()),
        _case("instructions", OK, SessionSpec(instructions="AHP_REQUIREMENT_INSTRUCTION")),
        _case("matching-model", OK, SessionSpec(model=MODEL_ID)),
        _case("mismatched-model", NO, SessionSpec(model="another-model")),
        _case(
            "diagnostics",
            OK,
            _spec(SessionRequirements(diagnostics=DiagnosticsRequirement.REQUIRED)),
            Capability.DIAGNOSTICS,
        ),
        _case(
            "deny-all-approval",
            OK,
            _spec(SessionRequirements(approval=ApprovalRequirement.DENY_ALL)),
        ),
        _case(
            "caller-approval",
            NO,
            _spec(SessionRequirements(approval=ApprovalRequirement.CALLER_DECIDES)),
            Capability.CALLER_APPROVAL,
        ),
        _case(
            "agent-reviewed-approval",
            NO,
            _spec(SessionRequirements(approval=ApprovalRequirement.AGENT_REVIEWED)),
            Capability.CALLER_APPROVAL,
        ),
        _case(
            "caller-answers",
            NO,
            _spec(SessionRequirements(questions=QuestionRequirement.CALLER_ANSWERS)),
            Capability.QUESTIONS,
        ),
        _case(
            "persistence",
            NO,
            _spec(SessionRequirements(persistence=PersistenceRequirement())),
            Capability.PERSISTENCE,
        ),
        _case(
            "workspace",
            NO,
            _spec(SessionRequirements(workspace=WorkspaceRequirement(working_directory="."))),
            Capability.WORKSPACE,
        ),
        _case(
            "read-only-filesystem",
            NO,
            _spec(SessionRequirements(execution=ExecutionConstraint(filesystem=READ_ONLY))),
            Capability.EXECUTION_CONSTRAINT,
        ),
        _case(
            "denied-network",
            NO,
            _spec(SessionRequirements(execution=ExecutionConstraint(network=NetworkAccess.DENIED))),
            Capability.EXECUTION_CONSTRAINT,
        ),
        _case(
            "structured-output",
            OK,
            SessionSpec(),
            Capability.STRUCTURED_OUTPUT,
            task=NO,
            requirements=structured,
        ),
    ]


def _disposition_cases(
    retention: CompatibilityStatus, visibility: CompatibilityStatus
) -> list[RequirementCase]:
    return [
        _case(
            "ephemeral-retention",
            retention,
            _spec(SessionRequirements(retention=ContextRetentionRequirement.EPHEMERAL)),
            Capability.CONTEXT_RETENTION,
        ),
        _case(
            "hidden-history",
            visibility,
            _spec(SessionRequirements(history_visibility=UserHistoryVisibilityRequirement.HIDDEN)),
            Capability.USER_HISTORY_VISIBILITY,
        ),
    ]


def _expected_support(retention: Support, visibility: Support) -> SupportReport:
    return SupportReport(
        {
            Capability.CALLER_APPROVAL: Unsupported("no approval route"),
            Capability.QUESTIONS: Unsupported("no question route"),
            Capability.PERSISTENCE: Unsupported("no cross-harness persistence"),
            Capability.WORKSPACE: Unsupported("no workspace"),
            Capability.EXECUTION_CONSTRAINT: Unsupported("no enforceable policy"),
            Capability.STRUCTURED_OUTPUT: Unsupported("no structured validation"),
            Capability.DIAGNOSTICS: SUPPORTED,
            Capability.CONTEXT_RETENTION: retention,
            Capability.USER_HISTORY_VISIBILITY: visibility,
        }
    )


_UNKNOWN = UnknownSupport("not declared")
_UNSUPPORTED = Unsupported("declared otherwise")


@dataclass(frozen=True, slots=True)
class _Profile:
    declared: FixtureProfile
    build: Callable[[ControlledChatModel], LangGraphHarness]


def _profiles() -> Sequence[_Profile]:
    return (
        _Profile(
            FixtureProfile(
                "adapter-owned-checkpointer",
                "default in-memory checkpointer owned by the adapter, visibility undeclared",
                _expected_support(SUPPORTED, _UNKNOWN),
                [*_common_cases(), *_disposition_cases(OK, UNSURE)],
            ),
            lambda model: LangGraphHarness(model, model_id=MODEL_ID),
        ),
        _Profile(
            FixtureProfile(
                "supplied-checkpointer-undeclared",
                "caller-supplied checkpointer without disposition declarations",
                _expected_support(_UNKNOWN, _UNKNOWN),
                [*_common_cases(), *_disposition_cases(UNSURE, UNSURE)],
            ),
            lambda model: LangGraphHarness(model, model_id=MODEL_ID, checkpointer=InMemorySaver()),
        ),
        _Profile(
            FixtureProfile(
                "declared-ephemeral-hidden",
                "caller-supplied checkpointer declared ephemeral, inputs declared hidden",
                _expected_support(SUPPORTED, SUPPORTED),
                [*_common_cases(), *_disposition_cases(OK, OK)],
            ),
            lambda model: LangGraphHarness(
                model,
                model_id=MODEL_ID,
                checkpointer=InMemorySaver(),
                context_retention=ContextRetentionDisposition.EPHEMERAL,
                history_visibility=UserHistoryVisibility.HIDDEN,
            ),
        ),
        _Profile(
            FixtureProfile(
                "declared-materialized-visible",
                "caller-supplied durable checkpointer, inputs visible elsewhere",
                _expected_support(_UNSUPPORTED, _UNSUPPORTED),
                [*_common_cases(), *_disposition_cases(NO, NO)],
            ),
            lambda model: LangGraphHarness(
                model,
                model_id=MODEL_ID,
                checkpointer=InMemorySaver(),
                context_retention=ContextRetentionDisposition.MATERIALIZED,
                history_visibility=UserHistoryVisibility.VISIBLE,
            ),
        ),
    )


class _RequirementsFixture:
    def __init__(self, observation: ModelBoundaryObservation) -> None:
        self._observation = observation
        self._profiles = {profile.declared.id: profile for profile in _profiles()}

    @property
    def provider(self) -> ProviderId:
        return PROVIDER

    @property
    def observation(self) -> RuntimeObservation:
        return self._observation

    def profiles(self) -> Sequence[FixtureProfile]:
        return tuple(profile.declared for profile in self._profiles.values())

    def create_harness(self, profile_id: str) -> AgentHarness:
        return self._profiles[profile_id].build(self._observation.model)


@asynccontextmanager
async def _requirements_fixture() -> AsyncGenerator[RuntimeRequirementsFixture, None]:
    async with _boundary() as observation:
        yield _RequirementsFixture(observation)


class TestLangGraphRequirements(RequirementsConformanceTests):
    """Requirements suite over four disposition profiles."""

    def requirement_fixture(self) -> FixtureResource[RuntimeRequirementsFixture]:
        """Provide the profile fixture."""
        return _requirements_fixture()
