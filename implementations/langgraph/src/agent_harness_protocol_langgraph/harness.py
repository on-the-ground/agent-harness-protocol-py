"""LangChain + LangGraph implementation of the Agent Harness Protocol port.

`LangGraphHarness` runs a LangChain chat model inside a ``START -> model -> END``
LangGraph graph, one checkpoint thread per AHP session. Requirements the graph cannot
keep are rejected at admission; see the package README for the support matrix.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import timedelta
from enum import StrEnum
from typing import Any
from uuid import uuid4

from agent_harness_protocol import (
    SUPPORTED,
    AgentHarness,
    AgentSession,
    AgentTask,
    AgentUsage,
    ApprovalRequirement,
    Cancelled,
    Capability,
    CleanupBudget,
    CompatibilityIssue,
    CompatibilityIssueKind,
    CompatibilityReport,
    Completed,
    ContextRetentionDisposition,
    ContextRetentionRequirement,
    DiagnosticEvent,
    DiagnosticGap,
    Failed,
    FailureKind,
    InteractionId,
    InteractionRequest,
    InteractionResponse,
    MessageCompleted,
    MessageId,
    MessageRole,
    NetworkAccess,
    ObservationGap,
    ProviderDiagnostic,
    ProviderId,
    QuestionRequirement,
    SessionBlockedError,
    SessionDisposition,
    SessionId,
    SessionSpec,
    StopReason,
    StructuredOutputRequirement,
    Support,
    SupportReport,
    TaskCancelled,
    TaskCompleted,
    TaskEvent,
    TaskFailed,
    TaskId,
    TaskOutcome,
    TaskRequest,
    TaskStarted,
    TaskState,
    TaskUnresolved,
    TextInput,
    TextOutput,
    UnknownSupport,
    Unresolved,
    UnresolvedReason,
    Unsupported,
    UsageChanged,
    UserHistoryVisibility,
    UserHistoryVisibilityRequirement,
)
from agent_harness_protocol.diagnostics import TaskDiagnostics
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from ._fanout import BoundedFanout
from ._graph import ModelOnlyGraph

PROVIDER = ProviderId("langchain-langgraph")


class CancellationSemantics(StrEnum):
    """What cancellation of the model coroutine proves at the configured boundary.

    Attributes:
        UNCONFIRMED: Coroutine termination is no evidence that the model provider
            stopped working. Cancelling a model call yields ``Unresolved``. This is the
            default and the right choice for remote providers.
        COROUTINE_TERMINATION_CONFIRMS_WORK_STOPPED: The model coroutine is the whole
            native work, for example an in-process model, so its termination yields
            ``Cancelled``.
    """

    UNCONFIRMED = "unconfirmed"
    COROUTINE_TERMINATION_CONFIRMS_WORK_STOPPED = "coroutine_termination_confirms_work_stopped"


class CheckpointCleanupError(RuntimeError):
    """Deleting a released session's LangGraph checkpoints failed or exceeded its budget.

    Task outcomes and closed handles are unaffected; only the stored context may remain.

    Attributes:
        failures: The deletion failure, or ``TimeoutError``, for each released session
            whose checkpoint thread may still exist.
    """

    def __init__(self, failures: Mapping[SessionId, BaseException]) -> None:
        """Create the error.

        Args:
            failures: The failure for each session whose checkpoint thread was not deleted.
        """
        self.failures: Mapping[SessionId, BaseException] = dict(failures)
        detail = "; ".join(
            f"{session_id.value}: {type(error).__name__}: {error}"
            for session_id, error in self.failures.items()
        )
        super().__init__(f"checkpoint cleanup failed for released sessions: {detail}")


_DEFAULT_BUDGET = CleanupBudget(
    per_task=timedelta(seconds=1),
    total=timedelta(seconds=3),
    aggregates_across_resources=True,
)


def _declared_support(declared: object, satisfies: object, capability: str) -> Support:
    """Map a declared disposition to discovery information for one capability."""
    if declared == satisfies:
        return SUPPORTED
    if declared in {ContextRetentionDisposition.UNKNOWN, UserHistoryVisibility.UNKNOWN}:
        return UnknownSupport(
            f"{capability} depends on the configured checkpointer, model provider, and tracing; "
            "declare it when constructing the harness"
        )
    return Unsupported(f"the harness was configured with {capability} {declared}")


def _declared_issue(path: str, declared: object, requirement: object) -> CompatibilityIssue:
    """Explain why a declared disposition cannot satisfy a requirement."""
    if declared in {ContextRetentionDisposition.UNKNOWN, UserHistoryVisibility.UNKNOWN}:
        return CompatibilityIssue(
            path,
            f"cannot confirm {requirement}: the harness did not declare this disposition",
            CompatibilityIssueKind.UNCONFIRMED,
        )
    return CompatibilityIssue(path, f"the harness was configured as {declared}, not {requirement}")


def _support_report(
    retention: ContextRetentionDisposition, visibility: UserHistoryVisibility
) -> SupportReport:
    """Build the support report for the given declared dispositions."""
    return SupportReport(
        {
            Capability.CALLER_APPROVAL: Unsupported(
                "this model-only graph has no caller approval interaction route"
            ),
            Capability.QUESTIONS: Unsupported(
                "this model-only graph has no caller question interaction route"
            ),
            Capability.PERSISTENCE: Unsupported(
                "the bundled in-memory checkpointer does not survive harness recreation"
            ),
            Capability.WORKSPACE: Unsupported(
                "this model-only graph does not establish a workspace"
            ),
            Capability.EXECUTION_CONSTRAINT: Unsupported(
                "this adapter cannot enforce model-provider filesystem or network policy"
            ),
            Capability.STRUCTURED_OUTPUT: Unsupported(
                "structured output validation is not part of this graph"
            ),
            Capability.DIAGNOSTICS: SUPPORTED,
            Capability.CONTEXT_RETENTION: _declared_support(
                retention, ContextRetentionDisposition.EPHEMERAL, "context retention"
            ),
            Capability.USER_HISTORY_VISIBILITY: _declared_support(
                visibility, UserHistoryVisibility.HIDDEN, "user-history visibility"
            ),
        }
    )


def _retrieve_result(task: asyncio.Future[Any]) -> None:
    """Mark a background result as observed; failures are reported through other paths."""
    if not task.cancelled():
        task.exception()


@dataclass(eq=False)
class _Runtime:
    """State shared by one harness and the sessions and tasks it owns."""

    model: BaseChatModel
    checkpointer: BaseCheckpointSaver[str]
    cancellation_semantics: CancellationSemantics
    cleanup_budget: CleanupBudget
    event_buffer_capacity: int
    diagnostic_buffer_capacity: int
    disposition: SessionDisposition
    sessions: set[_LangGraphSession] = field(default_factory=lambda: set[_LangGraphSession]())
    background: set[asyncio.Task[None]] = field(default_factory=lambda: set[asyncio.Task[None]]())
    closed: bool = False

    def spawn_background(self, work: Coroutine[Any, Any, None]) -> None:
        """Run detached cleanup work; its failure is observed but has no report path."""
        task = asyncio.create_task(work)
        self.background.add(task)
        task.add_done_callback(self.background.discard)
        task.add_done_callback(_retrieve_result)


class LangGraphHarness(AgentHarness):
    """Run a LangChain chat model through a stateful LangGraph graph.

    Facts the adapter cannot observe are declared at construction:

    * ``context_retention`` describes the checkpointer. Without a ``checkpointer`` the
      adapter owns an in-memory saver deleted on release, so it is ``EPHEMERAL``. A
      supplied checkpointer is ``UNKNOWN`` unless declared.
    * ``history_visibility`` describes whether inputs become visible outside this
      application, for example through provider logs or LangSmith tracing. It is
      ``UNKNOWN`` unless declared.

    A cancelled turn's input stays in the session's checkpoint and is part of the next
    task's context, because LangGraph records the input before the model node runs.
    """

    def __init__(
        self,
        model: BaseChatModel,
        *,
        model_id: str | None = None,
        checkpointer: BaseCheckpointSaver[str] | None = None,
        cancellation_semantics: CancellationSemantics = CancellationSemantics.UNCONFIRMED,
        cleanup_budget: CleanupBudget = _DEFAULT_BUDGET,
        event_buffer_capacity: int = 256,
        diagnostic_buffer_capacity: int = 256,
        context_retention: ContextRetentionDisposition | None = None,
        history_visibility: UserHistoryVisibility = UserHistoryVisibility.UNKNOWN,
    ) -> None:
        """Configure a harness around one chat model.

        Args:
            model: The LangChain chat model called by every session's model node.
            model_id: The model identity admitted for ``SessionSpec.model``. Without it,
                any explicit model request is rejected.
            checkpointer: The LangGraph checkpointer holding session threads. When omitted,
                the harness owns an in-memory saver and reports ``EPHEMERAL`` retention.
            cancellation_semantics: What coroutine termination proves about the model
                provider. See `CancellationSemantics`.
            cleanup_budget: Upper bounds for settling tasks and deleting threads during
                ``release()`` and ``aclose()``.
            event_buffer_capacity: Entries buffered per semantic-event subscription,
                including gap markers.
            diagnostic_buffer_capacity: Entries buffered per diagnostic subscription,
                including gap markers.
            context_retention: The declared retention of a supplied ``checkpointer``.
                Defaults to ``UNKNOWN`` for a supplied checkpointer and to ``EPHEMERAL``
                for the adapter-owned one.
            history_visibility: Whether inputs stay hidden outside the application, for
                example from provider logs and tracing. Defaults to ``UNKNOWN``.

        Raises:
            ValueError: If a buffer capacity is below two, or if a retention other than
                ``EPHEMERAL`` is declared for the adapter-owned checkpointer.
        """
        if checkpointer is None:
            if context_retention not in {None, ContextRetentionDisposition.EPHEMERAL}:
                raise ValueError("the adapter-owned in-memory checkpointer is ephemeral")
            context_retention = ContextRetentionDisposition.EPHEMERAL
        elif context_retention is None:
            context_retention = ContextRetentionDisposition.UNKNOWN
        if event_buffer_capacity < 2 or diagnostic_buffer_capacity < 2:
            raise ValueError("buffer capacities must be at least two")
        self._model_id = model_id
        self._runtime = _Runtime(
            model=model,
            checkpointer=checkpointer or InMemorySaver(),
            cancellation_semantics=cancellation_semantics,
            cleanup_budget=cleanup_budget,
            event_buffer_capacity=event_buffer_capacity,
            diagnostic_buffer_capacity=diagnostic_buffer_capacity,
            disposition=SessionDisposition(context_retention, history_visibility),
        )
        self._close_task: asyncio.Task[None] | None = None

    @property
    def provider(self) -> ProviderId:
        """The provider identity reported by this adapter."""
        return PROVIDER

    @property
    def support(self) -> SupportReport:
        """Capability discovery for this graph and the declared dispositions.

        Discovery is informational; `validate` and admission decide each request.
        """
        disposition = self._runtime.disposition
        return _support_report(disposition.retention, disposition.history_visibility)

    @property
    def cleanup_budget(self) -> CleanupBudget:
        """Upper bounds honoured by ``session.release()`` and `aclose`."""
        return self._runtime.cleanup_budget

    def validate(self, spec: SessionSpec) -> CompatibilityReport:
        """Diagnose a session spec without calling the model.

        Args:
            spec: The requested session configuration and guarantees.

        Returns:
            ``INCOMPATIBLE`` for requirements this graph cannot keep, ``UNCONFIRMED`` for
            requirements that depend on an undeclared disposition, otherwise
            ``COMPATIBLE``.
        """
        issues: list[CompatibilityIssue] = []
        requirements = spec.requirements
        if spec.model is not None and spec.model != self._model_id:
            issues.append(
                CompatibilityIssue(
                    "model",
                    "the requested model does not match the model bound to this harness",
                )
            )
        if requirements.approval not in {
            ApprovalRequirement.PROVIDER_DEFAULT,
            ApprovalRequirement.DENY_ALL,
        }:
            issues.append(
                CompatibilityIssue(
                    "requirements.approval",
                    "this model-only graph cannot route approval decisions",
                )
            )
        if requirements.questions is not QuestionRequirement.NOT_REQUIRED:
            issues.append(
                CompatibilityIssue(
                    "requirements.questions",
                    "this model-only graph cannot route caller answers",
                )
            )
        if requirements.persistence is not None:
            issues.append(
                CompatibilityIssue(
                    "requirements.persistence",
                    "in-memory checkpoints do not satisfy cross-harness persistence",
                )
            )
        if requirements.workspace is not None:
            issues.append(
                CompatibilityIssue(
                    "requirements.workspace",
                    "this model-only graph does not establish a workspace",
                )
            )
        if requirements.execution is not None:
            dimensions: list[str] = []
            if requirements.execution.filesystem is not None:
                dimensions.append("filesystem")
            if requirements.execution.network is not None:
                dimensions.append(
                    "network allowed"
                    if requirements.execution.network is NetworkAccess.ALLOWED
                    else "network denied"
                )
            issues.append(
                CompatibilityIssue(
                    "requirements.execution",
                    f"cannot enforce provider execution constraint: {', '.join(dimensions)}",
                )
            )
        retention = self._runtime.disposition.retention
        if requirements.retention is ContextRetentionRequirement.EPHEMERAL:
            if retention is not ContextRetentionDisposition.EPHEMERAL:
                issues.append(
                    _declared_issue("requirements.retention", retention, requirements.retention)
                )
        elif requirements.retention is not ContextRetentionRequirement.PROVIDER_DEFAULT:
            issues.append(
                CompatibilityIssue(
                    "requirements.retention", "unsupported context-retention requirement"
                )
            )
        visibility = self._runtime.disposition.history_visibility
        if requirements.history_visibility is UserHistoryVisibilityRequirement.HIDDEN:
            if visibility is not UserHistoryVisibility.HIDDEN:
                issues.append(
                    _declared_issue(
                        "requirements.history_visibility",
                        visibility,
                        requirements.history_visibility,
                    )
                )
        elif (
            requirements.history_visibility is not UserHistoryVisibilityRequirement.PROVIDER_DEFAULT
        ):
            issues.append(
                CompatibilityIssue(
                    "requirements.history_visibility",
                    "unsupported user-history visibility requirement",
                )
            )
        return CompatibilityReport(issues)

    async def create_session(self, spec: SessionSpec) -> AgentSession:
        """Admit a spec and create a session with its own checkpoint thread.

        Args:
            spec: The requested session configuration and guarantees.

        Returns:
            A new session whose sequential tasks share one conversation.

        Raises:
            RuntimeError: If the harness is closed.
            IncompatibleRequirementError: If a requirement cannot be kept.
            RequirementUnconfirmedError: If a requirement cannot be confirmed.
        """
        if self._runtime.closed:
            raise RuntimeError("harness is closed")
        self.validate(spec).require_compatible()
        session = _LangGraphSession(self._runtime, spec)
        self._runtime.sessions.add(session)
        return session

    async def aclose(self) -> None:
        """Release every session within ``cleanup_budget.total``.

        Idempotent. Cancelling the caller does not interrupt the cleanup.

        Raises:
            CheckpointCleanupError: If stored context could not be deleted. Every task
                is still settled and every handle is still closed in that case.
        """
        if self._close_task is None:
            self._runtime.closed = True
            self._close_task = asyncio.create_task(self._close_all())
            self._close_task.add_done_callback(_retrieve_result)
        await asyncio.shield(self._close_task)

    async def _close_all(self) -> None:
        """Release all sessions concurrently and force the ones exceeding the budget."""
        sessions = tuple(self._runtime.sessions)
        if not sessions:
            return
        releases = {asyncio.ensure_future(session.release()): session for session in sessions}
        done, pending = await asyncio.wait(
            releases, timeout=self._runtime.cleanup_budget.total.total_seconds()
        )
        for release in pending:
            releases[release].force_unresolved_cleanup()
            release.cancel()
        failures: dict[SessionId, BaseException] = {}
        for release in done:
            error = release.exception()
            if isinstance(error, CheckpointCleanupError):
                failures.update(error.failures)
            elif error is not None:
                failures[releases[release].id] = error
        if failures:
            raise CheckpointCleanupError(failures)


class _LangGraphSession(AgentSession):
    """One AHP session backed by one LangGraph checkpoint thread."""

    def __init__(self, runtime: _Runtime, spec: SessionSpec) -> None:
        self.runtime = runtime
        self._id = SessionId(uuid4().hex)
        self._spec = spec
        self.thread_id = f"ahp:{self._id.value}"
        # The task whose graph run has not ended yet. It may already be settled when the
        # adapter could not confirm termination in time; the session stays blocked then.
        self._running: _LangGraphTask | None = None
        self._closed = False
        self._release_task: asyncio.Task[None] | None = None
        self._usage = AgentUsage.ZERO
        self.graph = ModelOnlyGraph(self._call_model, runtime.checkpointer)

    async def _call_model(self, history: Sequence[BaseMessage]) -> BaseMessage:
        """Model node body: prepend the session instructions and call the chat model."""
        running = self._running
        if running is not None:
            running.mark_model_called()
        messages = list(history)
        if self._spec.instructions is not None:
            messages.insert(0, SystemMessage(content=self._spec.instructions))
        return await self.runtime.model.ainvoke(messages)

    @property
    def id(self) -> SessionId:
        """Adapter-local session identifier."""
        return self._id

    @property
    def spec(self) -> SessionSpec:
        """The admitted session spec."""
        return self._spec

    @property
    def disposition(self) -> SessionDisposition:
        """Retention and visibility as declared to the harness."""
        return self.runtime.disposition

    @property
    def persistent_ref(self) -> None:
        """Always ``None``: this adapter does not offer persistent sessions."""
        return None

    def validate(self, request: TaskRequest) -> CompatibilityReport:
        """Diagnose a task request; only text input with text output is compatible."""
        issues: list[CompatibilityIssue] = []
        if not isinstance(request.input, TextInput):
            issues.append(
                CompatibilityIssue("input", "this graph accepts only provider-independent text")
            )
        if isinstance(request.requirements.output, StructuredOutputRequirement):
            issues.append(
                CompatibilityIssue(
                    "requirements.output",
                    "this graph does not validate structured model output",
                )
            )
        return CompatibilityReport(issues)

    async def start_task(self, request: TaskRequest) -> AgentTask:
        """Admit a request and start its graph run without waiting for completion.

        Raises:
            SessionBlockedError: If the session was released, or a previous graph run
                that outlived its settled task is still running.
            RuntimeError: If another task of this session is still running.
            IncompatibleRequirementError: If the request cannot be served.
        """
        if self._closed or self.runtime.closed:
            raise SessionBlockedError(self.id, "session has been released")
        running = self._running
        if running is not None:
            if running.state.is_terminal:
                raise SessionBlockedError(
                    self.id,
                    "the previous graph run has not ended although its task was settled",
                )
            raise RuntimeError("a session cannot run overlapping tasks")
        self.validate(request).require_compatible()
        text = request.input
        assert isinstance(text, TextInput)
        task = _LangGraphTask(self, text)
        self._running = task
        task.start()
        return task

    async def release(self) -> None:
        """Idempotently release the session within the harness cleanup budget.

        Active work is cancelled, then the checkpoint thread is deleted. Cancelling the
        caller does not interrupt the release.

        Raises:
            CheckpointCleanupError: If the checkpoint thread could not be deleted.
        """
        if self._release_task is None:
            self._closed = True
            self._release_task = asyncio.create_task(self._release())
            self._release_task.add_done_callback(_retrieve_result)
        await asyncio.shield(self._release_task)

    async def _release(self) -> None:
        """Stop the running task, delete the thread, and detach from the harness."""
        loop = asyncio.get_running_loop()
        started = loop.time()
        budget = self.runtime.cleanup_budget
        try:
            running = self._running
            if running is not None:
                await running.stop(
                    budget.per_task.total_seconds(), UnresolvedReason.CLEANUP_BOUND_EXCEEDED
                )
            remaining = max(0.0, budget.total.total_seconds() - (loop.time() - started))
            try:
                async with asyncio.timeout(remaining):
                    await self.runtime.checkpointer.adelete_thread(self.thread_id)
            except Exception as error:
                raise CheckpointCleanupError({self.id: error}) from error
        finally:
            self.runtime.sessions.discard(self)
            self._delete_after_graph_run()

    def _delete_after_graph_run(self) -> None:
        """A graph run that outlived cleanup may still write; delete again once it ends."""
        running = self._running
        if running is None:
            return
        runtime = self.runtime
        thread_id = self.thread_id
        running.when_graph_run_ends(
            lambda: runtime.spawn_background(runtime.checkpointer.adelete_thread(thread_id))
        )

    def force_unresolved_cleanup(self) -> None:
        """Close the session and settle its running task after the total budget expired."""
        self._closed = True
        running = self._running
        if running is not None:
            running.force_unresolved(
                UnresolvedReason.CLEANUP_BOUND_EXCEEDED,
                "cleanup ended before native termination could be confirmed",
            )

    def projected_usage(self, task_usage: AgentUsage) -> AgentUsage:
        """Session usage if ``task_usage`` were committed now."""
        return self._usage + task_usage

    def commit_usage(self, task_usage: AgentUsage) -> AgentUsage:
        """Add one settled task's usage exactly once and return the session total."""
        self._usage = self._usage + task_usage
        return self._usage

    def graph_run_ended(self, task: _LangGraphTask) -> None:
        """Unblock the session once the given task's graph run has really ended."""
        if self._running is task:
            self._running = None


class _LangGraphTask(AgentTask, TaskDiagnostics):
    """One delegated task, executed as one graph run on the session thread."""

    def __init__(self, session: _LangGraphSession, text: TextInput) -> None:
        self._session = session
        self._text = text
        self._id = TaskId(uuid4().hex)
        self._state = TaskState.STARTING
        self._outcome: asyncio.Future[TaskOutcome] = asyncio.get_running_loop().create_future()
        self._worker: asyncio.Task[TaskOutcome] | None = None
        self._model_called = False
        self._cancel_sent = False
        self._observed_usage: AgentUsage | None = None
        task_id = self._id
        self._events = BoundedFanout[TaskEvent](
            capacity=session.runtime.event_buffer_capacity,
            gap=lambda count: ObservationGap(task_id, count),
        )
        self._diagnostics = BoundedFanout[DiagnosticEvent](
            capacity=session.runtime.diagnostic_buffer_capacity,
            gap=lambda count: DiagnosticGap(task_id, count),
        )

    @property
    def id(self) -> TaskId:
        """Adapter-local task identifier."""
        return self._id

    @property
    def session_id(self) -> SessionId:
        """Identifier of the owning session."""
        return self._session.id

    @property
    def state(self) -> TaskState:
        """Current lifecycle snapshot."""
        return self._state

    def events(self) -> AsyncIterator[TaskEvent]:
        """Open a bounded semantic-event subscription registered immediately."""
        return self._events.subscribe()

    def diagnostics(self) -> AsyncIterator[DiagnosticEvent]:
        """Open a bounded subscription to graph start and terminal diagnostics."""
        return self._diagnostics.subscribe()

    @property
    def pending_interactions(self) -> tuple[InteractionRequest, ...]:
        """Always empty: this graph has no approval or question route."""
        return ()

    async def respond(self, interaction_id: InteractionId, response: InteractionResponse) -> None:
        """Reject every response because this graph never requests an interaction.

        Raises:
            ValueError: Always.
        """
        raise ValueError(f"task has no pending interaction: {interaction_id}")

    async def request_cancellation(self) -> None:
        """Cancel the graph run and wait at most ``cleanup_budget.per_task``.

        Returns once the task is settled. If the run does not end in time, the task is
        settled as ``Unresolved`` and the session stays blocked until the run ends.
        """
        if self._outcome.done():
            return
        await self.stop(
            self._session.runtime.cleanup_budget.per_task.total_seconds(),
            UnresolvedReason.CANCELLATION_UNCONFIRMED,
        )

    async def await_outcome(self) -> TaskOutcome:
        """Return the stable terminal outcome; cancelling the waiter leaves the task running."""
        return await asyncio.shield(self._outcome)

    # Adapter-internal protocol used by the session.

    def start(self) -> None:
        """Start the graph run in a background task."""
        worker = asyncio.create_task(self._run())
        self._worker = worker
        worker.add_done_callback(self._graph_run_ended)

    def mark_model_called(self) -> None:
        """Record that the model boundary was reached, which makes cancellation uncertain."""
        self._model_called = True

    def when_graph_run_ends(self, callback: Callable[[], None]) -> None:
        """Call ``callback`` once the graph run has ended, immediately if it already has."""
        worker = self._worker
        if worker is None or worker.done():
            callback()
        else:
            worker.add_done_callback(lambda _: callback())

    async def stop(self, timeout: float, reason: UnresolvedReason) -> None:
        """Cancel the graph run and settle within ``timeout`` without hiding uncertainty."""
        worker = self._worker
        if worker is None:
            return
        self._cancel_graph_run()
        await asyncio.wait({worker}, timeout=timeout)
        if worker.done():
            # The done callback may still be queued when the worker had already ended.
            self._graph_run_ended(worker)
        else:
            self.force_unresolved(
                reason, "the graph run did not end within the cleanup budget after cancellation"
            )

    def force_unresolved(self, reason: UnresolvedReason, known: str) -> None:
        """Settle as ``Unresolved`` now while the graph run may still be running."""
        self._cancel_graph_run()
        self._settle(Unresolved(self.id, reason, known, usage=self._usage_so_far()))

    def _cancel_graph_run(self) -> None:
        """Deliver one cancellation request; repeating it would only interrupt native cleanup."""
        worker = self._worker
        if worker is None or worker.done() or self._cancel_sent:
            return
        self._cancel_sent = True
        worker.cancel()

    async def _run(self) -> TaskOutcome:
        """Execute one graph turn and translate the final AI message into events."""
        self._state = TaskState.RUNNING
        self._events.publish(TaskStarted(self.id))
        self._diagnostics.publish(self._diagnostic("graph_started", {"node": "model"}))
        response = await self._session.graph.run_turn(self._session.thread_id, self._text.text)
        if response is None:
            return Completed(self.id, StopReason.PROVIDER_STOPPED, None, self._usage_so_far())
        text = _message_text(response)
        self._events.publish(
            MessageCompleted(
                self.id, MessageId(response.id or uuid4().hex), text, MessageRole.ANSWER
            )
        )
        usage = _usage(response)
        self._observed_usage = usage
        self._events.publish(UsageChanged(self.id, usage, self._session.projected_usage(usage)))
        return Completed(self.id, _stop_reason(response), TextOutput(text), usage)

    def _graph_run_ended(self, worker: asyncio.Task[TaskOutcome]) -> None:
        """Settle from the finished graph run unless a forced judgment came first."""
        self._session.graph_run_ended(self)
        if self._outcome.done():
            _retrieve_result(worker)
            return
        if worker.cancelled():
            outcome = self._cancellation_outcome()
        else:
            error = worker.exception()
            if error is None:
                outcome = worker.result()
            else:
                outcome = Failed(
                    self.id,
                    FailureKind.UNKNOWN,
                    str(error) or type(error).__name__,
                    error,
                    usage=self._usage_so_far(),
                )
        self._settle(outcome)

    def _cancellation_outcome(self) -> TaskOutcome:
        """Judge a cancelled graph run by whether the model was reached and the semantics."""
        usage = self._usage_so_far()
        semantics = self._session.runtime.cancellation_semantics
        if (
            not self._model_called
            or semantics is CancellationSemantics.COROUTINE_TERMINATION_CONFIRMS_WORK_STOPPED
        ):
            return Cancelled(self.id, usage=usage)
        return Unresolved(
            self.id,
            UnresolvedReason.CANCELLATION_UNCONFIRMED,
            "the LangChain coroutine ended without provider termination evidence",
            usage=usage,
        )

    def _usage_so_far(self) -> AgentUsage:
        """Observed usage, measured zero before the model call, otherwise unknown."""
        if self._observed_usage is not None:
            return self._observed_usage
        return AgentUsage.UNKNOWN if self._model_called else AgentUsage.ZERO

    def _settle(self, outcome: TaskOutcome) -> None:
        """Fix the terminal outcome once, commit session usage, and close the streams."""
        if self._outcome.done():
            return
        session_usage = self._session.commit_usage(outcome.usage)
        if isinstance(outcome, Completed):
            final = replace(outcome, session_usage=session_usage)
            self._state = TaskState.COMPLETED
            terminal: TaskEvent = TaskCompleted(self.id, final)
        elif isinstance(outcome, Failed):
            final = replace(outcome, session_usage=session_usage)
            self._state = TaskState.FAILED
            terminal = TaskFailed(self.id, final)
        elif isinstance(outcome, Cancelled):
            final = replace(outcome, session_usage=session_usage)
            self._state = TaskState.CANCELLED
            terminal = TaskCancelled(self.id, final)
        else:
            assert isinstance(outcome, Unresolved)
            final = replace(outcome, session_usage=session_usage)
            self._state = TaskState.UNRESOLVED
            terminal = TaskUnresolved(self.id, final)
        self._diagnostics.publish(self._diagnostic("graph_terminal", {"state": self._state.value}))
        self._events.close(terminal)
        self._diagnostics.close()
        self._outcome.set_result(final)

    def _diagnostic(self, name: str, payload: Mapping[str, str]) -> ProviderDiagnostic:
        """Build a provider diagnostic with a JSON payload."""
        return ProviderDiagnostic(self.id, PROVIDER, name, json.dumps(payload, sort_keys=True))


def _message_text(message: AIMessage) -> str:
    """Concatenate the text content of an AI message, ignoring non-text blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    fragments: list[str] = []
    for block in content:
        if isinstance(block, str):
            fragments.append(block)
        elif block.get("type") == "text":
            value = block.get("text")
            if isinstance(value, str):
                fragments.append(value)
    return "".join(fragments)


def _usage(message: AIMessage) -> AgentUsage:
    """Translate LangChain usage metadata, keeping missing fields unknown."""
    metadata = message.usage_metadata
    if metadata is None:
        return AgentUsage.UNKNOWN
    input_details = metadata.get("input_token_details") or {}
    output_details = metadata.get("output_token_details") or {}
    return AgentUsage(
        input_tokens=metadata.get("input_tokens"),
        cached_input_tokens=input_details.get("cache_read"),
        output_tokens=metadata.get("output_tokens"),
        reasoning_tokens=output_details.get("reasoning"),
        total_tokens=metadata.get("total_tokens"),
        cache_write_input_tokens=input_details.get("cache_creation"),
    )


def _stop_reason(message: AIMessage) -> StopReason:
    """Map the provider finish reason; an unrecognised reason is ``PROVIDER_STOPPED``."""
    raw = message.response_metadata.get("finish_reason")
    if raw in {"stop", "end_turn", "completed"}:
        return StopReason.FINISHED
    if raw in {"length", "max_tokens", "max_iterations"}:
        return StopReason.ITERATION_LIMIT
    return StopReason.PROVIDER_STOPPED if raw else StopReason.FINISHED
