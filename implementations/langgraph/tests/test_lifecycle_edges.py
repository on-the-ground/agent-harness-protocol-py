"""Adapter-specific lifecycle edges that the shared conformance suites do not reach."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import timedelta
from time import monotonic
from typing import Any

import pytest
from agent_harness_protocol import (
    AgentSession,
    AgentTask,
    AgentUsage,
    Cancelled,
    CleanupBudget,
    Completed,
    Failed,
    FailureKind,
    SessionBlockedError,
    SessionSpec,
    TaskCancelled,
    TaskEvent,
    TaskRequest,
    TaskState,
    TerminalEvent,
    TextInput,
    Unresolved,
    UnresolvedReason,
)
from langchain_core.messages import UsageMetadata
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from model_boundary import ControlledChatModel, contents

from agent_harness_protocol_langgraph import (
    CancellationSemantics,
    CheckpointCleanupError,
    LangGraphHarness,
)

BUDGET = CleanupBudget(
    per_task=timedelta(milliseconds=200),
    total=timedelta(milliseconds=600),
    aggregates_across_resources=True,
)
MARGIN = 0.3
SPEC = SessionSpec(instructions="AHP_TEST_INSTRUCTION")
BOTH_SEMANTICS = pytest.mark.parametrize("semantics", list(CancellationSemantics))


class FaultySaver(InMemorySaver):
    """A real in-memory checkpointer whose thread deletion can fail or stall."""

    def __init__(self) -> None:
        """Start with deletion succeeding."""
        # InMemorySaver.__init__ exposes a partially typed ``factory`` parameter.
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.failure: Exception | None = None
        self.stall = False
        self.deleted: list[str] = []

    async def adelete_thread(self, thread_id: str) -> None:
        """Stall or fail when configured, otherwise delete and record the thread."""
        if self.stall:
            await asyncio.sleep(60)
        if self.failure is not None:
            raise self.failure
        self.deleted.append(thread_id)
        await super().adelete_thread(thread_id)


def _harness(
    model: ControlledChatModel,
    semantics: CancellationSemantics = CancellationSemantics.UNCONFIRMED,
    saver: InMemorySaver | None = None,
) -> LangGraphHarness:
    return LangGraphHarness(
        model,
        checkpointer=saver,
        cancellation_semantics=semantics,
        cleanup_budget=BUDGET,
    )


def _request(text: str) -> TaskRequest:
    return TaskRequest(TextInput(text))


async def _collect(stream: AsyncIterator[TaskEvent]) -> list[TaskEvent]:
    return [event async for event in stream]


async def _until(condition: Callable[[], bool]) -> None:
    async with asyncio.timeout(5):
        while not condition():
            await asyncio.sleep(0.005)


async def _stored_threads(saver: InMemorySaver) -> set[str]:
    threads: set[str] = set()
    async for checkpoint in saver.alist(None):
        config: RunnableConfig = checkpoint.config
        configurable: dict[str, Any] = config.get("configurable") or {}
        threads.add(str(configurable["thread_id"]))
    return threads


def _usage(input_tokens: int, output_tokens: int) -> UsageMetadata:
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def _last_inputs(model: ControlledChatModel) -> Sequence[str]:
    return contents(model.calls[-1])


@BOTH_SEMANTICS
async def test_cancellation_before_graph_run_starts_is_confirmed(
    semantics: CancellationSemantics,
) -> None:
    model = ControlledChatModel()
    async with _harness(model, semantics) as harness:
        session = await harness.create_session(SPEC)
        task = await session.start_task(_request("never sent"))
        events = asyncio.create_task(_collect(task.events()))
        # Awaited directly so that nothing yields to the event loop before cancellation.
        await task.request_cancellation()

        outcome = await asyncio.wait_for(task.await_outcome(), 1)
        assert isinstance(outcome, Cancelled)
        assert outcome.usage == AgentUsage.ZERO
        assert task.state is TaskState.CANCELLED
        assert model.calls == ()
        received = await asyncio.wait_for(events, 1)
        assert isinstance(received[-1], TaskCancelled)
        assert sum(isinstance(event, TerminalEvent) for event in received) == 1

        follow_up = await session.start_task(_request("after early cancel"))
        assert isinstance(await asyncio.wait_for(follow_up.await_outcome(), 5), Completed)
        assert "never sent" not in _last_inputs(model)


async def test_release_right_after_start_settles_within_budget() -> None:
    model = ControlledChatModel()
    async with _harness(model) as harness:
        session = await harness.create_session(SPEC)
        model.hold()
        task = await session.start_task(_request("released immediately"))
        started = monotonic()
        await session.release()
        assert monotonic() - started <= BUDGET.total.total_seconds() + MARGIN

        outcome = await asyncio.wait_for(task.await_outcome(), 1)
        # Stopping is confirmed only if the graph had not reached the model yet.
        assert isinstance(outcome, Unresolved if model.calls else Cancelled)
        model.release()


async def test_mid_call_cancellation_is_unresolved_by_default() -> None:
    model = ControlledChatModel()
    async with _harness(model) as harness:
        session = await harness.create_session(SPEC)
        model.hold()
        task = await session.start_task(_request("in flight"))
        await _until(lambda: len(model.calls) == 1)
        await asyncio.wait_for(task.request_cancellation(), 1)

        outcome = await task.await_outcome()
        assert isinstance(outcome, Unresolved)
        assert outcome.reason is UnresolvedReason.CANCELLATION_UNCONFIRMED
        assert outcome.usage == AgentUsage.UNKNOWN
        assert model.cancellations == 1


async def test_graph_run_ignoring_cancellation_is_bounded_and_blocks_session() -> None:
    model = ControlledChatModel(ignore_cancellation=True, usage=_usage(3, 4))
    async with _harness(
        model, CancellationSemantics.COROUTINE_TERMINATION_CONFIRMS_WORK_STOPPED
    ) as harness:
        session = await harness.create_session(SPEC)
        model.hold()
        task = await session.start_task(_request("stubborn"))
        events = asyncio.create_task(_collect(task.events()))
        await _until(lambda: len(model.calls) == 1)

        started = monotonic()
        await asyncio.wait_for(task.request_cancellation(), 5)
        assert monotonic() - started <= BUDGET.per_task.total_seconds() + MARGIN

        outcome = await asyncio.wait_for(task.await_outcome(), 1)
        assert isinstance(outcome, Unresolved)
        assert outcome.reason is UnresolvedReason.CANCELLATION_UNCONFIRMED
        with pytest.raises(SessionBlockedError):
            await session.start_task(_request("while graph still runs"))
        assert len(model.calls) == 1

        model.release()
        await _until(lambda: model.completed == 1)
        next_task = await _start_when_unblocked(session, _request("after graph ended"))
        next_outcome = await asyncio.wait_for(next_task.await_outcome(), 5)

        assert await task.await_outcome() == outcome
        assert task.state is TaskState.UNRESOLVED
        received = await asyncio.wait_for(events, 1)
        assert sum(isinstance(event, TerminalEvent) for event in received) == 1
        assert isinstance(next_outcome, Completed)
        # The late native result was never attributed to the unresolved task.
        assert next_outcome.session_usage == AgentUsage.UNKNOWN


async def _start_when_unblocked(session: AgentSession, request: TaskRequest) -> AgentTask:
    async with asyncio.timeout(5):
        while True:
            try:
                return await session.start_task(request)
            except SessionBlockedError:
                await asyncio.sleep(0.005)


async def test_caller_cancellation_of_request_cancellation_propagates() -> None:
    model = ControlledChatModel(ignore_cancellation=True)
    async with _harness(model) as harness:
        session = await harness.create_session(SPEC)
        model.hold()
        task = await session.start_task(_request("caller gives up"))
        await _until(lambda: len(model.calls) == 1)
        requester = asyncio.create_task(task.request_cancellation())
        await _until(lambda: model.cancellations == 1)
        requester.cancel()
        with pytest.raises(asyncio.CancelledError):
            await requester

        model.release()
        outcome = await asyncio.wait_for(task.await_outcome(), 5)
        assert isinstance(outcome, Unresolved)


async def test_session_usage_is_committed_once_per_task() -> None:
    model = ControlledChatModel(usage=_usage(10, 5))
    async with _harness(model) as harness:
        session = await harness.create_session(SPEC)
        first = await (await session.start_task(_request("first"))).await_outcome()
        cancelled_task = await session.start_task(_request("cancelled before model"))
        await cancelled_task.request_cancellation()
        cancelled = await cancelled_task.await_outcome()
        model.usage = _usage(1, 2)
        third = await (await session.start_task(_request("third"))).await_outcome()

    assert isinstance(first, Completed)
    assert first.session_usage == first.usage
    assert isinstance(cancelled, Cancelled)
    assert cancelled.session_usage == first.usage
    assert isinstance(third, Completed)
    assert third.session_usage == first.usage + third.usage
    assert third.session_usage is not None
    assert (third.session_usage.input_tokens, third.session_usage.total_tokens) == (11, 18)


async def test_model_failure_is_failed_with_cause_and_session_stays_usable() -> None:
    model = ControlledChatModel(failure="provider exploded")
    async with _harness(model) as harness:
        session = await harness.create_session(SPEC)
        outcome = await (await session.start_task(_request("fails"))).await_outcome()
        assert isinstance(outcome, Failed)
        assert outcome.kind is FailureKind.UNKNOWN
        assert isinstance(outcome.cause, RuntimeError)
        assert "provider exploded" in outcome.message
        assert outcome.usage == AgentUsage.UNKNOWN

        model.failure = None
        retry = await (await session.start_task(_request("retry"))).await_outcome()
        assert isinstance(retry, Completed)


async def test_checkpoint_delete_failure_is_reported_without_touching_outcomes() -> None:
    model = ControlledChatModel()
    saver = FaultySaver()
    harness = _harness(model, saver=saver)
    session = await harness.create_session(SPEC)
    task = await session.start_task(_request("stored context"))
    outcome = await task.await_outcome()
    saver.failure = OSError("storage down")

    with pytest.raises(CheckpointCleanupError) as first:
        await session.release()
    assert isinstance(first.value.failures[session.id], OSError)
    with pytest.raises(CheckpointCleanupError):
        await session.release()
    with pytest.raises(SessionBlockedError):
        await session.start_task(_request("after failed release"))
    assert await task.await_outcome() == outcome
    assert task.state is TaskState.COMPLETED

    # The released session no longer belongs to the harness close operation.
    saver.failure = None
    await harness.aclose()


async def test_close_aggregates_cleanup_failures_after_settling_every_task() -> None:
    model = ControlledChatModel()
    saver = FaultySaver()
    harness = _harness(model, saver=saver)
    sessions = [await harness.create_session(SPEC) for _ in range(2)]
    model.hold()
    tasks = [
        await session.start_task(_request(f"close-{index}"))
        for index, session in enumerate(sessions)
    ]
    await _until(lambda: len(model.calls) == 2)
    saver.failure = OSError("storage down")

    started = monotonic()
    with pytest.raises(CheckpointCleanupError) as raised:
        await harness.aclose()
    assert monotonic() - started <= BUDGET.total.total_seconds() + MARGIN
    assert set(raised.value.failures) == {session.id for session in sessions}
    for task in tasks:
        assert task.state.is_terminal
        assert isinstance(await asyncio.wait_for(task.await_outcome(), 1), Unresolved)


async def test_stalled_checkpoint_delete_stays_within_total_budget() -> None:
    model = ControlledChatModel()
    saver = FaultySaver()
    async with _harness(model, saver=saver) as harness:
        session = await harness.create_session(SPEC)
        await (await session.start_task(_request("stall"))).await_outcome()
        saver.stall = True

        started = monotonic()
        with pytest.raises(CheckpointCleanupError) as raised:
            await session.release()
        assert monotonic() - started <= BUDGET.total.total_seconds() + MARGIN
        assert isinstance(raised.value.failures[session.id], TimeoutError)
        saver.stall = False


async def test_close_beyond_budget_deletes_context_once_graph_run_ends() -> None:
    model = ControlledChatModel(ignore_cancellation=True)
    saver = FaultySaver()
    harness = _harness(model, saver=saver)
    session = await harness.create_session(SPEC)
    await (await session.start_task(_request("stored earlier"))).await_outcome()
    model.hold()
    task = await session.start_task(_request("outlives cleanup"))
    await _until(lambda: len(model.calls) == 2)

    started = monotonic()
    await harness.aclose()
    assert monotonic() - started <= BUDGET.total.total_seconds() + MARGIN
    outcome = await asyncio.wait_for(task.await_outcome(), 1)
    assert isinstance(outcome, Unresolved)
    assert outcome.reason is UnresolvedReason.CLEANUP_BOUND_EXCEEDED

    model.release()
    await _until(lambda: model.completed == 2)
    await _until(lambda: len(saver.deleted) == 2)
    assert await _stored_threads(saver) == set()
    assert await task.await_outcome() == outcome
