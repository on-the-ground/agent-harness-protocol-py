"""Controlled native-runtime conformance scenarios."""

import asyncio
from dataclasses import replace
from time import monotonic

import pytest

from ..compatibility import CompatibilityStatus, IncompatibleRequirementError
from ..diagnostics import ProviderDiagnostic, TaskDiagnostics
from ..events import TaskCompleted, TaskState, TerminalEvent
from ..harness import AgentHarness, PersistentSessions
from ..identity import StorageNamespace
from ..outcome import Cancelled, Completed, TextOutput, Unresolved
from ..requirements import (
    DiagnosticsRequirement,
    PersistenceRequirement,
    SessionRequirements,
    SessionSpec,
    StructuredOutputRequirement,
    TaskRequest,
    TaskRequirements,
    TextInput,
)
from ..support import Capability, Supported
from ._support import cancel_waiter, collect, managed, wait_until, within
from .fixtures import FixtureResource, RuntimeObservation


class RuntimeConformanceTests:
    def boundary(self) -> FixtureResource[RuntimeObservation]:
        raise NotImplementedError

    def harness(self, boundary: RuntimeObservation) -> AgentHarness:
        raise NotImplementedError

    def spec(self) -> SessionSpec:
        return SessionSpec(instructions="AHP_NATIVE_INSTRUCTION")

    async def test_native_task_completes_without_observer_and_preserves_state_output(self) -> None:
        async with managed(self.boundary()) as boundary, self.harness(boundary) as harness:
            task = await (await harness.create_session(self.spec())).start_task(
                TaskRequest(TextInput("first marker-alpha"))
            )
            outcome = await within(60, task.await_outcome())
            assert isinstance(outcome, Completed)
            assert task.state is TaskState.COMPLETED
            assert isinstance(outcome.output, TextOutput)
            assert outcome.output.text == "native-result"
            assert not task.pending_interactions
            assert any("marker-alpha" in value for value in boundary.observed_contexts)
            assert any("AHP_NATIVE_INSTRUCTION" in value for value in boundary.observed_contexts)

    async def test_whitespace_only_input_reaches_native_model_without_trimming(self) -> None:
        async with managed(self.boundary()) as boundary, self.harness(boundary) as harness:
            whitespace = " \t  "
            task = await (await harness.create_session(self.spec())).start_task(
                TaskRequest(TextInput(whitespace))
            )
            assert isinstance(await within(60, task.await_outcome()), Completed)
            assert whitespace in boundary.observed_text_values

    async def test_same_session_carries_context_while_new_session_stays_isolated(self) -> None:
        async with managed(self.boundary()) as boundary, self.harness(boundary) as harness:
            session = await harness.create_session(self.spec())
            assert isinstance(
                await within(
                    60,
                    (
                        await session.start_task(TaskRequest(TextInput("remember marker-alpha")))
                    ).await_outcome(),
                ),
                Completed,
            )
            first_count = len(boundary.observed_contexts)
            assert isinstance(
                await within(
                    60,
                    (
                        await session.start_task(TaskRequest(TextInput("followup marker-beta")))
                    ).await_outcome(),
                ),
                Completed,
            )
            subsequent = " ".join(boundary.observed_contexts[first_count:])
            assert "marker-alpha" in subsequent
            assert "native-result" in subsequent
            next_count = len(boundary.observed_contexts)
            independent = await harness.create_session(self.spec())
            await within(
                60,
                (
                    await independent.start_task(TaskRequest(TextInput("independent marker-gamma")))
                ).await_outcome(),
            )
            actual = " ".join(boundary.observed_contexts[next_count:])
            assert "marker-gamma" in actual
            assert "marker-alpha" not in actual

    async def test_overlap_rejected_and_cancelled_waiter_does_not_cancel_native_work(self) -> None:
        async with managed(self.boundary()) as boundary, self.harness(boundary) as harness:
            session = await harness.create_session(self.spec())
            boundary.hold()
            task = await session.start_task(TaskRequest(TextInput("held native task")))
            await wait_until(
                lambda: any("held native task" in value for value in boundary.observed_contexts)
            )
            first = asyncio.create_task(task.await_outcome())
            second = asyncio.create_task(task.await_outcome())
            await asyncio.sleep(0)
            await cancel_waiter(first)
            assert not task.state.is_terminal
            with pytest.raises(RuntimeError):
                await session.start_task(TaskRequest(TextInput("must not be sent")))
            assert all("must not be sent" not in value for value in boundary.observed_contexts)
            boundary.release()
            outcome = await within(60, second)
            assert isinstance(outcome, Completed)
            assert await task.await_outcome() == outcome

    async def test_close_bounds_active_native_work_and_settles_waiter(self) -> None:
        async with managed(self.boundary()) as boundary:
            harness = self.harness(boundary)
            try:
                session = await harness.create_session(self.spec())
                boundary.hold()
                task = await session.start_task(TaskRequest(TextInput("hold until cleanup")))
                await wait_until(
                    lambda: any(
                        "hold until cleanup" in value for value in boundary.observed_contexts
                    )
                )
                started = monotonic()
                await harness.aclose()
                elapsed = monotonic() - started
                assert elapsed <= harness.cleanup_budget.total.total_seconds() + 2
                outcome = await within(2, task.await_outcome())
                assert isinstance(outcome, (Cancelled, Unresolved))
                assert task.state.is_terminal
                assert not task.pending_interactions
            finally:
                boundary.release()
                await harness.aclose()

    async def test_explicit_cancellation_confirms_termination_and_session_is_reusable(self) -> None:
        async with managed(self.boundary()) as boundary, self.harness(boundary) as harness:
            session = await harness.create_session(self.spec())
            boundary.hold()
            task = await session.start_task(TaskRequest(TextInput("cancel native request")))
            await wait_until(
                lambda: any(
                    "cancel native request" in value for value in boundary.observed_contexts
                )
            )
            await within(10, task.request_cancellation())
            outcome = await within(10, task.await_outcome())
            assert isinstance(outcome, Cancelled)
            assert task.state is TaskState.CANCELLED
            assert not task.pending_interactions
            boundary.release()
            next_task = await session.start_task(
                TaskRequest(TextInput("continue after confirmed cancellation"))
            )
            assert isinstance(await within(60, next_task.await_outcome()), Completed)


class RuntimeProfileConformanceTests(RuntimeConformanceTests):
    async def test_unsupported_task_requirements_rejected_before_model_call(self) -> None:
        async with managed(self.boundary()) as boundary, self.harness(boundary) as harness:
            session = await harness.create_session(self.spec())
            before = len(boundary.observed_contexts)
            request = TaskRequest(
                TextInput("structured"),
                TaskRequirements(StructuredOutputRequirement('{"type":"object"}')),
            )
            assert session.validate(request).status is CompatibilityStatus.INCOMPATIBLE
            with pytest.raises(IncompatibleRequirementError):
                await session.start_task(request)
            assert len(boundary.observed_contexts) == before

    async def test_semantic_and_diagnostic_observers_finish_and_late_terminal_is_retained(
        self,
    ) -> None:
        async with managed(self.boundary()) as boundary, self.harness(boundary) as harness:
            assert isinstance(harness.support[Capability.DIAGNOSTICS], Supported)
            boundary.hold()
            configured = replace(
                self.spec(),
                requirements=SessionRequirements(diagnostics=DiagnosticsRequirement.REQUIRED),
            )
            task = await (await harness.create_session(configured)).start_task(
                TaskRequest(TextInput("observe actual native work"))
            )
            assert isinstance(task, TaskDiagnostics)
            semantic = asyncio.create_task(collect(task.events()))
            diagnostic = asyncio.create_task(collect(task.diagnostics()))
            await asyncio.sleep(0)
            boundary.release()
            outcome = await within(60, task.await_outcome())
            assert isinstance(outcome, Completed)
            events = await within(2, semantic)
            assert sum(isinstance(event, TerminalEvent) for event in events) == 1
            assert isinstance(events[-1], TaskCompleted)
            assert any(
                isinstance(event, ProviderDiagnostic) for event in await within(2, diagnostic)
            )
            late = await within(2, collect(task.events()))
            assert isinstance(late[-1], TaskCompleted)
            await task.request_cancellation()
            assert await task.await_outcome() == outcome


class RuntimePersistenceConformanceTests(RuntimeProfileConformanceTests):
    supports_changed_instructions_on_reopen = True

    async def test_persistent_reopen_preserves_context_and_desired_instructions(self) -> None:
        async with managed(self.boundary()) as boundary:
            configured = replace(
                self.spec(),
                requirements=SessionRequirements(persistence=PersistenceRequirement()),
            )
            async with self.harness(boundary) as first_harness:
                session = await first_harness.create_session(configured)
                task = await session.start_task(TaskRequest(TextInput("durable marker-delta")))
                assert isinstance(await within(60, task.await_outcome()), Completed)
                reference = session.persistent_ref
                assert reference is not None
                await session.release()
            async with self.harness(boundary) as harness:
                assert isinstance(harness, PersistentSessions)
                before = len(boundary.observed_contexts)
                desired = replace(configured, instructions="AHP_REOPEN_INSTRUCTION")
                if not self.supports_changed_instructions_on_reopen:
                    with pytest.raises(IncompatibleRequirementError):
                        await harness.reopen_session(reference, desired)
                    assert len(boundary.observed_contexts) == before
                reopened = await harness.reopen_session(
                    reference,
                    desired if self.supports_changed_instructions_on_reopen else configured,
                )
                task = await reopened.start_task(TaskRequest(TextInput("after harness recreation")))
                assert isinstance(await within(60, task.await_outcome()), Completed)
                actual = " ".join(boundary.observed_contexts[before:])
                assert "marker-delta" in actual
                if self.supports_changed_instructions_on_reopen:
                    assert "AHP_REOPEN_INSTRUCTION" in actual
                with pytest.raises(ValueError):
                    await harness.reopen_session(
                        replace(reference, namespace=StorageNamespace("foreign")), configured
                    )
                unsupported = replace(
                    configured,
                    requirements=replace(
                        configured.requirements,
                        persistence=PersistenceRequirement(across_process_restart=True),
                    ),
                )
                with pytest.raises(IncompatibleRequirementError):
                    await harness.create_session(unsupported)
