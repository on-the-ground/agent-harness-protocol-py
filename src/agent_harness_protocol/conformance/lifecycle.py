"""Minimum task lifecycle conformance scenarios."""

import asyncio

from ..events import ObservationGap, TaskEvent, TaskState, TerminalEvent
from ..outcome import Failed, FailureKind, TextOutput
from ..requirements import SessionSpec, TaskRequest, TextInput
from ._support import cancel_waiter, collect, managed, within
from .fixtures import FixtureResource, LifecycleFixture, TextObservation


class LifecycleConformanceTests:
    def lifecycle_fixture(self) -> FixtureResource[LifecycleFixture]:
        raise NotImplementedError

    def compatible_spec(self) -> SessionSpec:
        return SessionSpec()

    async def _task(self, fixture: LifecycleFixture):
        session = await fixture.harness.create_session(self.compatible_spec())
        return await session.start_task(TaskRequest(TextInput("x")))

    async def test_state_is_terminal_before_await_outcome_returns(self) -> None:
        async with managed(self.lifecycle_fixture()) as fixture:
            task = await self._task(fixture)
            await fixture.control(task).report_running()
            await fixture.control(task).report_completion(TextObservation("done"))
            await within(5, task.await_outcome())
            assert task.state.is_terminal

    async def test_completes_without_an_event_collector(self) -> None:
        async with managed(self.lifecycle_fixture()) as fixture:
            task = await self._task(fixture)
            control = fixture.control(task)
            await control.report_running()
            for index in range(500):
                await control.report_message_delta("message", f"chunk{index} ")
            await control.report_completion(TextObservation("done"))
            await within(5, task.await_outcome())

    async def test_slow_collector_does_not_block_lifecycle(self) -> None:
        async with managed(self.lifecycle_fixture()) as fixture:
            task = await self._task(fixture)
            gate = asyncio.Event()

            async def stalled() -> None:
                async for _ in task.events():
                    await gate.wait()

            collector = asyncio.create_task(stalled())
            await asyncio.sleep(0)
            control = fixture.control(task)
            await control.report_running()
            for index in range(2_000):
                await control.report_message_delta("message", f"chunk{index} ")
            await control.report_completion(TextObservation("done"))
            await within(5, task.await_outcome())
            assert task.state is TaskState.COMPLETED
            gate.set()
            await cancel_waiter(collector)

    async def test_failure_is_reported_through_state_and_await_outcome(self) -> None:
        async with managed(self.lifecycle_fixture()) as fixture:
            task = await self._task(fixture)
            await fixture.control(task).report_running()
            await fixture.control(task).report_failure("boom")
            outcome = await within(5, task.await_outcome())
            assert isinstance(outcome, Failed)
            assert outcome.message == "boom"
            assert outcome.kind is FailureKind.UNKNOWN
            assert task.state is TaskState.FAILED

    async def test_completion_wins_the_race_against_cancel(self) -> None:
        async with managed(self.lifecycle_fixture()) as fixture:
            task = await self._task(fixture)
            await fixture.control(task).report_running()
            await task.request_cancellation()
            await fixture.control(task).report_completion(TextObservation("done"))
            outcome = await within(5, task.await_outcome())
            assert isinstance(outcome.output, TextOutput)
            assert outcome.output.text == "done"
            assert task.state is TaskState.COMPLETED

    async def test_terminal_is_exactly_once_and_last(self) -> None:
        async with managed(self.lifecycle_fixture()) as fixture:
            task = await self._task(fixture)
            seen_task = asyncio.create_task(collect(task.events()))
            await asyncio.sleep(0)
            await fixture.control(task).report_running()
            await fixture.control(task).report_completion(TextObservation("done"))
            await fixture.control(task).report_failure("late")
            await within(5, task.await_outcome())
            seen = await within(5, seen_task)
            terminal_indexes = [
                index for index, event in enumerate(seen) if isinstance(event, TerminalEvent)
            ]
            assert terminal_indexes == [len(seen) - 1]
            assert task.state is TaskState.COMPLETED

    async def test_different_sessions_execute_concurrently(self) -> None:
        async with managed(self.lifecycle_fixture()) as fixture:
            first_session, second_session = await asyncio.gather(
                fixture.harness.create_session(self.compatible_spec()),
                fixture.harness.create_session(self.compatible_spec()),
            )
            first, second = await asyncio.gather(
                first_session.start_task(TaskRequest(TextInput("a"))),
                second_session.start_task(TaskRequest(TextInput("b"))),
            )
            await fixture.control(first).report_running()
            await fixture.control(second).report_running()
            await fixture.control(first).report_completion(TextObservation("A"))
            await fixture.control(second).report_completion(TextObservation("B"))
            first_outcome, second_outcome = await asyncio.gather(
                within(5, first.await_outcome()), within(5, second.await_outcome())
            )
            assert isinstance(first_outcome.output, TextOutput)
            assert first_outcome.output.text == "A"
            assert isinstance(second_outcome.output, TextOutput)
            assert second_outcome.output.text == "B"

    async def test_harness_close_without_native_termination_evidence_settles_unresolved(
        self,
    ) -> None:
        from ..outcome import Unresolved

        async with managed(self.lifecycle_fixture()) as fixture:
            task = await self._task(fixture)
            await fixture.control(task).report_running()
            await fixture.harness.aclose()
            assert isinstance(await within(5, task.await_outcome()), Unresolved)
            assert task.state is TaskState.UNRESOLVED

    async def test_overflow_is_explicit_and_terminal_survives(self) -> None:
        async with managed(self.lifecycle_fixture()) as fixture:
            task = await self._task(fixture)
            gate = asyncio.Event()
            seen: list[TaskEvent] = []

            async def slow() -> None:
                async for event in task.events():
                    seen.append(event)
                    if len(seen) == 1:
                        await gate.wait()

            collector = asyncio.create_task(slow())
            await asyncio.sleep(0)
            control = fixture.control(task)
            await control.report_running()
            for index in range(5_000):
                await control.report_message_delta("message", f"chunk{index} ")
            await control.report_completion(TextObservation("done"))
            await within(5, task.await_outcome())
            gate.set()
            await within(5, collector)
            assert any(isinstance(event, ObservationGap) for event in seen)
            assert isinstance(seen[-1], TerminalEvent)
            assert sum(isinstance(event, TerminalEvent) for event in seen) == 1
