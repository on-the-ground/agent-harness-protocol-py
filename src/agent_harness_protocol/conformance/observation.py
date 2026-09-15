"""Independent semantic and diagnostic load scenarios."""

import asyncio

from ..diagnostics import DiagnosticEvent, DiagnosticGap, TaskDiagnostics
from ..events import ObservationGap, TaskEvent, TaskStarted, TaskState, TerminalEvent
from ..outcome import Completed, TextOutput
from ..requirements import TaskRequest, TextInput
from ._support import collect, managed, wait_until, within
from .fixtures import AcceptanceFixture, FixtureResource


class ObservationLoadConformanceTests:
    def load_fixture(self) -> FixtureResource[AcceptanceFixture]:
        raise NotImplementedError

    async def test_stalled_readers_have_independent_counted_gaps_and_terminal(self) -> None:
        async with managed(self.load_fixture()) as fixture:
            fixture.observation.hold()
            task = await (await fixture.harness.create_session(fixture.spec)).start_task(
                TaskRequest(TextInput("native-observation-load"))
            )
            await wait_until(lambda: task.state is TaskState.RUNNING)
            assert isinstance(task, TaskDiagnostics)
            release = asyncio.Event()
            semantic: list[TaskEvent] = []
            diagnostic: list[DiagnosticEvent] = []

            async def slow_semantic() -> None:
                async for event in task.events():
                    semantic.append(event)
                    if len(semantic) == 1:
                        await release.wait()

            async def slow_diagnostic() -> None:
                async for event in task.diagnostics():
                    diagnostic.append(event)
                    if len(diagnostic) == 1:
                        await release.wait()

            fast_semantic = asyncio.create_task(collect(task.events()))
            fast_diagnostic = asyncio.create_task(collect(task.diagnostics()))
            slow_semantic_task = asyncio.create_task(slow_semantic())
            slow_diagnostic_task = asyncio.create_task(slow_diagnostic())
            await asyncio.sleep(0)
            fixture.observation.release()
            outcome = await within(60, task.await_outcome())
            assert isinstance(outcome, Completed)
            assert isinstance(outcome.output, TextOutput)
            assert outcome.output.text.endswith("load-complete")
            fast_events = await within(5, fast_semantic)
            fast_diagnostics = await within(5, fast_diagnostic)
            release.set()
            await within(5, asyncio.gather(slow_semantic_task, slow_diagnostic_task))
            assert any(isinstance(event, ObservationGap) for event in semantic)
            assert any(isinstance(event, DiagnosticGap) for event in diagnostic)

            def semantic_count(values: list[TaskEvent]) -> int:
                return sum(
                    event.dropped_events
                    if isinstance(event, ObservationGap)
                    else 0
                    if isinstance(event, TaskStarted)
                    else 1
                    for event in values
                )

            def diagnostic_count(values: list[DiagnosticEvent]) -> int:
                return sum(
                    event.dropped_records if isinstance(event, DiagnosticGap) else 1
                    for event in values
                )

            assert semantic_count(fast_events) == semantic_count(semantic)
            assert diagnostic_count(fast_diagnostics) == diagnostic_count(diagnostic)
            assert semantic_count(semantic) > 256
            assert diagnostic_count(diagnostic) > 256
            assert sum(isinstance(event, TerminalEvent) for event in semantic) == 1
            assert isinstance(semantic[-1], TerminalEvent)
            assert await task.await_outcome() == outcome
