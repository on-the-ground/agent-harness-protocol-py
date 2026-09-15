"""Outcome and partial-output conformance scenarios."""

import asyncio

from ..events import MessageCompleted, MessageDelta, TaskState, UsageChanged
from ..outcome import AgentUsage, TextOutput
from ..requirements import TaskRequest, TextInput
from ._support import managed, within
from .fixtures import FixtureResource, OutcomeFixture, OutputCase


class OutcomeConformanceTests:
    missing_response_state = TaskState.COMPLETED
    empty_response_state = TaskState.COMPLETED
    empty_text_reaches_adapter = True

    def outcome_fixture(self, output_case: OutputCase) -> FixtureResource[OutcomeFixture]:
        raise NotImplementedError

    async def test_completed_runtime_preserves_captured_output(self) -> None:
        await self._partial(TaskState.COMPLETED)

    async def test_failed_runtime_preserves_captured_partial_output(self) -> None:
        await self._partial(TaskState.FAILED)

    async def test_cancelled_runtime_preserves_captured_partial_output(self) -> None:
        await self._partial(TaskState.CANCELLED)

    async def test_unresolved_runtime_preserves_captured_partial_output(self) -> None:
        await self._partial(TaskState.UNRESOLVED)

    async def _partial(self, expected: TaskState) -> None:
        async with managed(self.outcome_fixture(OutputCase.PARTIAL)) as fixture:
            session = await fixture.harness.create_session(fixture.spec)
            task = await session.start_task(TaskRequest(TextInput("partial-output-evidence")))
            partial_seen = asyncio.Event()
            latest_usage = AgentUsage.UNKNOWN

            async def observe() -> None:
                nonlocal latest_usage
                async for event in task.events():
                    if isinstance(event, UsageChanged):
                        latest_usage = event.task
                    text = event.text if isinstance(event, (MessageDelta, MessageCompleted)) else ""
                    if "retained-partial" in text:
                        partial_seen.set()

            events = asyncio.create_task(observe())
            await asyncio.sleep(0)
            fixture.begin_model()
            await within(60, partial_seen.wait())
            if expected is TaskState.COMPLETED:
                fixture.finish_model()
            elif expected is TaskState.FAILED:
                fixture.fail_model()
            elif expected is TaskState.CANCELLED:
                await task.request_cancellation()
            elif expected is TaskState.UNRESOLVED:
                await fixture.lose_observation(session)
            outcome = await within(60, task.await_outcome())
            assert task.state is expected
            assert isinstance(outcome.output, TextOutput)
            assert outcome.output.text == "retained-partial"
            if expected is not TaskState.COMPLETED:
                assert not outcome.output.complete
            assert not task.pending_interactions
            await within(5, events)
            assert outcome.usage == latest_usage
            known_partial_usage = getattr(fixture, "known_partial_usage", None)
            if known_partial_usage is not None:
                assert outcome.usage == known_partial_usage
            fixture.finish_model()
            fixture.begin_model()
            assert await task.await_outcome() == outcome

    async def test_response_without_content_preserves_termination_and_output_absence(self) -> None:
        await self._no_partial(OutputCase.MISSING)

    async def test_empty_model_response_preserves_output_presence(self) -> None:
        await self._no_partial(OutputCase.EMPTY)

    async def _no_partial(self, mode: OutputCase) -> None:
        async with managed(self.outcome_fixture(mode)) as fixture:
            fixture.finish_model()
            fixture.begin_model()
            task = await (await fixture.harness.create_session(fixture.spec)).start_task(
                TaskRequest(TextInput("output-presence"))
            )
            outcome = await within(60, task.await_outcome())
            expected = (
                self.missing_response_state
                if mode is OutputCase.MISSING
                else self.empty_response_state
            )
            assert task.state is expected
            if mode is OutputCase.MISSING or not self.empty_text_reaches_adapter:
                assert outcome.output is None
            else:
                assert isinstance(outcome.output, TextOutput)
                assert outcome.output.text == ""
