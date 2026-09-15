"""Usage, message identity, and inner-work accounting scenarios."""

import asyncio

from ..events import MessageCompleted, MessageDelta, ToolCallChanged, UsageChanged, WorkStatus
from ..outcome import AgentUsage, Cancelled, Completed, TextOutput
from ..requirements import TaskRequest, TextInput
from ._support import collect, managed, wait_until, within
from .fixtures import AcceptanceFixture, AccountingFixture, FixtureResource


class AccountingConformanceTests:
    def accounting_fixture(self) -> FixtureResource[AccountingFixture]:
        raise NotImplementedError

    async def test_task_accounting_resets_and_retains_the_last_snapshot(self) -> None:
        async with managed(self.accounting_fixture()) as fixture:
            session = await fixture.harness.create_session(fixture.spec)
            for index, expected in enumerate(fixture.measurements):
                fixture.observation.hold()
                task = await session.start_task(TaskRequest(TextInput(f"accounting-{index}")))
                observed_task = asyncio.create_task(collect(task.events()))
                await asyncio.sleep(0)
                fixture.observation.release()
                outcome = await within(60, task.await_outcome())
                assert isinstance(outcome, Completed)
                self._assert_measured(expected, outcome.usage)
                expected_session = fixture.session_measurements[index]
                if expected_session is None:
                    assert outcome.session_usage is None
                else:
                    assert outcome.session_usage is not None
                    self._assert_measured(expected_session, outcome.session_usage)
                events = await within(5, observed_task)
                usage_events = [event for event in events if isinstance(event, UsageChanged)]
                assert usage_events
                assert usage_events[-1].task == outcome.usage
                assert usage_events[-1].session == outcome.session_usage
                assert all(event.task_id == task.id for event in events)
                messages = [event for event in events if isinstance(event, MessageCompleted)]
                assert messages
                assert all(message.message_id.value.strip() for message in messages)
                assert {message.role for message in messages} == set(
                    fixture.completed_message_roles
                )
                assert len(messages) == len({message.message_id for message in messages})
                deltas: dict[object, list[MessageDelta]] = {}
                for event in events:
                    if isinstance(event, MessageDelta):
                        deltas.setdefault(event.message_id, []).append(event)
                for message_id, fragments in deltas.items():
                    matching = [message for message in messages if message.message_id == message_id]
                    assert len(matching) == 1
                    assert "".join(fragment.text for fragment in fragments) == matching[0].text
                assert isinstance(outcome.output, TextOutput)
                assert outcome.output.text == "measured-result"

    async def test_cancellation_before_measurement_preserves_unknown_instead_of_zero(self) -> None:
        async with managed(self.accounting_fixture()) as fixture:
            fixture.observation.hold()
            task = await (await fixture.harness.create_session(fixture.spec)).start_task(
                TaskRequest(TextInput("unmeasured-cancellation"))
            )
            await wait_until(
                lambda: any(
                    "unmeasured-cancellation" in value
                    for value in fixture.observation.observed_contexts
                )
            )
            await task.request_cancellation()
            outcome = await within(10, task.await_outcome())
            assert isinstance(outcome, Cancelled)
            assert outcome.usage == AgentUsage.UNKNOWN
            assert outcome.session_usage is None

    @staticmethod
    def _assert_measured(expected: AgentUsage, actual: AgentUsage) -> None:
        assert actual.input_tokens == expected.input_tokens
        assert actual.output_tokens == expected.output_tokens
        assert actual.total_tokens == expected.total_tokens


class AccountingSequenceConformanceTests:
    def sequence_fixture(self) -> FixtureResource[AcceptanceFixture]:
        raise NotImplementedError

    async def test_unmeasured_segment_keeps_task_total_unknown_after_measured_zero(self) -> None:
        async with managed(self.sequence_fixture()) as fixture:
            fixture.observation.hold()
            task = await (await fixture.harness.create_session(fixture.spec)).start_task(
                TaskRequest(TextInput("measure-several-native-calls"))
            )
            events_task = asyncio.create_task(collect(task.events()))
            await asyncio.sleep(0)
            fixture.observation.release()
            outcome = await within(60, task.await_outcome())
            assert isinstance(outcome, Completed)
            assert outcome.usage.input_tokens == 17
            assert outcome.usage.output_tokens is None
            assert outcome.usage.total_tokens is None
            events = await within(5, events_task)
            work = [event for event in events if isinstance(event, ToolCallChanged)]
            started = [event for event in work if event.status is WorkStatus.STARTED]
            completed = [event for event in work if event.status is WorkStatus.COMPLETED]
            assert len(started) == 2
            assert len({event.work_id for event in started}) == 2
            assert {event.work_id for event in started} == {event.work_id for event in completed}
            assert len(completed) == 2
            assert all(event.task_id == task.id for event in events)
            snapshots = [event.task for event in events if isinstance(event, UsageChanged)]
            assert [value.input_tokens for value in snapshots] == [10, 17, 17]
            assert [value.output_tokens for value in snapshots] == [5, None, None]
            assert snapshots[-1] == outcome.usage
