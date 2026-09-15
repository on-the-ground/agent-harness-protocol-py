"""Total cleanup-budget and cancellation-resilience scenarios."""

import asyncio
from time import monotonic

import pytest

from ..compatibility import SessionBlockedError
from ..events import TaskState, TerminalEvent
from ..outcome import Cancelled, Completed, Unresolved
from ..requirements import TaskRequest, TextInput
from ._support import collect, managed, wait_until, within
from .fixtures import AcceptanceFixture, FixtureResource


class CleanupBudgetConformanceTests:
    def cleanup_fixture(self) -> FixtureResource[AcceptanceFixture]:
        raise NotImplementedError

    @staticmethod
    def _request(value: str) -> TaskRequest:
        return TaskRequest(TextInput(value))

    async def test_close_settles_active_sessions_within_one_total_budget(self) -> None:
        async with managed(self.cleanup_fixture()) as fixture:
            fixture.observation.hold()
            sessions = [await fixture.harness.create_session(fixture.spec) for _ in range(4)]
            tasks = [
                await session.start_task(self._request(f"cleanup-resource-{index}"))
                for index, session in enumerate(sessions)
            ]
            await wait_until(
                lambda: all(
                    any(
                        f"cleanup-resource-{index}" in value
                        for value in fixture.observation.observed_contexts
                    )
                    for index in range(len(tasks))
                )
            )
            started = monotonic()
            await fixture.harness.aclose()
            elapsed = monotonic() - started
            assert elapsed <= fixture.harness.cleanup_budget.total.total_seconds() + 0.5
            outcomes = [await within(2, task.await_outcome()) for task in tasks]
            assert all(isinstance(outcome, (Cancelled, Unresolved)) for outcome in outcomes)
            for task in tasks:
                assert task.state.is_terminal
                assert not task.pending_interactions
                late = await within(2, collect(task.events()))
                assert sum(isinstance(event, TerminalEvent) for event in late) == 1
                assert isinstance(late[-1], TerminalEvent)
            for session in sessions:
                with pytest.raises(SessionBlockedError):
                    await session.start_task(self._request("closed-start"))
            fixture.observation.release()
            await fixture.harness.aclose()
            assert outcomes == [await task.await_outcome() for task in tasks]

    async def test_cancelling_release_caller_still_settles_work_and_closes_handle(self) -> None:
        async with managed(self.cleanup_fixture()) as fixture:
            fixture.observation.hold()
            session = await fixture.harness.create_session(fixture.spec)
            task = await session.start_task(self._request("cancel-release-caller"))
            await wait_until(
                lambda: any(
                    "cancel-release-caller" in value
                    for value in fixture.observation.observed_contexts
                )
            )
            started = monotonic()
            release = asyncio.create_task(session.release())
            await asyncio.sleep(0)
            release.cancel()
            try:
                await release
            except asyncio.CancelledError:
                pass
            elapsed = monotonic() - started
            assert elapsed <= fixture.harness.cleanup_budget.total.total_seconds() + 0.5
            outcome = await within(2, task.await_outcome())
            assert isinstance(outcome, (Cancelled, Unresolved))
            with pytest.raises(SessionBlockedError):
                await session.start_task(self._request("after-release"))
            await session.release()
            assert await task.await_outcome() == outcome

    async def test_cleanup_preserves_completed_outcomes_alongside_active_work(self) -> None:
        async with managed(self.cleanup_fixture()) as fixture:
            done = await (await fixture.harness.create_session(fixture.spec)).start_task(
                self._request("completed-before-close")
            )
            original = await within(60, done.await_outcome())
            assert isinstance(original, Completed)
            fixture.observation.hold()
            held = await (await fixture.harness.create_session(fixture.spec)).start_task(
                self._request("active-during-close")
            )
            await wait_until(
                lambda: any(
                    "active-during-close" in value
                    for value in fixture.observation.observed_contexts
                )
            )
            await fixture.harness.aclose()
            await within(2, held.await_outcome())
            assert await done.await_outcome() == original
            assert done.state is TaskState.COMPLETED
