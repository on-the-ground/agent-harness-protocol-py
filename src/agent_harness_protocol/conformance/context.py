"""Persistent-context identity, aliasing, and blocking scenarios."""

import pytest

from ..compatibility import SessionBlockedError, TaskStartUnconfirmedError
from ..harness import AgentSession, PersistentSessions
from ..outcome import Cancelled, Completed
from ..requirements import TaskRequest, TextInput
from ._support import managed, wait_until, within
from .fixtures import ContextFixture, FixtureResource


class ContextConformanceTests:
    def context_fixture(self) -> FixtureResource[ContextFixture]:
        raise NotImplementedError

    @staticmethod
    def _text(value: str) -> TaskRequest:
        return TaskRequest(TextInput(value))

    async def _completed(self, session: AgentSession, value: str) -> Completed:
        outcome = await within(60, (await session.start_task(self._text(value))).await_outcome())
        assert isinstance(outcome, Completed)
        return outcome

    async def test_reopened_aliases_share_exclusion_and_released_handle_boundaries(self) -> None:
        async with managed(self.context_fixture()) as fixture:
            original = await fixture.harness.create_session(fixture.spec)
            await self._completed(original, "persist-before-alias")
            reference = original.persistent_ref
            assert reference is not None
            assert isinstance(fixture.harness, PersistentSessions)
            alias = await fixture.harness.reopen_session(reference, fixture.spec)
            assert original.id == alias.id
            fixture.observation.hold()
            task = await alias.start_task(self._text("active-via-alias"))
            before = fixture.start.observed_submissions()
            with pytest.raises(RuntimeError):
                await original.start_task(self._text("overlap-via-original"))
            with pytest.raises(RuntimeError):
                await fixture.harness.reopen_session(reference, fixture.spec)
            assert fixture.start.observed_submissions() == before
            fixture.observation.release()
            assert isinstance(await within(60, task.await_outcome()), Completed)
            await original.release()
            with pytest.raises(SessionBlockedError):
                await original.start_task(self._text("released-handle"))
            await self._completed(alias, "alias-still-usable")
            assert "persist-before-alias" in fixture.observation.observed_contexts[-1]

    async def test_unconfirmed_start_blocks_aliases_and_recreation_but_not_fresh_context(
        self,
    ) -> None:
        async with managed(self.context_fixture()) as fixture:
            original = await fixture.harness.create_session(fixture.spec)
            await self._completed(original, "persist-before-loss")
            reference = original.persistent_ref
            assert reference is not None
            assert isinstance(fixture.harness, PersistentSessions)
            alias = await fixture.harness.reopen_session(reference, fixture.spec)
            fixture.start.lose_acceptance_acknowledgement(False)
            with pytest.raises(TaskStartUnconfirmedError):
                await alias.start_task(self._text("lost-start"))
            before = fixture.start.observed_submissions()
            with pytest.raises(SessionBlockedError):
                await original.start_task(self._text("alias-retry"))
            with pytest.raises(SessionBlockedError):
                await fixture.harness.reopen_session(reference, fixture.spec)
            await original.release()
            await alias.release()
            assert fixture.start.observed_submissions() == before
            await fixture.harness.aclose()
            async with fixture.recreate_harness() as recreated:
                assert isinstance(recreated, PersistentSessions)
                with pytest.raises(SessionBlockedError):
                    await recreated.reopen_session(reference, fixture.spec)
                await self._completed(
                    await recreated.create_session(fixture.spec), "independent-after-recreation"
                )
                assert "persist-before-loss" not in fixture.observation.observed_contexts[-1]

    async def test_releasing_one_active_context_leaves_another_progressing(self) -> None:
        async with managed(self.context_fixture()) as fixture:
            first = await fixture.harness.create_session(fixture.spec)
            second = await fixture.harness.create_session(fixture.spec)
            fixture.observation.hold()
            task_a = await first.start_task(self._text("release-this-context"))
            task_b = await second.start_task(self._text("retain-this-context"))
            await wait_until(
                lambda: any(
                    "retain-this-context" in value
                    for value in fixture.observation.observed_contexts
                )
            )
            await first.release()
            assert isinstance(await within(5, task_a.await_outcome()), Cancelled)
            assert not task_b.state.is_terminal
            fixture.observation.release()
            assert isinstance(await within(60, task_b.await_outcome()), Completed)
            assert all(
                "release-this-context" not in value
                for value in fixture.observation.observed_contexts
                if "retain-this-context" in value
            )
