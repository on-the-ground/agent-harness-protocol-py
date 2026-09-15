"""Approval-scope conformance scenarios."""

import pytest

from ..harness import AgentTask
from ..interaction import ApprovalDecision, ApprovalRequest, ApprovalResponse
from ..outcome import Completed
from ..requirements import TaskRequest, TextInput
from ._support import managed, wait_until, within
from .fixtures import FixtureResource, RepeatedApprovalFixture


class ApprovalScopeConformanceTests:
    def approval_fixture(self) -> FixtureResource[RepeatedApprovalFixture]:
        raise NotImplementedError

    @staticmethod
    async def _pending(task: AgentTask) -> ApprovalRequest:
        await wait_until(lambda: bool(task.pending_interactions))
        assert len(task.pending_interactions) == 1
        request = task.pending_interactions[0]
        assert isinstance(request, ApprovalRequest)
        return request

    @staticmethod
    def _request() -> TaskRequest:
        return TaskRequest(TextInput("guarded-effect"))

    async def test_opaque_session_approval_is_not_exposed_as_an_unbounded_grant(self) -> None:
        async with managed(self.approval_fixture()) as fixture:
            task = await (await fixture.harness.create_session(fixture.spec)).start_task(
                self._request()
            )
            approval = await self._pending(task)
            assert approval.session_grant is None
            assert ApprovalDecision.APPROVE_FOR_SESSION not in approval.available_decisions
            with pytest.raises(ValueError):
                await task.respond(
                    approval.interaction_id,
                    ApprovalResponse(ApprovalDecision.APPROVE_FOR_SESSION),
                )
            assert fixture.response.observed_submissions() == 0
            assert fixture.effect_count() == 0
            await task.respond(approval.interaction_id, ApprovalResponse(ApprovalDecision.DECLINE))
            fixture.observation.release()
            assert isinstance(await within(60, task.await_outcome()), Completed)
            assert fixture.effect_count() == 0

    async def test_one_shot_approval_does_not_authorize_next_task_in_same_context(self) -> None:
        await self._repeat_effect(False)

    async def test_one_shot_approval_does_not_leak_into_another_session(self) -> None:
        await self._repeat_effect(True)

    async def _repeat_effect(self, new_session: bool) -> None:
        async with managed(self.approval_fixture()) as fixture:
            session = await fixture.harness.create_session(fixture.spec)
            first = await session.start_task(self._request())
            await first.respond(
                (await self._pending(first)).interaction_id,
                ApprovalResponse(ApprovalDecision.APPROVE_ONCE),
            )
            fixture.observation.release()
            assert isinstance(await within(60, first.await_outcome()), Completed)
            assert fixture.effect_count() == 1
            fixture.prepare_next_effect()
            target_session = (
                await fixture.harness.create_session(fixture.spec) if new_session else session
            )
            second = await target_session.start_task(self._request())
            approval = await self._pending(second)
            assert first.id != second.id
            assert fixture.effect_count() == 1
            await second.respond(
                approval.interaction_id, ApprovalResponse(ApprovalDecision.DECLINE)
            )
            fixture.observation.release()
            assert isinstance(await within(60, second.await_outcome()), Completed)
            assert fixture.effect_count() == 1
