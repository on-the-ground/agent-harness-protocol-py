"""Admission acknowledgement conformance scenarios."""

import asyncio

import pytest

from ..compatibility import (
    HarnessTransportError,
    InteractionResponseUnconfirmedError,
    SessionBlockedError,
    TaskStartUnconfirmedError,
    UnconfirmedResponse,
)
from ..events import InteractionResolved
from ..harness import AgentTask
from ..interaction import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
    Cleared,
    ClearReason,
)
from ..outcome import Completed
from ..requirements import TaskRequest, TextInput
from ._support import collect, managed, wait_until, within
from .fixtures import (
    AcceptanceFixture,
    FixtureResource,
    ResponseAcceptanceFixture,
    StartAcceptanceFixture,
)


def _text(value: str) -> TaskRequest:
    return TaskRequest(TextInput(value))


async def _pending_approval(task: AgentTask) -> ApprovalRequest:
    await wait_until(lambda: bool(task.pending_interactions))
    request = task.pending_interactions[0]
    assert len(task.pending_interactions) == 1
    assert isinstance(request, ApprovalRequest)
    return request


class AcceptedStartConformanceTests:
    """Subclass from a pytest ``Test*`` class and provide a real-boundary fixture."""

    def acceptance_fixture(self) -> FixtureResource[AcceptanceFixture]:
        raise NotImplementedError

    async def test_accepted_start_yields_a_usable_identity_while_real_work_is_held(self) -> None:
        async with managed(self.acceptance_fixture()) as fixture:
            fixture.observation.hold()
            session = await fixture.harness.create_session(fixture.spec)
            task = await within(60, session.start_task(_text("held-accepted-start")))
            assert task.session_id == session.id
            await wait_until(
                lambda: any(
                    "held-accepted-start" in value
                    for value in fixture.observation.observed_contexts
                )
            )
            assert not task.state.is_terminal
            fixture.observation.release()
            outcome = await within(60, task.await_outcome())
            assert isinstance(outcome, Completed)
            assert outcome.task_id == task.id


class StartAcceptanceConformanceTests(AcceptedStartConformanceTests):
    def acceptance_fixture(self) -> FixtureResource[StartAcceptanceFixture]:
        raise NotImplementedError

    async def test_confirmed_nondelivery_permits_retry_without_a_native_start(self) -> None:
        async with managed(self.acceptance_fixture()) as fixture:
            session = await fixture.harness.create_session(fixture.spec)
            fixture.start.reject_before_delivery("controlled nondelivery")
            with pytest.raises(HarnessTransportError):
                await session.start_task(_text("not-delivered"))
            assert fixture.start.observed_submissions() == 1
            assert fixture.start.observed_accepted_starts() == 0
            assert not fixture.observation.observed_contexts
            fixture.start.accept()
            outcome = await within(
                60, (await session.start_task(_text("safe-retry"))).await_outcome()
            )
            assert isinstance(outcome, Completed)
            assert fixture.start.observed_submissions() == 2
            assert fixture.start.observed_accepted_starts() == 1

    async def test_lost_acknowledgement_after_native_acceptance_blocks_resubmission(self) -> None:
        await self._lost_start(True)

    async def test_lost_acknowledgement_without_native_acceptance_blocks_resubmission(self) -> None:
        await self._lost_start(False)

    async def _lost_start(self, accepted: bool) -> None:
        async with managed(self.acceptance_fixture()) as fixture:
            fixture.observation.hold()
            session = await fixture.harness.create_session(fixture.spec)
            fixture.start.lose_acceptance_acknowledgement(accepted)
            with pytest.raises(TaskStartUnconfirmedError) as caught:
                await session.start_task(_text("uncertain-start"))
            reference = caught.value.reference
            assert reference.session_id == session.id
            assert reference.request_id and not reference.request_id.isspace()
            assert tuple(fixture.submitted_starts) == (reference,)
            assert fixture.start.observed_submissions() == 1
            assert fixture.start.observed_accepted_starts() == (1 if accepted else 0)
            with pytest.raises(SessionBlockedError):
                await session.start_task(_text("must-not-resubmit"))
            assert fixture.start.observed_submissions() == 1
            if accepted:
                await wait_until(
                    lambda: any(
                        "uncertain-start" in value
                        for value in fixture.observation.observed_contexts
                    )
                )
            else:
                assert not fixture.observation.observed_contexts
            fixture.observation.release()
            fixture.start.accept()
            independent = await fixture.harness.create_session(fixture.spec)
            assert isinstance(
                await within(
                    60,
                    (await independent.start_task(_text("independent-after-loss"))).await_outcome(),
                ),
                Completed,
            )
            with pytest.raises(SessionBlockedError):
                await session.start_task(_text("late-retry"))
            assert fixture.start.observed_submissions() == 2


class ResponseAcceptanceConformanceTests:
    def response_fixture(self) -> FixtureResource[ResponseAcceptanceFixture]:
        raise NotImplementedError

    _approval = ApprovalResponse(ApprovalDecision.APPROVE_ONCE)

    async def test_confirmed_response_nondelivery_leaves_native_request_retryable(self) -> None:
        async with managed(self.response_fixture()) as fixture:
            task = await (await fixture.harness.create_session(fixture.spec)).start_task(
                _text("guarded-effect")
            )
            request = await _pending_approval(task)
            assert fixture.effect_count() == 0
            fixture.response.reject_before_delivery("controlled nondelivery")
            with pytest.raises(HarnessTransportError):
                await task.respond(request.interaction_id, self._approval)
            assert any(
                item.interaction_id == request.interaction_id for item in task.pending_interactions
            )
            assert not fixture.response.observed_accepted_responses()
            assert fixture.effect_count() == 0
            fixture.response.accept()
            await task.respond(request.interaction_id, self._approval)
            fixture.observation.release()
            assert isinstance(await within(60, task.await_outcome()), Completed)
            assert fixture.effect_count() == 1
            assert tuple(fixture.response.observed_accepted_responses()) == (self._approval,)
            assert fixture.response.observed_submissions() == 2

    async def test_response_acceptance_loss_after_native_acceptance_prevents_duplicate_effects(
        self,
    ) -> None:
        await self._lost_response(True)

    async def test_response_acceptance_loss_before_native_acceptance_prevents_duplicate_submission(
        self,
    ) -> None:
        await self._lost_response(False)

    async def _lost_response(self, accepted: bool) -> None:
        async with managed(self.response_fixture()) as fixture:
            task = await (await fixture.harness.create_session(fixture.spec)).start_task(
                _text("guarded-effect")
            )
            events = asyncio.create_task(collect(task.events()))
            await asyncio.sleep(0)
            request = await _pending_approval(task)
            fixture.response.lose_acceptance_acknowledgement(accepted)
            with pytest.raises(InteractionResponseUnconfirmedError) as caught:
                await task.respond(request.interaction_id, self._approval)
            assert caught.value.reference == UnconfirmedResponse(task.id, request.interaction_id)
            assert all(
                item.interaction_id != request.interaction_id for item in task.pending_interactions
            )
            with pytest.raises(RuntimeError):
                await task.respond(request.interaction_id, self._approval)
            assert fixture.response.observed_submissions() == 1
            expected = (self._approval,) if accepted else ()
            assert tuple(fixture.response.observed_accepted_responses()) == expected
            if accepted:
                fixture.observation.release()
                assert isinstance(await within(60, task.await_outcome()), Completed)
                assert fixture.effect_count() == 1
            else:
                assert fixture.effect_count() == 0
                assert not task.state.is_terminal
                await task.request_cancellation()
                await within(60, task.await_outcome())
            observed = await within(5, events)
            resolutions = [
                event.resolution
                for event in observed
                if isinstance(event, InteractionResolved)
                and event.interaction_id == request.interaction_id
            ]
            assert resolutions == [Cleared(ClearReason.RESPONSE_UNCONFIRMED)]
