"""Caller interaction validation and race conformance scenarios."""

import asyncio

import pytest

from ..compatibility import InteractionResponseUnconfirmedError
from ..events import EffectChanged, InteractionResolved, TerminalEvent, WorkStatus
from ..interaction import (
    AnswerResponse,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
    Cleared,
    ClearReason,
)
from ..outcome import Cancelled, Completed
from ..requirements import TaskRequest, TextInput
from ._support import collect, managed, wait_until, within
from .fixtures import FixtureResource, InteractionRaceFixture


class InteractionConformanceTests:
    def interaction_fixture(self) -> FixtureResource[InteractionRaceFixture]:
        raise NotImplementedError

    _approve = ApprovalResponse(ApprovalDecision.APPROVE_ONCE)

    async def _open(self, fixture: InteractionRaceFixture):
        session = await fixture.harness.create_session(fixture.spec)
        task = await session.start_task(TaskRequest(TextInput("guarded-effect")))
        await wait_until(lambda: bool(task.pending_interactions))
        assert len(task.pending_interactions) == 1
        request = task.pending_interactions[0]
        assert isinstance(request, ApprovalRequest)
        return task, request

    async def test_wrong_response_type_and_unavailable_decision_never_reach_native_delivery(
        self,
    ) -> None:
        async with managed(self.interaction_fixture()) as fixture:
            task, request = await self._open(fixture)
            with pytest.raises(ValueError):
                await task.respond(request.interaction_id, AnswerResponse("yes"))
            unavailable = next(
                (
                    decision
                    for decision in ApprovalDecision
                    if decision not in request.available_decisions
                ),
                None,
            )
            if unavailable is not None:
                with pytest.raises(ValueError):
                    await task.respond(request.interaction_id, ApprovalResponse(unavailable))
            assert fixture.response.observed_submissions() == 0
            assert fixture.effect_count() == 0
            await task.respond(request.interaction_id, self._approve)
            fixture.observation.release()
            assert isinstance(await within(60, task.await_outcome()), Completed)
            assert fixture.effect_count() == 1

    async def test_cancelled_response_caller_retains_the_single_submission_gate(self) -> None:
        async with managed(self.interaction_fixture()) as fixture:
            task, request = await self._open(fixture)
            fixture.hold_response()
            first = asyncio.create_task(task.respond(request.interaction_id, self._approve))
            await within(5, fixture.await_response_submission())
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            with pytest.raises(RuntimeError):
                await task.respond(request.interaction_id, self._approve)
            assert fixture.response.observed_submissions() == 1
            fixture.release_response()
            fixture.observation.release()
            assert isinstance(await within(60, task.await_outcome()), Completed)
            assert fixture.effect_count() == 1

    async def test_cancellation_clears_pending_before_returning_and_prevents_unapproved_effect(
        self,
    ) -> None:
        async with managed(self.interaction_fixture()) as fixture:
            task, request = await self._open(fixture)
            seen_task = asyncio.create_task(collect(task.events()))
            await asyncio.sleep(0)
            await task.request_cancellation()
            assert not task.pending_interactions
            with pytest.raises(RuntimeError):
                await task.respond(request.interaction_id, self._approve)
            assert isinstance(await within(60, task.await_outcome()), Cancelled)
            fixture.observation.release()
            assert fixture.effect_count() == 0
            assert fixture.response.observed_submissions() == 0
            seen = await within(5, seen_task)
            resolutions = [
                event.resolution
                for event in seen
                if isinstance(event, InteractionResolved)
                and event.interaction_id == request.interaction_id
            ]
            assert resolutions == [Cleared(ClearReason.CANCELLATION_REQUESTED)]
            assert isinstance(seen[-1], TerminalEvent)

    async def test_terminal_while_response_is_in_flight_stays_immutable_after_late_delivery(
        self,
    ) -> None:
        async with managed(self.interaction_fixture()) as fixture:
            task, request = await self._open(fixture)
            seen_task = asyncio.create_task(collect(task.events()))
            await asyncio.sleep(0)
            fixture.hold_response()
            response = asyncio.create_task(task.respond(request.interaction_id, self._approve))
            await within(5, fixture.await_response_submission())
            await task.request_cancellation()
            outcome = await within(60, task.await_outcome())
            assert isinstance(outcome, Cancelled)
            fixture.release_response()
            with pytest.raises(InteractionResponseUnconfirmedError):
                await within(10, response)
            fixture.observation.release()
            assert await task.await_outcome() == outcome
            assert not task.pending_interactions
            assert fixture.effect_count() == 0
            seen = await within(5, seen_task)
            assert sum(isinstance(event, TerminalEvent) for event in seen) == 1
            assert (
                sum(
                    isinstance(event, InteractionResolved)
                    and event.interaction_id == request.interaction_id
                    for event in seen
                )
                == 1
            )
            assert isinstance(seen[-1], TerminalEvent)

    async def test_completed_approval_rejects_late_duplicate_without_repeating_effect(self) -> None:
        async with managed(self.interaction_fixture()) as fixture:
            task, request = await self._open(fixture)
            seen_task = asyncio.create_task(collect(task.events()))
            await asyncio.sleep(0)
            await task.respond(request.interaction_id, self._approve)
            fixture.observation.release()
            outcome = await within(60, task.await_outcome())
            assert isinstance(outcome, Completed)
            with pytest.raises(RuntimeError):
                await task.respond(request.interaction_id, self._approve)
            await task.request_cancellation()
            assert await task.await_outcome() == outcome
            assert fixture.response.observed_submissions() == 1
            assert fixture.effect_count() == 1
            effects = [
                event
                for event in await within(5, seen_task)
                if isinstance(event, EffectChanged) and event.status is WorkStatus.COMPLETED
            ]
            assert len(effects) == 1
            assert request.work_id is not None
            assert effects[0].work_id == request.work_id
