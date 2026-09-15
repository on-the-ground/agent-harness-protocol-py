"""Disposition is reported only as far as the adapter owns or was told the fact."""

from __future__ import annotations

import pytest
from agent_harness_protocol import (
    Capability,
    CompatibilityStatus,
    ContextRetentionDisposition,
    ContextRetentionRequirement,
    IncompatibleRequirementError,
    RequirementUnconfirmedError,
    SessionRequirements,
    SessionSpec,
    Supported,
    TaskRequest,
    TextInput,
    UnknownSupport,
    Unsupported,
    UserHistoryVisibility,
    UserHistoryVisibilityRequirement,
)
from langgraph.checkpoint.memory import InMemorySaver
from model_boundary import ControlledChatModel

from agent_harness_protocol_langgraph import LangGraphHarness

EPHEMERAL = SessionSpec(
    requirements=SessionRequirements(retention=ContextRetentionRequirement.EPHEMERAL)
)
HIDDEN = SessionSpec(
    requirements=SessionRequirements(history_visibility=UserHistoryVisibilityRequirement.HIDDEN)
)


async def test_adapter_owned_checkpointer_is_ephemeral_and_visibility_is_unknown() -> None:
    model = ControlledChatModel()
    async with LangGraphHarness(model) as harness:
        assert isinstance(harness.support[Capability.CONTEXT_RETENTION], Supported)
        assert isinstance(harness.support[Capability.USER_HISTORY_VISIBILITY], UnknownSupport)
        assert harness.validate(EPHEMERAL).status is CompatibilityStatus.COMPATIBLE
        assert harness.validate(HIDDEN).status is CompatibilityStatus.UNCONFIRMED

        session = await harness.create_session(EPHEMERAL)
        assert session.disposition.retention is ContextRetentionDisposition.EPHEMERAL
        assert session.disposition.history_visibility is UserHistoryVisibility.UNKNOWN
        with pytest.raises(RequirementUnconfirmedError):
            await harness.create_session(HIDDEN)
    assert model.calls == ()


async def test_supplied_checkpointer_without_declaration_is_unconfirmed() -> None:
    model = ControlledChatModel()
    async with LangGraphHarness(model, checkpointer=InMemorySaver()) as harness:
        assert isinstance(harness.support[Capability.CONTEXT_RETENTION], UnknownSupport)
        assert harness.validate(EPHEMERAL).status is CompatibilityStatus.UNCONFIRMED
        with pytest.raises(RequirementUnconfirmedError):
            await harness.create_session(EPHEMERAL)
        session = await harness.create_session(SessionSpec())
        assert session.disposition.retention is ContextRetentionDisposition.UNKNOWN
        task = await session.start_task(TaskRequest(TextInput("provider default still runs")))
        await task.await_outcome()
    assert len(model.calls) == 1


async def test_declared_materialized_and_visible_are_incompatible() -> None:
    model = ControlledChatModel()
    harness = LangGraphHarness(
        model,
        checkpointer=InMemorySaver(),
        context_retention=ContextRetentionDisposition.MATERIALIZED,
        history_visibility=UserHistoryVisibility.VISIBLE,
    )
    async with harness:
        assert isinstance(harness.support[Capability.CONTEXT_RETENTION], Unsupported)
        assert isinstance(harness.support[Capability.USER_HISTORY_VISIBILITY], Unsupported)
        for spec in (EPHEMERAL, HIDDEN):
            assert harness.validate(spec).status is CompatibilityStatus.INCOMPATIBLE
            with pytest.raises(IncompatibleRequirementError):
                await harness.create_session(spec)
    assert model.calls == ()


async def test_declared_ephemeral_and_hidden_are_admitted_and_reported() -> None:
    harness = LangGraphHarness(
        ControlledChatModel(),
        checkpointer=InMemorySaver(),
        context_retention=ContextRetentionDisposition.EPHEMERAL,
        history_visibility=UserHistoryVisibility.HIDDEN,
    )
    both = SessionSpec(
        requirements=SessionRequirements(
            retention=ContextRetentionRequirement.EPHEMERAL,
            history_visibility=UserHistoryVisibilityRequirement.HIDDEN,
        )
    )
    async with harness:
        assert isinstance(harness.support[Capability.CONTEXT_RETENTION], Supported)
        assert isinstance(harness.support[Capability.USER_HISTORY_VISIBILITY], Supported)
        session = await harness.create_session(both)
        assert session.disposition.retention is ContextRetentionDisposition.EPHEMERAL
        assert session.disposition.history_visibility is UserHistoryVisibility.HIDDEN


@pytest.mark.parametrize(
    "declared",
    [ContextRetentionDisposition.MATERIALIZED, ContextRetentionDisposition.UNKNOWN],
)
def test_adapter_owned_checkpointer_cannot_be_declared_otherwise(
    declared: ContextRetentionDisposition,
) -> None:
    with pytest.raises(ValueError):
        LangGraphHarness(ControlledChatModel(), context_retention=declared)
