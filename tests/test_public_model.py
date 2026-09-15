from dataclasses import replace

import pytest

from agent_harness_protocol import (
    READ_ONLY,
    AgentUsage,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalScopeId,
    CompatibilityIssue,
    CompatibilityIssueKind,
    CompatibilityReport,
    CompatibilityStatus,
    ContextRetentionDisposition,
    ContextRetentionRequirement,
    DiagnosticEvent,
    DiagnosticGap,
    EffectKind,
    ExecutionConstraint,
    IncompatibleRequirementError,
    InteractionId,
    InteractionResponseUnconfirmedError,
    NetworkAccess,
    ProviderDiagnostic,
    ProviderId,
    RequirementUnconfirmedError,
    SessionApprovalGrant,
    SessionDisposition,
    SessionRequirements,
    StopReason,
    TaskEvent,
    TaskId,
    TextInput,
    TextOutput,
    UnconfirmedResponse,
    UserHistoryVisibility,
    UserHistoryVisibilityRequirement,
    WorkId,
)
from agent_harness_protocol.outcome import Completed


def test_unconfirmed_requirement_is_neither_compatible_nor_confirmed_rejection() -> None:
    report = CompatibilityReport(
        [
            CompatibilityIssue(
                "requirements.persistence",
                "storage is unreachable",
                CompatibilityIssueKind.UNCONFIRMED,
            )
        ]
    )
    assert report.status is CompatibilityStatus.UNCONFIRMED
    assert not report.is_compatible
    with pytest.raises(RequirementUnconfirmedError) as caught:
        report.require_compatible()
    assert caught.value.issues == report.issues


def test_known_incompatibility_remains_decisive_and_advisory_remains_compatible() -> None:
    report = CompatibilityReport(
        [
            CompatibilityIssue("requirements.questions", "unavailable"),
            CompatibilityIssue(
                "requirements.persistence",
                "not checked",
                CompatibilityIssueKind.UNCONFIRMED,
            ),
        ]
    )
    assert report.status is CompatibilityStatus.INCOMPATIBLE
    with pytest.raises(IncompatibleRequirementError) as caught:
        report.require_compatible()
    assert caught.value.issues == report.issues
    advisory = CompatibilityReport(
        [CompatibilityIssue("model", "using configured default", CompatibilityIssueKind.ADVISORY)]
    )
    advisory.require_compatible()
    assert advisory.is_compatible


def test_no_output_differs_from_actual_empty_output() -> None:
    no_output = Completed(TaskId("task"), StopReason.FINISHED)
    empty_output = replace(no_output, output=TextOutput(""))
    assert no_output.output is None
    assert empty_output.output == TextOutput("")


def test_unmeasured_segment_prevents_fabricated_total() -> None:
    first = AgentUsage(input_tokens=10, output_tokens=3, cache_write_input_tokens=4)
    second = AgentUsage(input_tokens=None, output_tokens=2, cache_write_input_tokens=2)
    total = first + second
    assert total.input_tokens is None
    assert total.output_tokens == 5
    assert total.cache_write_input_tokens == 6
    assert total.total_tokens is None
    assert AgentUsage.ZERO + first == first
    assert AgentUsage.UNKNOWN + first == AgentUsage.UNKNOWN


def test_counter_reset_does_not_yield_negative_usage() -> None:
    current = AgentUsage(input_tokens=4, output_tokens=20, cache_write_input_tokens=7)
    baseline = AgentUsage(input_tokens=10, output_tokens=12, cache_write_input_tokens=3)
    delta = current - baseline
    assert delta.input_tokens is None
    assert delta.output_tokens == 8
    assert delta.cache_write_input_tokens == 4
    assert current - AgentUsage.UNKNOWN == AgentUsage.UNKNOWN


def test_network_only_constraint_does_not_choose_filesystem_policy() -> None:
    requirement = ExecutionConstraint(network=NetworkAccess.DENIED)
    assert requirement.filesystem is None
    assert requirement.network is NetworkAccess.DENIED
    assert ExecutionConstraint(filesystem=READ_ONLY).network is None
    with pytest.raises(ValueError):
        ExecutionConstraint()


def test_retention_visibility_and_persistence_remain_distinct() -> None:
    requirements = SessionRequirements(retention=ContextRetentionRequirement.EPHEMERAL)
    assert requirements.history_visibility is UserHistoryVisibilityRequirement.PROVIDER_DEFAULT
    assert requirements.persistence is None
    observed = SessionDisposition(
        ContextRetentionDisposition.EPHEMERAL, UserHistoryVisibility.UNKNOWN
    )
    assert observed.retention is ContextRetentionDisposition.EPHEMERAL
    assert observed.history_visibility is UserHistoryVisibility.UNKNOWN


def test_session_approval_requires_explicit_enforceable_grant() -> None:
    decisions = {ApprovalDecision.APPROVE_FOR_SESSION, ApprovalDecision.DECLINE}
    with pytest.raises(ValueError):
        ApprovalRequest(
            InteractionId("approval"),
            WorkId("effect"),
            "update the report",
            EffectKind.FILE_CHANGE,
            decisions,
        )
    grant = SessionApprovalGrant(
        ApprovalScopeId("reports"), "Write files only in the report directory"
    )
    request = ApprovalRequest(
        InteractionId("approval"),
        WorkId("effect"),
        "update the report",
        EffectKind.FILE_CHANGE,
        decisions,
        grant,
    )
    assert request.session_grant == grant
    with pytest.raises(ValueError):
        ApprovalRequest(
            InteractionId("approval"),
            WorkId("effect"),
            "update",
            EffectKind.FILE_CHANGE,
            {ApprovalDecision.APPROVE_ONCE},
            grant,
        )
    with pytest.raises(ValueError):
        SessionApprovalGrant(ApprovalScopeId("reports"), " ")


def test_response_uncertainty_retains_task_and_interaction_identity() -> None:
    reference = UnconfirmedResponse(TaskId("task"), InteractionId("approval"))
    failure = InteractionResponseUnconfirmedError(reference, "acknowledgement lost")
    assert failure.reference == reference


def test_provider_diagnostics_are_not_semantic_events() -> None:
    record: DiagnosticEvent = ProviderDiagnostic(
        TaskId("task"), ProviderId("test"), "trace", "payload"
    )
    assert not isinstance(record, TaskEvent)
    with pytest.raises(ValueError):
        DiagnosticGap(TaskId("task"), 0)


def test_text_input_rejects_empty_but_preserves_whitespace() -> None:
    with pytest.raises(ValueError):
        TextInput("")
    assert TextInput(" \t ").text == " \t "
