"""Typed requests for caller judgment and information."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from .identity import ApprovalScopeId, InteractionId, WorkId


class EffectKind(StrEnum):
    COMMAND = "command"
    FILE_CHANGE = "file_change"
    WEB_SEARCH = "web_search"
    OTHER = "other"


class ApprovalDecision(StrEnum):
    APPROVE_ONCE = "approve_once"
    APPROVE_FOR_SESSION = "approve_for_session"
    DECLINE = "decline"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class SessionApprovalGrant:
    scope_id: ApprovalScopeId
    description: str

    def __post_init__(self) -> None:
        if not self.description or self.description.isspace():
            raise ValueError("session grant must describe its targets and conditions")


class InteractionRequest:
    interaction_id: InteractionId
    work_id: WorkId | None
    detail: str | None


@dataclass(frozen=True, slots=True)
class ApprovalRequest(InteractionRequest):
    interaction_id: InteractionId
    work_id: WorkId | None
    prompt: str
    effect: EffectKind | None
    available_decisions: frozenset[ApprovalDecision]
    session_grant: SessionApprovalGrant | None = None
    detail: str | None = None

    def __init__(
        self,
        interaction_id: InteractionId,
        work_id: WorkId | None,
        prompt: str,
        effect: EffectKind | None,
        available_decisions: Iterable[ApprovalDecision],
        session_grant: SessionApprovalGrant | None = None,
        detail: str | None = None,
    ) -> None:
        decisions = frozenset(available_decisions)
        if not decisions:
            raise ValueError("approval must offer at least one decision")
        offers_session = ApprovalDecision.APPROVE_FOR_SESSION in decisions
        if offers_session != (session_grant is not None):
            raise ValueError(
                "session approval requires an explicit grant, and only that decision may carry one"
            )
        object.__setattr__(self, "interaction_id", interaction_id)
        object.__setattr__(self, "work_id", work_id)
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "effect", effect)
        object.__setattr__(self, "available_decisions", decisions)
        object.__setattr__(self, "session_grant", session_grant)
        object.__setattr__(self, "detail", detail)


@dataclass(frozen=True, slots=True)
class QuestionRequest(InteractionRequest):
    interaction_id: InteractionId
    work_id: WorkId | None
    prompt: str
    choices: tuple[str, ...] = ()
    allows_free_form: bool = False
    detail: str | None = None

    def __init__(
        self,
        interaction_id: InteractionId,
        work_id: WorkId | None,
        prompt: str,
        choices: Iterable[str] = (),
        allows_free_form: bool | None = None,
        detail: str | None = None,
    ) -> None:
        choices_tuple = tuple(choices)
        object.__setattr__(self, "interaction_id", interaction_id)
        object.__setattr__(self, "work_id", work_id)
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "choices", choices_tuple)
        object.__setattr__(
            self,
            "allows_free_form",
            not choices_tuple if allows_free_form is None else allows_free_form,
        )
        object.__setattr__(self, "detail", detail)


class InteractionResponse:
    pass


@dataclass(frozen=True, slots=True)
class ApprovalResponse(InteractionResponse):
    decision: ApprovalDecision


@dataclass(frozen=True, slots=True)
class AnswerResponse(InteractionResponse):
    text: str


class ClearReason(StrEnum):
    CANCELLATION_REQUESTED = "cancellation_requested"
    TASK_ENDED = "task_ended"
    SUPERSEDED = "superseded"
    PROVIDER_WITHDRAWN = "provider_withdrawn"
    RESPONSE_UNCONFIRMED = "response_unconfirmed"


class InteractionResolution:
    pass


@dataclass(frozen=True, slots=True)
class Responded(InteractionResolution):
    response: InteractionResponse


@dataclass(frozen=True, slots=True)
class Cleared(InteractionResolution):
    reason: ClearReason
