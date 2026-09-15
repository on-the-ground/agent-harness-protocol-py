# Python protocol mapping

The normative viewpoint is the business application. A harness is an outsourced
execution responsibility behind the port. Whether it is a Python library, a graph,
another process, a server, or a cloud service is an adapter concern.

## Admission and requirements

`support` is discovery information. `validate()` is side-effect-free preflight.
`create_session()` and `start_task()` are the authoritative admission boundaries and
must repeat the relevant checks even when preflight was skipped.

`CompatibilityStatus.INCOMPATIBLE` means the adapter knows it cannot preserve a
required meaning. `UNCONFIRMED` means it cannot establish the necessary fact. They
raise `IncompatibleRequirementError` and `RequirementUnconfirmedError`, respectively,
at admission. Neither may be weakened to a warning or provider default.

`TaskStartUnconfirmedError` is different: delivery occurred but acceptance was not
confirmed. The context is blocked because retrying could duplicate work. Confirmed
nondelivery uses `HarnessTransportError` and permits a caller-controlled retry.

## Task lifecycle

An `AgentTask` is one business delegation, not one model call, graph node, tool call,
or provider turn. One task may contain many of those internal operations.

Every call to `events()` opens an independent bounded subscription. Slow readers get
an exact `ObservationGap` for discarded non-terminal events. One terminal event is
retained, appears last, and agrees with `await_outcome()`. State, pending interactions,
and terminal progress never depend on an event reader.

Cancelling a coroutine waiting in `await_outcome()` does not cancel the agent task.
`request_cancellation()` first clears pending interactions, then asks the native
runtime to stop. Only observed native termination yields `Cancelled`. If termination
cannot be confirmed, the result is `Unresolved` with the known evidence retained.

## Interactions and effects

Approvals and questions share delivery, one-response, and cleanup behavior but retain
typed request and response meanings. Invalid response kinds and unavailable approval
decisions are rejected before native delivery.

`APPROVE_FOR_SESSION` may be offered only with a `SessionApprovalGrant` whose scope can
be described and enforced. A one-shot approval never leaks to a later task or another
session. Execution constraints are hard upper bounds and cannot be widened by an
approval.

## Context and persistence

Sessions provide context continuity by default, with sequential task admission.
Persistence is optional and represented by `PersistentSessions` plus a qualified
`PersistentSessionRef`. A plain `SessionId` is not a durable or global identifier.

Aliases for one persistent context share active-work exclusion and uncertainty
blocking. Reopening does not imply checkpoint restoration, in-flight reconnection, or
external-effect rollback.

## Outcomes and accounting

All four outcomes retain acquired output and usage. Non-completed output is partial
(`complete=False`). `Unresolved` is an honest terminal judgment for this handle and
does not claim the external work stopped.

Usage events are cumulative snapshots, not deltas. Unknown fields remain `None` during
addition or subtraction. A measured all-zero baseline is `AgentUsage.ZERO`; no known
measurement is `AgentUsage.UNKNOWN`.

