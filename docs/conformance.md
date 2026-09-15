# Binding the conformance suites

The suite classes intentionally do not begin with `Test`, so installing this package
does not collect them by itself. An adapter defines a `Test*` subclass and returns the
smallest fixture required by that suite. Inherited `test_*` methods are then collected
by pytest.

Use actual boundary evidence:

- `RuntimeObservation.observed_contexts` records real model inputs.
- `StartControl` and `ResponseControl` observe submission and acknowledgement at the
  real delivery boundary.
- effect and workspace fixtures inspect isolated files and network endpoints.
- persistence fixtures manipulate the adapter's actual backing storage.
- lifecycle controls are reserved for SDK-boundary lifecycle assertions; they must
  feed the same adapter ingestion path as native notifications.

Do not implement a universal fake harness to satisfy the suites. Do not compute
fixture expectations from `support`, `validate`, task state, events, or outcomes. A
passing test must compare the public port with evidence independently observed at the
boundary being claimed.

The suites are grouped by the facts an adapter can actually expose. Unsupported
features should be rejected honestly by the requirements matrix rather than simulated
as successful. Bind only suites whose fixture seam exists, while declaring every
`Capability` explicitly in each requirements profile, including `UnknownSupport`.

Important timing rules:

- suite timeouts are generous outer safety bounds, not advertised cleanup budgets;
- cleanup assertions compare elapsed time to `CleanupBudget.total` once per operation;
- cancelling a Python waiter or cleanup caller must not erase the underlying work or
  cleanup obligation;
- start and response acknowledgement loss must retain the submitted identity and
  prevent automatic duplicate delivery.

## Current adapter bindings

This section records which suites existing adapters actually bind. It does not relax
any rule above; an unbound suite means the adapter has no such route and rejects the
corresponding requirement at admission.

### LangChain + LangGraph (`implementations/langgraph`)

Fixture seam: a controllable LangChain `BaseChatModel` replaces only the model. The
harness, compiled `StateGraph`, and checkpointer are real, and `RuntimeObservation` is
read from the messages the model received.

| Suite | Status |
|---|---|
| `RuntimeProfileConformanceTests` | bound; the fixture selects confirmed coroutine cancellation because the controlled model's coroutine is the whole native work |
| `RequirementsConformanceTests` | bound with four disposition profiles |
| `CleanupBudgetConformanceTests` | bound with confirmed, unconfirmed, and cancellation-ignoring models |
| `InteractionConformanceTests`, `ApprovalScopeConformanceTests`, `ResponseAcceptanceConformanceTests` | not bound: no approval or question route |
| `StartAcceptanceConformanceTests` | not bound: the in-process graph has no delivery boundary where acknowledgement can be lost |
| `RuntimePersistenceConformanceTests`, `PersistenceFailureConformanceTests`, `ContextConformanceTests` | not bound: persistence and reopening are unsupported |
| `WorkspaceConformanceTests`, `ExecutionConstraintConformanceTests` | not bound: workspace and execution constraints are unsupported |
| `LifecycleConformanceTests` | not bound: its controls must feed a native notification ingestion path, which this graph does not have |
| `ObservationLoadConformanceTests` | not bound yet: no `MessageDelta` stream |
| `AcceptedStartConformanceTests`, `OutcomeConformanceTests`, `AccountingConformanceTests`, `AccountingSequenceConformanceTests` | not bound yet |
