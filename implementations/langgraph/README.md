# Agent Harness Protocol for LangChain + LangGraph

`agent-harness-protocol-langgraph` implements the Python Agent Harness Protocol (AHP)
`0.2.3` port with a LangChain `BaseChatModel` running inside a LangGraph `StateGraph`.
It is a separate distribution from the core `agent-harness-protocol` package, which
contains no LangChain or LangGraph dependency.

Each AHP session compiles a `START -> model -> END` graph with its own checkpoint
thread. Sequential tasks in one session share the conversation; different sessions are
isolated. The adapter reports what this graph can actually guarantee and rejects the
rest at admission instead of simulating it.

## Installation

```bash
python -m pip install agent-harness-protocol-langgraph
```

Requirements: Python 3.11+, `langchain-core>=1.2,<2`, `langgraph>=1.1,<2`.

## Minimal example

```python
from agent_harness_protocol import Completed, SessionSpec, TaskRequest, TextInput
from agent_harness_protocol_langgraph import LangGraphHarness
from langchain_core.language_models import BaseChatModel


async def run(model: BaseChatModel, model_id: str) -> None:
    async with LangGraphHarness(model, model_id=model_id) as harness:
        session = await harness.create_session(SessionSpec(instructions="Answer briefly"))
        task = await session.start_task(TaskRequest(TextInput("Summarize the report")))

        async for event in task.events():
            print(event)

        outcome = await task.await_outcome()
        if isinstance(outcome, Completed):
            print(outcome.output)
```

`SessionSpec.model` is admitted only when it equals the `model_id` given at
construction. Without a `model_id`, any explicit model request is rejected.

## Support matrix

| Capability | Support | Admission |
|---|---|---|
| Diagnostics | `Supported` | `task.diagnostics()` streams graph start and terminal records |
| Context retention | Declared, see below | `EPHEMERAL` requirement follows the declaration |
| User-history visibility | Declared, see below | `HIDDEN` requirement follows the declaration |
| Caller approval | `Unsupported` | `CALLER_DECIDES` and `AGENT_REVIEWED` are incompatible; `DENY_ALL` is admitted because the graph has no effect route |
| Questions | `Unsupported` | `CALLER_ANSWERS` is incompatible |
| Persistence | `Unsupported` | any `PersistenceRequirement` is incompatible |
| Workspace and skills | `Unsupported` | any `WorkspaceRequirement` is incompatible |
| Execution constraint | `Unsupported` | any `ExecutionConstraint` is incompatible |
| Structured output | `Unsupported` | sessions are admitted; a structured task request is incompatible |

Only `TextInput` is accepted. Semantic events are `TaskStarted`, `MessageCompleted`,
`UsageChanged`, and one terminal event. `MessageDelta` is not produced because the
graph is not streamed yet.

## Disposition declarations

Whether stored context is ephemeral and whether inputs stay hidden depend on facts the
adapter cannot observe, so they are declared at construction and reflected identically
in `support`, `validate()`, admission, and `session.disposition`.

```python
from agent_harness_protocol import ContextRetentionDisposition, UserHistoryVisibility

harness = LangGraphHarness(
    model,
    checkpointer=my_saver,
    context_retention=ContextRetentionDisposition.MATERIALIZED,
    history_visibility=UserHistoryVisibility.VISIBLE,
)
```

| Configuration | Retention | Visibility |
|---|---|---|
| no `checkpointer` | `EPHEMERAL`: adapter-owned in-memory saver, deleted on release; declaring anything else raises `ValueError` | declared, default `UNKNOWN` |
| supplied `checkpointer` | declared, default `UNKNOWN` | declared, default `UNKNOWN` |

| Declaration | Support | `EPHEMERAL` / `HIDDEN` requirement |
|---|---|---|
| `EPHEMERAL` / `HIDDEN` | `Supported` | compatible |
| `MATERIALIZED` / `VISIBLE` | `Unsupported` | `IncompatibleRequirementError` |
| `UNKNOWN` | `UnknownSupport` | `RequirementUnconfirmedError` |

Visibility defaults to `UNKNOWN` because provider request logs or LangSmith tracing can
expose inputs outside the application. Declare `HIDDEN` only when that is known to be
false for your deployment. A type name is not evidence: `InMemorySaver(factory=...)` can
write to disk.

## Cancellation semantics

Cancelling a LangChain coroutine does not by itself prove that a remote provider stopped
working, so the outcome depends on `cancellation_semantics`:

| Situation | `UNCONFIRMED` (default) | `COROUTINE_TERMINATION_CONFIRMS_WORK_STOPPED` |
|---|---|---|
| cancelled before the model was called | `Cancelled`, usage measured zero | `Cancelled`, usage measured zero |
| graph run ended after cancelling a model call | `Unresolved(CANCELLATION_UNCONFIRMED)` | `Cancelled` |
| graph run did not end within `cleanup_budget.per_task` | `Unresolved(CANCELLATION_UNCONFIRMED)` | `Unresolved(CANCELLATION_UNCONFIRMED)` |

Choose the second value only when coroutine termination really ends the native work,
for example with a local in-process model.

The adapter delivers one cancellation request per task. While a graph run that outlived
its settled task is still running, `start_task()` raises `SessionBlockedError`, so two
runs never write to the same checkpoint thread. The session becomes usable again when
that run ends.

A cancelled turn's input remains in the checkpoint and is part of the next task's
context, without a matching model reply, because LangGraph records the input before
the model node runs.

## Cleanup

`session.release()` and `harness.aclose()` settle active tasks within the published
`CleanupBudget` (default: 1 s per task, 3 s total). Cancelling the caller does not
interrupt cleanup.

Releasing a session deletes its checkpoint thread within the remaining total budget.
If deletion fails or times out, `CheckpointCleanupError` is raised; task outcomes are
unchanged and the handle is still closed. `aclose()` settles every task before raising
one aggregated `CheckpointCleanupError`. When a graph run outlives cleanup, its thread
is deleted again after the run ends; that background deletion has no failure report.

## Usage accounting

`AIMessage.usage_metadata` becomes `AgentUsage`. Missing metadata or a missing field is
`None` (unknown), never zero. Session usage is committed once per task outcome and a
field stays unknown once any task left it unknown.

Failures are reported as `Failed(FailureKind.UNKNOWN)` with the original exception as
`cause`; the adapter does not classify errors from message text.

## Verification

The tests replace only the chat model with a controllable `BaseChatModel`; the harness,
compiled graph, and checkpointer are real, and every observation is read from what the
model received.

Bound shared suites:

- `RuntimeProfileConformanceTests`
- `RequirementsConformanceTests`, with four disposition profiles
- `CleanupBudgetConformanceTests`, with confirmed, unconfirmed, and cancellation-ignoring
  models

Suites for interactions, approval scope, persistence, workspace, and execution
constraints are not bound because this graph has no such route; the requirements suite
verifies that those requirements are rejected. The full list, including suites not bound
yet, is in [the conformance guide](../../docs/conformance.md#current-adapter-bindings).

```bash
python -m pip install -e ../..[test] -e .[test]
python -m pytest
```

Design decisions are recorded in
[`docs/langgraph-adapter-design.md`](../../docs/langgraph-adapter-design.md).
