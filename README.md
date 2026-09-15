# Agent Harness Protocol for Python

Agent Harness Protocol (AHP) is the business-facing port for delegating work to an
agent harness. A local graph, an embedded CLI or agent server, and a remote agent
service can implement the same purpose without exposing their execution loop, model
calls, transport, or persistence layout to application code.

## Why a port, when LangChain and LangGraph already exist?

LangChain and LangGraph are for *building* an agent: choosing the model, drawing the
graph, wiring tools, and checkpointing state. AHP is for the application that
*delegates work* to an agent and then has to act on the result. It does not replace a
graph framework. A LangGraph graph is one of the things AHP delegates to, alongside
command-line agents, agent servers, and remote agent services.

The port adds a contract written from the caller's side:

- **Purpose, not mechanism.** A caller states the guarantees it requires (keep no
  context after release, let the caller decide approvals, return output that matches a
  schema) instead of configuring one runtime's knobs. Capabilities are optional
  contracts that an adapter may add; they extend the port rather than shrinking it to
  what every runtime has in common.
- **Refusal instead of silent downgrade.** Discovery (`support`), preflight
  (`validate()`), and admission are separate facts. A requirement an adapter cannot keep
  raises `IncompatibleRequirementError`; one it cannot confirm raises
  `RequirementUnconfirmedError`.
- **Honest outcomes.** Cancelling a coroutine does not prove that a remote provider
  stopped, so `Unresolved` stays distinct from `Cancelled`. Unknown usage stays `None`
  rather than zero, and missing output differs from empty output.
- **Lifecycle guarantees written once.** Bounded cleanup, waiters whose cancellation
  leaves the work running, rejection of overlapping tasks, exact loss accounting for
  slow observers, and one stable terminal outcome belong to the contract instead of
  being rebuilt in every application.
- **Executable conformance.** Reusable suites compare an adapter with evidence observed
  at its real boundary, so an application can change harnesses without rewriting the
  logic that depends on these guarantees.

Building the LangGraph adapter showed the gap concretely. Used directly, LangGraph
leaves the following implicit; behind AHP each needs an explicit, tested answer:

- cancelling `ainvoke()` does not show that the model provider stopped working;
- the input of a cancelled turn remains in the checkpoint and reaches the next turn;
- `InMemorySaver` can be configured to write to disk;
- tracing and provider logs can expose inputs outside the application.

If an application designs its own graph and will only ever run LangGraph, use LangGraph
directly. AHP pays off when an application delegates tasks, judges their results by
explicit guarantees, and may run more than one kind of harness.

This repository is the native Python expression of the Kotlin AHP `0.2.3` contract.
The `agent-harness-protocol` distribution contains the protocol and reusable
conformance suites only. It deliberately ships no adapter, reference harness, or mock
implementation.

Adapters are separate distributions kept under `implementations/`, following the
Kotlin repository's protocol/implementations split:

| Distribution | Location | Runtime |
|---|---|---|
| `agent-harness-protocol` | `src/` | protocol and conformance suites, no runtime dependency |
| `agent-harness-protocol-langgraph` | `implementations/langgraph` | LangChain `BaseChatModel` inside a LangGraph `StateGraph` |

Each adapter documents its own support matrix; an adapter's limits never change the
protocol contract described here.

## What the contract preserves

- `AgentHarness` creates logical `AgentSession` contexts and admits task requirements.
- `AgentSession.start_task()` returns an `AgentTask` after admission, without waiting
  for completion.
- `AgentTask` exposes current state, independent event subscriptions, pending typed
  interactions, cancellation, and one stable terminal outcome.
- `Completed`, `Failed`, `Cancelled`, and `Unresolved` remain distinct. Requesting
  cancellation or losing a stream never fabricates confirmed termination.
- Missing output differs from `TextOutput("")`; unknown usage fields differ from
  measured zero.
- Capability discovery, preflight validation, and actual admission are separate facts.
  A caller may state requirements without querying support first.

The port contains no provider-payload escape hatch in inbound requests. Provider
configuration belongs to an adapter's construction boundary; portable application
logic deals in required purpose and guarantees.

## Python API

```python
from agent_harness_protocol import AgentHarness, Completed, SessionSpec, TaskRequest, TextInput


async def run(harness: AgentHarness) -> None:
    async with harness:
        session = await harness.create_session(SessionSpec(instructions="Keep changes reviewable"))
        task = await session.start_task(TaskRequest(TextInput("Update the report")))

        async for event in task.events():
            render(event)

        outcome = await task.await_outcome()
        if isinstance(outcome, Completed):
            consume(outcome.output)
```

Python's `async with`, `AsyncIterator`, snake_case, frozen dataclasses, and `Error`
suffixes replace Kotlin-specific resource, `Flow`, naming, and exception forms. The
semantic distinctions remain the same:

| Kotlin representation | Python representation |
|---|---|
| `Flow<TaskEvent>` | a fresh `AsyncIterator[TaskEvent]` from `task.events()` |
| `StateFlow<TaskState>` | immediately readable `task.state` snapshot |
| `StateFlow<List<InteractionRequest>>` | immutable `task.pending_interactions` snapshot |
| `close()` / `release()` | awaitable `aclose()` / `release()` and `async with` |
| `PersistenceRequirement.NotRequired` | `None` |
| `WorkspaceRequirement.NotRequired` | `None` |
| `ExecutionConstraint.ProviderDefault` | `None` |
| Kotlin `*Exception` | Python `*Error` |

`SessionSpec.instructions=None` still means provider default, while `""` explicitly
requests empty instructions. `None` in an `AgentUsage` field means unknown; it is not
zero. `None` output means no output was acquired; it is not an empty response.

## Conformance suites

Install the test extra and inherit the relevant suite from a pytest class whose name
starts with `Test`:

```bash
python -m pip install -e ".[test]"
```

```python
from agent_harness_protocol.conformance import RuntimeProfileConformanceTests


class TestLangGraphRuntime(RuntimeProfileConformanceTests):
    def boundary(self):
        return langgraph_boundary_fixture()

    def harness(self, boundary):
        return make_langgraph_harness(boundary)
```

Fixture observations must come from the actual model, SDK, process, storage, or effect
boundary. They must not be copied from `AgentTask.state`, events, or outcomes, and a
fixture must not directly force public task state just to pass a scenario. The package
contains 58 reusable scenario methods across admission, lifecycle, interaction,
cleanup, persistence, execution bounds, workspace/skills, output, accounting, and
observation load. The requirements scenario expands every declared profile and case
in both preflight and direct-admission modes.

See [the protocol mapping](https://github.com/on-the-ground/agent-harness-protocol-py/blob/main/docs/protocol.md) and
[conformance guide](https://github.com/on-the-ground/agent-harness-protocol-py/blob/main/docs/conformance.md) before binding an adapter. A complete binding
against a real runtime is in
[`implementations/langgraph/tests/test_conformance.py`](https://github.com/on-the-ground/agent-harness-protocol-py/blob/main/implementations/langgraph/tests/test_conformance.py).