# LangGraph 1.x ships ``langgraph.graph`` without complete type information. The two
# relaxations below are confined to this module, which exposes a fully typed surface.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
"""Typed boundary around the partially typed LangGraph graph API."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph

ModelCall = Callable[[Sequence[BaseMessage]], Awaitable[BaseMessage]]
"""Model node body: receives the thread's messages and returns the reply."""


class ModelOnlyGraph:
    """A compiled ``START -> model -> END`` graph whose state lives in a checkpointer."""

    def __init__(self, call_model: ModelCall, checkpointer: BaseCheckpointSaver[str]) -> None:
        """Compile the graph.

        Args:
            call_model: Called once per turn with the accumulated thread messages.
            checkpointer: Stores the messages of every thread.
        """

        async def model(state: MessagesState) -> dict[str, list[BaseMessage]]:
            return {"messages": [await call_model(state["messages"])]}

        builder = StateGraph(MessagesState)
        builder.add_node("model", model)
        builder.add_edge(START, "model")
        builder.add_edge("model", END)
        self._compiled = builder.compile(checkpointer=checkpointer)

    async def run_turn(self, thread_id: str, text: str) -> AIMessage | None:
        """Append one human message to the thread and run the model node.

        Args:
            thread_id: The checkpoint thread of the session.
            text: The human message, passed through unchanged.

        Returns:
            The latest AI message in the thread, or ``None`` if there is none.

        Raises:
            TypeError: If the graph returns an unexpected state shape.
        """
        result: object = await self._compiled.ainvoke(
            {"messages": [HumanMessage(content=text)]},
            {"configurable": {"thread_id": thread_id}},
        )
        return _last_ai_message(result)


def _last_ai_message(result: object) -> AIMessage | None:
    """Narrow the untyped graph result to its latest AI message."""
    if not isinstance(result, dict):
        raise TypeError(f"graph returned {type(result).__name__}, expected a state mapping")
    messages = cast(dict[str, object], result).get("messages")
    if not isinstance(messages, list):
        raise TypeError("graph state has no message list")
    for message in reversed(cast(list[object], messages)):
        if isinstance(message, AIMessage):
            return message
    return None
