"""A controllable LangChain chat model used as the native model-boundary test double.

Only the model boundary is replaced. Every test drives the real ``LangGraphHarness``,
the real compiled ``StateGraph``, and a real LangGraph checkpointer. Observations are
recorded where the model receives its input, never copied from AHP task state.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


class ControlledChatModel(BaseChatModel):
    reply: str = "native-result"
    usage: UsageMetadata | None = None
    failure: str | None = None
    ignore_cancellation: bool = False

    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=lambda: list[list[BaseMessage]]())
    _cancellations: int = PrivateAttr(default=0)
    _completed: int = PrivateAttr(default=0)
    _in_flight: int = PrivateAttr(default=0)
    _gate: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)

    def model_post_init(self, context: Any, /) -> None:
        self._gate.set()

    @property
    def _llm_type(self) -> str:
        return "ahp-controlled"

    @property
    def calls(self) -> Sequence[Sequence[BaseMessage]]:
        return tuple(tuple(call) for call in self._calls)

    @property
    def cancellations(self) -> int:
        return self._cancellations

    @property
    def completed(self) -> int:
        return self._completed

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def hold(self) -> None:
        self._gate.clear()

    def release(self) -> None:
        self._gate.set()

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("the controlled model is asynchronous only")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._calls.append(list(messages))
        self._in_flight += 1
        try:
            try:
                await self._gate.wait()
            except asyncio.CancelledError:
                self._cancellations += 1
                if not self.ignore_cancellation:
                    raise
                await self._gate.wait()
            if self.failure is not None:
                raise RuntimeError(self.failure)
            self._completed += 1
            message = AIMessage(content=self.reply, usage_metadata=self.usage)
            return ChatResult(generations=[ChatGeneration(message=message)])
        finally:
            self._in_flight -= 1


def contents(messages: Sequence[BaseMessage]) -> list[str]:
    return [message.text for message in messages]


class ModelBoundaryObservation:
    """``RuntimeObservation`` read from what the controlled model actually received."""

    def __init__(self, model: ControlledChatModel) -> None:
        self.model = model

    @property
    def observed_contexts(self) -> Sequence[str]:
        return tuple("\n".join(contents(call)) for call in self.model.calls)

    @property
    def observed_text_values(self) -> Sequence[str]:
        return tuple(text for call in self.model.calls for text in contents(call))

    def hold(self) -> None:
        self.model.hold()

    def release(self) -> None:
        self.model.release()

    async def settle(self, timeout: float = 5.0) -> None:
        """Release held calls and wait until no native model call is still running."""
        self.model.release()
        async with asyncio.timeout(timeout):
            while self.model.in_flight:
                await asyncio.sleep(0.005)
        for _ in range(3):
            await asyncio.sleep(0)
