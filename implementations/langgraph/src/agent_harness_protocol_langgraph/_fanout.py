"""Bounded, independent event subscriptions with exact loss accounting."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class _Gap:
    """A run of consecutive items dropped at this position of one subscription."""

    count: int


@dataclass(slots=True, eq=False)
class _Subscriber(Generic[T]):
    """Buffer and wake-up signal of one subscription."""

    entries: deque[T | _Gap] = field(default_factory=lambda: deque[T | _Gap]())
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    ended: bool = False


class BoundedFanout(Generic[T]):
    """Independent bounded subscriptions with exact, positioned loss accounting.

    Each subscription buffers at most ``capacity`` entries, where a run of dropped items
    occupies one entry. When the buffer is full, the newest items are dropped and counted
    at the tail, so a gap always appears exactly where the loss happened. The optional
    final item is retained outside the buffer and is always delivered last; closing never
    evicts or reorders buffered entries.
    """

    def __init__(
        self,
        *,
        capacity: int,
        gap: Callable[[int], T],
    ) -> None:
        """Create a fan-out.

        Args:
            capacity: Entries buffered per subscription, including gap markers.
            gap: Builds the item that reports a run of dropped items.

        Raises:
            ValueError: If ``capacity`` is below two.
        """
        if capacity < 2:
            raise ValueError("fanout capacity must leave room for one item and one gap")
        self._capacity = capacity
        self._gap = gap
        self._subscribers: set[_Subscriber[T]] = set()
        self._closed = False
        self._final: T | None = None

    @property
    def closed(self) -> bool:
        """Whether `close` has been called."""
        return self._closed

    def publish(self, item: T) -> None:
        """Offer an item to every subscription; full subscriptions count it as dropped."""
        if self._closed:
            return
        for subscriber in self._subscribers:
            entries = subscriber.entries
            tail = entries[-1] if entries else None
            if isinstance(tail, _Gap) and len(entries) >= self._capacity:
                tail.count += 1
            elif len(entries) < self._capacity - 1:
                entries.append(item)
            elif isinstance(tail, _Gap):
                tail.count += 1
            else:
                entries.append(_Gap(1))
            subscriber.ready.set()

    def close(self, final_item: T | None = None) -> None:
        """End every subscription after its buffered entries.

        Args:
            final_item: Delivered last to current and later subscribers, if given.
        """
        if self._closed:
            return
        self._closed = True
        self._final = final_item
        for subscriber in self._subscribers:
            subscriber.ended = True
            subscriber.ready.set()
        self._subscribers.clear()

    def subscribe(self) -> AsyncIterator[T]:
        """Register immediately; items published after this call are observable."""
        if self._closed:
            return self._closed_stream()
        subscriber = _Subscriber[T]()
        self._subscribers.add(subscriber)
        return self._stream(subscriber)

    async def _closed_stream(self) -> AsyncIterator[T]:
        """Stream for a subscriber that arrives after close: only the final item."""
        if self._final is not None:
            yield self._final

    async def _stream(self, subscriber: _Subscriber[T]) -> AsyncIterator[T]:
        """Drain one subscription, materialising gaps, then deliver the final item."""
        try:
            while True:
                if subscriber.entries:
                    entry = subscriber.entries.popleft()
                    yield self._gap(entry.count) if isinstance(entry, _Gap) else entry
                    continue
                if subscriber.ended:
                    if self._final is not None:
                        yield self._final
                    return
                subscriber.ready.clear()
                await subscriber.ready.wait()
        finally:
            self._subscribers.discard(subscriber)
