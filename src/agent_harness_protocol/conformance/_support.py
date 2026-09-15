from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from .fixtures import FixtureResource

T = TypeVar("T")


@asynccontextmanager
async def managed(resource: FixtureResource[T]) -> AsyncGenerator[T, None]:
    async with resource as value:
        yield value


async def wait_until(
    condition: Callable[[], bool], timeout: float = 60.0, interval: float = 0.01
) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(interval)


async def collect(source: AsyncIterator[T]) -> list[T]:
    return [item async for item in source]


async def cancel_waiter(waiter: asyncio.Task[Any]) -> None:
    waiter.cancel()
    try:
        await waiter
    except asyncio.CancelledError:
        pass


async def within(seconds: float, awaitable: Awaitable[T]) -> T:
    return await asyncio.wait_for(awaitable, seconds)
