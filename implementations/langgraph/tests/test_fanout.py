"""Loss accounting and ordering of the bounded event fan-out."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator

import pytest

from agent_harness_protocol_langgraph._fanout import BoundedFanout


def _fanout(capacity: int) -> BoundedFanout[str]:
    return BoundedFanout[str](capacity=capacity, gap=lambda count: f"gap:{count}")


async def _drain(stream: AsyncIterator[str]) -> list[str]:
    return [item async for item in stream]


def test_capacity_must_leave_room_for_an_item_and_a_gap() -> None:
    with pytest.raises(ValueError):
        _fanout(1)


@pytest.mark.parametrize("capacity", [2, 3, 8])
async def test_close_with_full_buffer_keeps_order_gap_and_final(capacity: int) -> None:
    fanout = _fanout(capacity)
    stream = fanout.subscribe()
    published = [f"e{index}" for index in range(capacity + 5)]
    for item in published:
        fanout.publish(item)
    fanout.close("final")

    received = await asyncio.wait_for(_drain(stream), 1)

    kept = capacity - 1
    assert received == [*published[:kept], f"gap:{len(published) - kept}", "final"]


async def test_close_with_exactly_full_item_buffer_does_not_evict() -> None:
    fanout = _fanout(4)
    stream = fanout.subscribe()
    for item in ("a", "b", "c"):
        fanout.publish(item)
    fanout.close("final")

    assert await asyncio.wait_for(_drain(stream), 1) == ["a", "b", "c", "final"]


async def test_gap_marks_the_loss_position_and_delivery_resumes_after_drain() -> None:
    fanout = _fanout(3)
    stream = fanout.subscribe()
    for item in ("a", "b", "c", "d"):
        fanout.publish(item)
    assert await anext(stream) == "a"
    assert await anext(stream) == "b"
    fanout.publish("e")
    fanout.publish("f")
    fanout.close("final")

    # The undelivered gap still occupies one of the three buffer entries.
    assert await asyncio.wait_for(_drain(stream), 1) == ["gap:2", "e", "gap:1", "final"]


async def test_subscription_is_registered_before_first_iteration() -> None:
    fanout = _fanout(4)
    stream = fanout.subscribe()
    fanout.publish("before-iteration")
    fanout.close()

    assert await asyncio.wait_for(_drain(stream), 1) == ["before-iteration"]


async def test_late_subscribers_receive_only_the_retained_final() -> None:
    with_final = _fanout(4)
    with_final.publish("unobserved")
    with_final.close("final")
    without_final = _fanout(4)
    without_final.close()

    assert await _drain(with_final.subscribe()) == ["final"]
    assert await _drain(without_final.subscribe()) == []


async def test_waiting_subscriber_wakes_on_publish_and_close() -> None:
    fanout = _fanout(4)
    reader = asyncio.create_task(_drain(fanout.subscribe()))
    await asyncio.sleep(0)
    fanout.publish("a")
    await asyncio.sleep(0)
    fanout.close("final")

    assert await asyncio.wait_for(reader, 1) == ["a", "final"]


async def test_subscribers_are_independent() -> None:
    fanout = _fanout(2)
    slow = fanout.subscribe()
    fast = fanout.subscribe()
    fanout.publish("a")
    assert await anext(fast) == "a"
    fanout.publish("b")
    fanout.close("final")

    assert await _drain(fast) == ["b", "final"]
    assert await _drain(slow) == ["a", "gap:1", "final"]


@pytest.mark.parametrize("seed", range(40))
async def test_randomized_accounting_is_exact(seed: int) -> None:
    rng = random.Random(seed)
    capacity = rng.randint(2, 6)
    fanout = _fanout(capacity)
    reader = asyncio.create_task(_drain(fanout.subscribe()))
    published: list[str] = []
    for index in range(rng.randint(0, 80)):
        item = f"e{index}"
        published.append(item)
        fanout.publish(item)
        for _ in range(rng.choice((0, 0, 0, 1, 2))):
            await asyncio.sleep(0)
    fanout.close("final")
    received = await asyncio.wait_for(reader, 1)

    assert received[-1] == "final"
    body = received[:-1]
    delivered = [item for item in body if not item.startswith("gap:")]
    dropped = sum(int(item.split(":")[1]) for item in body if item.startswith("gap:"))
    assert delivered == [item for item in published if item in set(delivered)]
    assert len(delivered) + dropped == len(published)
    assert all(
        not (left.startswith("gap:") and right.startswith("gap:"))
        for left, right in zip(body, body[1:], strict=False)
    )
