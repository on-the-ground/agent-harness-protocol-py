"""Actual filesystem and network upper-bound scenarios."""

from ..outcome import Completed
from ._support import managed, within
from .fixtures import (
    ExecutionAttempt,
    ExecutionCase,
    ExecutionConstraintFixture,
    FixtureResource,
)


class ExecutionConstraintConformanceTests:
    def execution_fixture(
        self, execution_case: ExecutionCase
    ) -> FixtureResource[ExecutionConstraintFixture]:
        raise NotImplementedError

    async def test_read_only_with_deny_all_exposes_no_actual_write(self) -> None:
        await self._exercise(ExecutionCase.READ_ONLY)

    async def test_workspace_with_denied_network_does_not_exceed_upper_bounds(self) -> None:
        await self._exercise(ExecutionCase.WORKSPACE_DENIED_NETWORK)

    async def test_allowing_network_does_not_widen_filesystem_boundary(self) -> None:
        await self._exercise(ExecutionCase.WORKSPACE_ALLOWED_NETWORK)

    async def _exercise(self, execution_case: ExecutionCase) -> None:
        async with managed(self.execution_fixture(execution_case)) as fixture:
            session = await fixture.harness.create_session(fixture.spec)
            attempts = [
                ExecutionAttempt.WRITE_WORKSPACE,
                ExecutionAttempt.WRITE_ADDITIONAL,
                ExecutionAttempt.WRITE_OUTSIDE,
            ]
            if execution_case is not ExecutionCase.READ_ONLY:
                attempts.append(ExecutionAttempt.USE_NETWORK)
            for attempt in attempts:
                outcome = await within(
                    60, (await session.start_task(fixture.prepare(attempt))).await_outcome()
                )
                assert isinstance(outcome, Completed)
            written = set(fixture.written_targets())
            upper_bound: set[str] = (
                set() if execution_case is ExecutionCase.READ_ONLY else {"workspace", "additional"}
            )
            assert written <= upper_bound
            assert "outside" not in written
            if execution_case is ExecutionCase.WORKSPACE_DENIED_NETWORK:
                assert fixture.network_requests() == 0
