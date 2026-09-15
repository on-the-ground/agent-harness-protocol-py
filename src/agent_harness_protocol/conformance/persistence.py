"""Persistence configuration and storage-failure scenarios."""

from dataclasses import replace

import pytest

from ..compatibility import HarnessTransportError, IncompatibleRequirementError
from ..harness import PersistentSessions
from ..outcome import Completed
from ..requirements import TaskRequest, TextInput
from ._support import managed, within
from .fixtures import FixtureResource, PersistenceFailureFixture


class PersistenceFailureConformanceTests:
    def persistence_fixture(self) -> FixtureResource[PersistenceFailureFixture]:
        raise NotImplementedError

    @staticmethod
    def _request(value: str) -> TaskRequest:
        return TaskRequest(TextInput(value))

    async def test_changed_configuration_cannot_replace_live_handle_configuration(self) -> None:
        async with managed(self.persistence_fixture()) as fixture:
            original = await fixture.harness.create_session(fixture.spec)
            assert isinstance(
                await within(
                    60,
                    (
                        await original.start_task(self._request("before-config-change"))
                    ).await_outcome(),
                ),
                Completed,
            )
            reference = original.persistent_ref
            assert reference is not None
            desired = replace(fixture.spec, instructions="AHP_CHANGED_CONFIGURATION")
            assert isinstance(fixture.harness, PersistentSessions)
            with pytest.raises(IncompatibleRequirementError):
                await fixture.harness.reopen_session(reference, desired)
            before = len(fixture.observation.observed_contexts)
            assert isinstance(
                await within(
                    60,
                    (
                        await original.start_task(self._request("original-config-still-live"))
                    ).await_outcome(),
                ),
                Completed,
            )
            actual = " ".join(fixture.observation.observed_contexts[before:])
            assert fixture.spec.instructions is not None
            assert fixture.spec.instructions in actual
            assert "AHP_CHANGED_CONFIGURATION" not in actual
            await original.release()
            if fixture.supports_changed_instructions:
                updated = await fixture.harness.reopen_session(reference, desired)
                after = len(fixture.observation.observed_contexts)
                assert isinstance(
                    await within(
                        60,
                        (
                            await updated.start_task(self._request("updated-after-release"))
                        ).await_outcome(),
                    ),
                    Completed,
                )
                assert any(
                    "AHP_CHANGED_CONFIGURATION" in value
                    for value in fixture.observation.observed_contexts[after:]
                )
            else:
                with pytest.raises(IncompatibleRequirementError):
                    await fixture.harness.reopen_session(reference, desired)

    async def test_unknown_reference_is_rejected_without_replacement_context(self) -> None:
        async with managed(self.persistence_fixture()) as fixture:
            session = await fixture.harness.create_session(fixture.spec)
            assert isinstance(
                await within(
                    60, (await session.start_task(self._request("known-context"))).await_outcome()
                ),
                Completed,
            )
            reference = session.persistent_ref
            assert reference is not None
            before = len(fixture.observation.observed_contexts)
            assert isinstance(fixture.harness, PersistentSessions)
            with pytest.raises((HarnessTransportError, IncompatibleRequirementError)):
                await fixture.harness.reopen_session(
                    replace(reference, id="00000000-0000-0000-0000-000000000001"),
                    fixture.spec,
                )
            assert len(fixture.observation.observed_contexts) == before
            assert isinstance(
                await within(
                    60,
                    (await session.start_task(self._request("known-still-usable"))).await_outcome(),
                ),
                Completed,
            )

    async def test_unavailable_history_fails_reopen_and_restoration_recovers_context(self) -> None:
        async with managed(self.persistence_fixture()) as fixture:
            original = await fixture.harness.create_session(fixture.spec)
            assert isinstance(
                await within(
                    60,
                    (
                        await original.start_task(self._request("restore-original-marker"))
                    ).await_outcome(),
                ),
                Completed,
            )
            reference = original.persistent_ref
            assert reference is not None
            await original.release()
            await fixture.harness.aclose()
            fixture.hide_stored_context()
            async with fixture.recreate_harness() as unavailable:
                before = len(fixture.observation.observed_contexts)
                assert isinstance(unavailable, PersistentSessions)
                with pytest.raises(HarnessTransportError):
                    await unavailable.reopen_session(reference, fixture.spec)
                assert len(fixture.observation.observed_contexts) == before
            fixture.restore_stored_context()
            async with fixture.recreate_harness() as restored:
                assert isinstance(restored, PersistentSessions)
                session = await restored.reopen_session(reference, fixture.spec)
                before = len(fixture.observation.observed_contexts)
                assert isinstance(
                    await within(
                        60,
                        (
                            await session.start_task(self._request("after-storage-restored"))
                        ).await_outcome(),
                    ),
                    Completed,
                )
                assert any(
                    "restore-original-marker" in value
                    for value in fixture.observation.observed_contexts[before:]
                )
                assert session.persistent_ref == reference
