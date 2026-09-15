"""Workspace and skill-activation conformance scenarios."""

import pytest

from ..compatibility import IncompatibleRequirementError
from ..outcome import Completed
from ..requirements import TaskRequest, TextInput
from ._support import managed, within
from .fixtures import FixtureResource, WorkspaceFixture


class WorkspaceConformanceTests:
    def workspace_fixture(self) -> FixtureResource[WorkspaceFixture]:
        raise NotImplementedError

    async def test_workspace_and_active_skill_body_reach_each_native_task(self) -> None:
        async with managed(self.workspace_fixture()) as fixture:
            session = await fixture.harness.create_session(fixture.spec)
            for index in range(2):
                before = len(fixture.observation.observed_contexts)
                request = TaskRequest(TextInput(f"workspace-user-input-{index}"))
                outcome = await within(60, (await session.start_task(request)).await_outcome())
                assert isinstance(outcome, Completed)
                assert isinstance(request.input, TextInput)
                assert request.input.text == f"workspace-user-input-{index}"
                actual = " ".join(fixture.observation.observed_contexts[before:])
                assert fixture.active_skill_body in actual
                assert fixture.inactive_skill_body not in actual
                unescaped = actual.replace("\\\\", "\\")
                assert fixture.working_directory in unescaped
                assert fixture.active_skill_path in unescaped
                assert fixture.active_skill_name in actual
                assert fixture.inactive_skill_path in unescaped
                assert fixture.inactive_skill_name in actual
                assert f"workspace-user-input-{index}" in actual

    async def test_missing_skill_artifact_is_rejected_before_native_work(self) -> None:
        async with managed(self.workspace_fixture()) as fixture:
            assert not fixture.harness.validate(fixture.invalid_spec).is_compatible
            with pytest.raises(IncompatibleRequirementError):
                await fixture.harness.create_session(fixture.invalid_spec)
            assert not fixture.observation.observed_contexts
