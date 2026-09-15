"""Independent support, preflight, and actual-admission conformance."""

from collections.abc import Awaitable
from typing import TypeVar, cast

import pytest

from ..compatibility import (
    CompatibilityReport,
    CompatibilityStatus,
    IncompatibleRequirementError,
    RequirementUnconfirmedError,
)
from ..diagnostics import TaskDiagnostics
from ..events import TaskState
from ..harness import PersistentSessions
from ..outcome import Completed
from ..requirements import (
    DiagnosticsRequirement,
    SessionRequirements,
    TaskRequest,
    TaskRequirements,
    TextInput,
)
from ..support import (
    Capability,
    ConditionalSupport,
    Support,
    Supported,
    UnknownSupport,
    Unsupported,
)
from ._support import managed, within
from .fixtures import FixtureProfile, FixtureResource, RequirementCase, RuntimeRequirementsFixture

T = TypeVar("T")


class RequirementsConformanceTests:
    def requirement_fixture(self) -> FixtureResource[RuntimeRequirementsFixture]:
        raise NotImplementedError

    async def test_declared_requirements_match_support_validation_and_admission(self) -> None:
        async with managed(self.requirement_fixture()) as initial:
            provider = initial.provider
            profiles = tuple(initial.profiles())
            self._check_profiles(profiles)
        for profile in profiles:
            async with managed(self.requirement_fixture()) as fixture:
                assert fixture.provider == provider
                declared = next(item for item in fixture.profiles() if item.id == profile.id)
                async with fixture.create_harness(profile.id) as harness:
                    assert harness.provider == provider
                    for capability in Capability:
                        self._assert_support(
                            declared.expected_support[capability], harness.support[capability]
                        )
            for expected_case in profile.cases:
                for preflight in (True, False):
                    async with managed(self.requirement_fixture()) as fixture:
                        current_profile = next(
                            item for item in fixture.profiles() if item.id == profile.id
                        )
                        case = next(
                            item for item in current_profile.cases if item.id == expected_case.id
                        )
                        assert case.session_validation == expected_case.session_validation
                        assert case.task_validation == expected_case.task_validation
                        assert case.create_decision == expected_case.create_decision
                        assert case.start_decision == expected_case.start_decision
                        await within(
                            90, self._check_admission(fixture, profile.id, case, preflight)
                        )

    @staticmethod
    def _check_profiles(profiles: tuple[FixtureProfile, ...]) -> None:
        assert profiles
        assert len(profiles) == len({profile.id for profile in profiles})
        for profile in profiles:
            assert set(profile.expected_support.entries) == set(Capability)
            assert any(
                case.session_spec.requirements == SessionRequirements()
                and case.request.requirements == TaskRequirements()
                and case.create_decision is CompatibilityStatus.COMPATIBLE
                and case.start_decision is CompatibilityStatus.COMPATIBLE
                for case in profile.cases
            )

    @staticmethod
    def _assert_support(expected: Support, actual: Support) -> None:
        assert type(actual) is type(expected)
        if isinstance(actual, ConditionalSupport):
            assert actual.scope == cast(ConditionalSupport, expected).scope
            assert actual.condition.strip()
        elif isinstance(actual, (UnknownSupport, Unsupported)):
            assert actual.reason.strip()
        else:
            assert isinstance(actual, Supported)

    async def _check_admission(
        self,
        fixture: RuntimeRequirementsFixture,
        profile_id: str,
        case: RequirementCase,
        preflight: bool,
    ) -> None:
        async with fixture.create_harness(profile_id) as harness:
            before = len(fixture.observation.observed_contexts)
            if preflight:
                assert harness.validate(case.session_spec).status is case.session_validation
            assert len(fixture.observation.observed_contexts) == before
            session = await self._decide(
                case.create_decision, harness.create_session(case.session_spec)
            )
            if session is None:
                assert len(fixture.observation.observed_contexts) == before
                return
            try:
                assert len(fixture.observation.observed_contexts) == before
                if preflight:
                    assert session.validate(case.request).status is case.task_validation
                assert len(fixture.observation.observed_contexts) == before
                task = await self._decide(case.start_decision, session.start_task(case.request))
                if task is None:
                    assert len(fixture.observation.observed_contexts) == before
                    retry = await session.start_task(
                        TaskRequest(TextInput(f"after-rejection-{case.id}"))
                    )
                    assert isinstance(await within(60, retry.await_outcome()), Completed)
                    assert any(
                        f"after-rejection-{case.id}" in value
                        for value in fixture.observation.observed_contexts[before:]
                    )
                else:
                    assert isinstance(await within(60, task.await_outcome()), Completed)
                    assert task.state is TaskState.COMPLETED
                    assert isinstance(case.request.input, TextInput)
                    assert any(
                        case.request.input.text in value
                        for value in fixture.observation.observed_contexts[before:]
                    )
                    if case.session_spec.requirements.persistence is not None:
                        assert isinstance(harness, PersistentSessions)
                        assert session.persistent_ref is not None
                    if (
                        case.session_spec.requirements.diagnostics
                        is DiagnosticsRequirement.REQUIRED
                    ):
                        assert isinstance(task, TaskDiagnostics)
            finally:
                await session.release()

    @staticmethod
    async def _decide(expected: CompatibilityStatus | None, awaitable: Awaitable[T]) -> T | None:
        assert expected is not None
        if expected is CompatibilityStatus.COMPATIBLE:
            return await awaitable
        if expected is CompatibilityStatus.INCOMPATIBLE:
            with pytest.raises(IncompatibleRequirementError) as caught:
                await awaitable
            assert (
                CompatibilityReport(caught.value.issues).status is CompatibilityStatus.INCOMPATIBLE
            )
            return None
        with pytest.raises(RequirementUnconfirmedError) as caught:
            await awaitable
        assert CompatibilityReport(caught.value.issues).status is CompatibilityStatus.UNCONFIRMED
        return None
