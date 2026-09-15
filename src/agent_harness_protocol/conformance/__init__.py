"""Reusable pytest conformance suites and their real-boundary fixture seams."""

from .acceptance import (
    AcceptedStartConformanceTests,
    ResponseAcceptanceConformanceTests,
    StartAcceptanceConformanceTests,
)
from .accounting import AccountingConformanceTests, AccountingSequenceConformanceTests
from .approval import ApprovalScopeConformanceTests
from .cleanup import CleanupBudgetConformanceTests
from .context import ContextConformanceTests
from .execution import ExecutionConstraintConformanceTests
from .fixtures import (
    AcceptanceFixture,
    AccountingFixture,
    ContextFixture,
    ExecutionAttempt,
    ExecutionCase,
    ExecutionConstraintFixture,
    FixtureFactory,
    FixtureProfile,
    FixtureResource,
    InteractionRaceFixture,
    LifecycleFixture,
    MessageKind,
    OutcomeFixture,
    OutputCase,
    OutputObservation,
    PersistenceFailureFixture,
    ProfileFixture,
    RepeatedApprovalFixture,
    RequirementCase,
    ResponseAcceptanceFixture,
    ResponseControl,
    RuntimeObservation,
    RuntimeRequirementsFixture,
    StartAcceptanceFixture,
    StartControl,
    StructuredObservation,
    TaskLifecycleControl,
    TextObservation,
    WorkspaceFixture,
)
from .interaction import InteractionConformanceTests
from .lifecycle import LifecycleConformanceTests
from .observation import ObservationLoadConformanceTests
from .outcome import OutcomeConformanceTests
from .persistence import PersistenceFailureConformanceTests
from .requirements import RequirementsConformanceTests
from .runtime import (
    RuntimeConformanceTests,
    RuntimePersistenceConformanceTests,
    RuntimeProfileConformanceTests,
)
from .workspace import WorkspaceConformanceTests

__all__ = [name for name in globals() if not name.startswith("_")]
