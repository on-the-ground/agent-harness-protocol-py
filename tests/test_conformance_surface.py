import inspect

from agent_harness_protocol import AgentHarness, conformance

EXPECTED_SUITES = {
    "AcceptedStartConformanceTests": 1,
    "StartAcceptanceConformanceTests": 4,
    "ResponseAcceptanceConformanceTests": 3,
    "AccountingConformanceTests": 2,
    "AccountingSequenceConformanceTests": 1,
    "ApprovalScopeConformanceTests": 3,
    "CleanupBudgetConformanceTests": 3,
    "ContextConformanceTests": 3,
    "ExecutionConstraintConformanceTests": 3,
    "InteractionConformanceTests": 5,
    "LifecycleConformanceTests": 9,
    "ObservationLoadConformanceTests": 1,
    "OutcomeConformanceTests": 6,
    "PersistenceFailureConformanceTests": 3,
    "RequirementsConformanceTests": 1,
    "RuntimeConformanceTests": 6,
    "RuntimeProfileConformanceTests": 8,
    "RuntimePersistenceConformanceTests": 9,
    "WorkspaceConformanceTests": 2,
}


def test_all_reusable_suites_are_exported_with_expected_scenarios() -> None:
    for name, expected_count in EXPECTED_SUITES.items():
        suite = getattr(conformance, name)
        methods = {
            method_name
            for method_name, _ in inspect.getmembers(suite, inspect.iscoroutinefunction)
            if method_name.startswith("test_")
        }
        assert len(methods) == expected_count, (name, methods)


def test_no_concrete_harness_is_shipped() -> None:
    concrete: list[type[AgentHarness]] = []
    for _, candidate in inspect.getmembers(conformance, inspect.isclass):
        if candidate is AgentHarness:
            continue
        if issubclass(candidate, AgentHarness) and not inspect.isabstract(candidate):
            concrete.append(candidate)
    assert not concrete
