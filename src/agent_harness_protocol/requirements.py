"""Session and task requirements expressed in provider-independent terms."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class ContextRetentionRequirement(StrEnum):
    PROVIDER_DEFAULT = "provider_default"
    EPHEMERAL = "ephemeral"


class UserHistoryVisibilityRequirement(StrEnum):
    PROVIDER_DEFAULT = "provider_default"
    HIDDEN = "hidden"


class ApprovalRequirement(StrEnum):
    PROVIDER_DEFAULT = "provider_default"
    DENY_ALL = "deny_all"
    AGENT_REVIEWED = "agent_reviewed"
    CALLER_DECIDES = "caller_decides"


class QuestionRequirement(StrEnum):
    NOT_REQUIRED = "not_required"
    CALLER_ANSWERS = "caller_answers"


class DiagnosticsRequirement(StrEnum):
    NOT_REQUIRED = "not_required"
    REQUIRED = "required"


@dataclass(frozen=True, slots=True)
class PersistenceRequirement:
    """Required persistence scope. ``None`` means persistence is not required."""

    across_harness_restart: bool = True
    across_process_restart: bool = False
    concurrent_access: bool = False


@dataclass(frozen=True, slots=True)
class SkillReference:
    name: str
    path: str
    activate: bool = True

    def __post_init__(self) -> None:
        if not self.name or self.name.isspace():
            raise ValueError("skill name must not be blank")
        if not self.path or self.path.isspace():
            raise ValueError("skill path must not be blank")


@dataclass(frozen=True, slots=True)
class WorkspaceRequirement:
    """Required workspace and skills. ``None`` means no workspace guarantee is required."""

    working_directory: str | None = None
    skills: tuple[SkillReference, ...] = ()

    def __init__(
        self,
        working_directory: str | None = None,
        skills: Iterable[SkillReference] = (),
    ) -> None:
        object.__setattr__(self, "working_directory", working_directory)
        object.__setattr__(self, "skills", tuple(skills))


class FilesystemAccess:
    """Base class for an enforceable filesystem upper bound."""


@dataclass(frozen=True, slots=True)
class ReadOnly(FilesystemAccess):
    pass


@dataclass(frozen=True, slots=True)
class WorkspaceWrite(FilesystemAccess):
    additional_writable_roots: frozenset[str] = frozenset()

    def __init__(self, additional_writable_roots: Iterable[str] = ()) -> None:
        object.__setattr__(self, "additional_writable_roots", frozenset(additional_writable_roots))


@dataclass(frozen=True, slots=True)
class FullAccess(FilesystemAccess):
    pass


READ_ONLY = ReadOnly()
FULL_ACCESS = FullAccess()


class NetworkAccess(StrEnum):
    DENIED = "denied"
    ALLOWED = "allowed"


@dataclass(frozen=True, slots=True)
class ExecutionConstraint:
    """An execution upper bound. ``None`` means use the provider's current policy."""

    filesystem: FilesystemAccess | None = None
    network: NetworkAccess | None = None

    def __post_init__(self) -> None:
        if self.filesystem is None and self.network is None:
            raise ValueError("at least one execution constraint must be required")


class OutputRequirement:
    """Base class for the required task-output shape."""


@dataclass(frozen=True, slots=True)
class TextOutputRequirement(OutputRequirement):
    pass


@dataclass(frozen=True, slots=True)
class StructuredOutputRequirement(OutputRequirement):
    schema: str
    validated_by_harness: bool = True

    def __post_init__(self) -> None:
        if not self.schema or self.schema.isspace():
            raise ValueError("schema must not be blank")


TEXT_OUTPUT = TextOutputRequirement()


class TaskInput:
    """Base class for caller input, distinct from an interaction response."""


@dataclass(frozen=True, slots=True)
class TextInput(TaskInput):
    text: str

    def __post_init__(self) -> None:
        if self.text == "":
            raise ValueError("input text must not be empty")


@dataclass(frozen=True, slots=True)
class TaskRequirements:
    output: OutputRequirement = TEXT_OUTPUT


@dataclass(frozen=True, slots=True)
class TaskRequest:
    input: TaskInput
    requirements: TaskRequirements = TaskRequirements()


@dataclass(frozen=True, slots=True)
class SessionRequirements:
    approval: ApprovalRequirement = ApprovalRequirement.PROVIDER_DEFAULT
    questions: QuestionRequirement = QuestionRequirement.NOT_REQUIRED
    persistence: PersistenceRequirement | None = None
    workspace: WorkspaceRequirement | None = None
    execution: ExecutionConstraint | None = None
    diagnostics: DiagnosticsRequirement = DiagnosticsRequirement.NOT_REQUIRED
    retention: ContextRetentionRequirement = ContextRetentionRequirement.PROVIDER_DEFAULT
    history_visibility: UserHistoryVisibilityRequirement = (
        UserHistoryVisibilityRequirement.PROVIDER_DEFAULT
    )


@dataclass(frozen=True, slots=True)
class SessionSpec:
    """Configuration and required guarantees for one logical context."""

    instructions: str | None = None
    model: str | None = None
    requirements: SessionRequirements = SessionRequirements()
