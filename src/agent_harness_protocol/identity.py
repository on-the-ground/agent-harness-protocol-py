"""Opaque identities used by the public port."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _OpaqueId:
    value: str

    def __post_init__(self) -> None:
        if not self.value or self.value.isspace():
            raise ValueError(f"{type(self).__name__} must not be blank")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ProviderId(_OpaqueId):
    """Identifies an adapter kind, not an account or storage namespace."""


@dataclass(frozen=True, slots=True)
class SessionId(_OpaqueId):
    """Identifies a logical context within the harness that issued it."""


@dataclass(frozen=True, slots=True)
class TaskId(_OpaqueId):
    """Identifies one delegated task within its owning harness."""


@dataclass(frozen=True, slots=True)
class WorkId(_OpaqueId):
    """Identifies a tool execution or external effect within one task."""


@dataclass(frozen=True, slots=True)
class InteractionId(_OpaqueId):
    """Identifies one request for an external response within one task."""


@dataclass(frozen=True, slots=True)
class MessageId(_OpaqueId):
    """Connects message deltas to a completed message snapshot."""


@dataclass(frozen=True, slots=True)
class ApprovalScopeId(_OpaqueId):
    """Identifies an enforceable repeated-approval scope within one session."""


@dataclass(frozen=True, slots=True)
class StorageNamespace(_OpaqueId):
    """Identifies a persistent-context store configuration without credentials."""


@dataclass(frozen=True, slots=True)
class PersistentSessionRef:
    """A provider- and storage-qualified reference to persistent context."""

    provider: ProviderId
    namespace: StorageNamespace
    id: str

    def __post_init__(self) -> None:
        if not self.id or self.id.isspace():
            raise ValueError("persistent session id must not be blank")
