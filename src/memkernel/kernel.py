from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from memkernel.backend.backend import (
    MemoryDecision,
    MemoryPolicy,
    MemoryRecord,
    MemoryTier,
    MemoryUsage,
    ScoredMemory,
)
from memkernel.extractor import ExtractedResult, Extractor
from memkernel.provenance import (
    MemorySourceRecord,
    RegexSourceSanitizer,
    SourceEvent,
    SourceRole,
    SourceSanitizer,
    SourceType,
)
from memkernel.retriever_v2 import SemanticRetriever
from memkernel.retriver import RecallResults, Retriever


class KernelBackend(Protocol):
    def remember(
        self,
        extracted: ExtractedResult,
        source_event: SourceEvent,
        *,
        policy: MemoryPolicy | None = None,
    ) -> list[MemoryDecision]: ...

    def get(self, memory_id: str) -> MemoryRecord | None: ...

    def get_history(self, memory_id: str) -> list[MemoryRecord] | None: ...

    def get_usage(self, memory_id: str) -> MemoryUsage | None: ...

    def search_current(self, content: str, top_k: int = 5) -> list[ScoredMemory]: ...

    def search_current_by_tier(
        self,
        content: str,
        *,
        top_k: int = 5,
        tiers: Sequence[MemoryTier],
        reference_time: datetime | str | None = None,
    ) -> list[ScoredMemory]: ...

    def search_history(self, content: str, top_k: int = 5) -> list[ScoredMemory]: ...

    def list_memories(self) -> list[MemoryRecord]: ...

    def get_sources(self, memory_id: str) -> list[MemorySourceRecord]: ...


@dataclass(slots=True, frozen=True)
class PostMemory:
    date: str | None
    content: str
    source_type: SourceType = "message"
    role: SourceRole | None = "user"
    metadata: dict[str, Any] = field(default_factory=dict)
    expires_at: str | None = None
    importance: float = 0.5
    pinned: bool = False
    tier: MemoryTier = "HOT"


class MemKernel:
    """Facade that composes extraction, reconciliation, storage, and recall."""

    def __init__(
        self,
        extractor: Extractor,
        memory_backend: KernelBackend,
        retriever: Retriever | None = None,
        source_sanitizer: SourceSanitizer | None = None,
    ):
        self.extractor = extractor
        self.memory_backend = memory_backend
        self.retriever = retriever or SemanticRetriever(memory_backend)
        self.source_sanitizer = source_sanitizer or RegexSourceSanitizer()

    def remember(self, raw: PostMemory | str) -> list[MemoryDecision]:
        if isinstance(raw, PostMemory):
            content = raw.content
            source_type = raw.source_type
            role = raw.role
            observed_at = self._normalize_observed_at(raw.date)
            metadata = raw.metadata
        else:
            raise Exception("Input format error.")

        if source_type not in {"message", "tool", "document"}:
            raise ValueError("source_type must be message, tool, or document")
        if role not in {None, "user", "assistant", "system", "tool"}:
            raise ValueError("role must be user, assistant, system, tool, or null")

        # safety
        sanitized_content = self.source_sanitizer.sanitize_text(content)
        if not sanitized_content.strip():
            raise ValueError("source content must be a non-empty string")
        sanitized_metadata = self.source_sanitizer.sanitize_metadata(metadata)
        try:
            json.dumps(sanitized_metadata)
        except (TypeError, ValueError) as error:
            raise ValueError("source metadata must be JSON serializable") from error

        source_event = SourceEvent(
            id=str(uuid.uuid4()),
            content=sanitized_content,
            source_type=source_type,
            role=role,
            observed_at=observed_at,
            metadata=sanitized_metadata,
        )
        extracted = self.extractor.extract_with_source(source_event)
        policy = MemoryPolicy(
            tier=raw.tier,
            importance=raw.importance,
            expires_at=self._normalize_optional_timestamp(
                raw.expires_at,
                field_name="expires_at",
            ),
            pinned=raw.pinned,
        )
        # no policy
        if policy == MemoryPolicy():
            policy = None
        return self.memory_backend.remember(
            extracted,
            source_event=source_event,
            policy=policy,
        )

    @staticmethod
    def _normalize_observed_at(value: str | None) -> str:
        if value is None or not value.strip():
            return datetime.now(timezone.utc).isoformat()
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("date must be a valid ISO-8601 timestamp") from error
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _normalize_optional_timestamp(
        value: str | None,
        *,
        field_name: str,
    ) -> str | None:
        if value is None:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as error:
            raise ValueError(
                f"{field_name} must be a valid ISO-8601 timestamp"
            ) from error
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()

    def recall(
        self,
        query: str,
        *,
        current_top_k: int = 5,
        history_top_k: int = 0,
        threshold: float = 0.5,
    ) -> RecallResults:
        return self.retriever.recall(
            query,
            current_top_k=current_top_k,
            history_top_k=history_top_k,
            threshold=threshold,
        )

    def get_history(self, memory_id: str) -> list[MemoryRecord] | None:
        """Return a memory's supersession chain from oldest to newest."""
        return self.memory_backend.get_history(memory_id)

    def list_memories(self) -> list[MemoryRecord]:
        """Return all memories for administrative and debugging views."""
        return self.memory_backend.list_memories()

    def get_sources(self, memory_id: str) -> list[MemorySourceRecord] | None:
        if self.memory_backend.get(memory_id) is None:
            return None
        return self.memory_backend.get_sources(memory_id)

    def get_usage(self, memory_id: str) -> MemoryUsage | None:
        return self.memory_backend.get_usage(memory_id)
