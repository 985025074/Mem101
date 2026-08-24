from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


SourceType = Literal["message", "tool", "document"]
SourceRole = Literal["user", "assistant", "system", "tool"]
SourceLinkType = Literal["DERIVED", "CONFIRMED"]


@dataclass(frozen=True, slots=True)
class SourceEvent:
    id: str
    content: str
    source_type: SourceType
    role: SourceRole | None
    observed_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceEventRecord:
    id: str
    content: str
    source_type: SourceType
    role: SourceRole | None
    observed_at: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemorySourceRecord:
    source: SourceEventRecord
    evidence_quote: str
    link_type: SourceLinkType
    linked_at: str


class SourceSanitizer(Protocol):
    def sanitize_text(self, text: str) -> str: ...

    def sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]: ...


class RegexSourceSanitizer:
    """Best-effort redaction for common secrets before extraction and storage."""

    _PRIVATE_KEY = re.compile(
        r"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----.*?"
        r"-----END (?:[A-Z0-9]+ )?PRIVATE KEY-----",
        flags=re.DOTALL,
    )
    _BEARER_TOKEN = re.compile(
        r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"
    )
    _KNOWN_TOKEN = re.compile(
        r"(?<![A-Za-z0-9])(?:"
        r"sk-[A-Za-z0-9_-]{16,}|"
        r"gh[pousr]_[A-Za-z0-9]{20,}|"
        r"xox[baprs]-[A-Za-z0-9-]{10,}"
        r")(?![A-Za-z0-9])"
    )
    _NAMED_SECRET = re.compile(
        r"(?i)\b("
        r"api[_-]?key|access[_-]?token|auth[_-]?token|password|"
        r"secret(?:[_-]?key)?"
        r")(\s*[:=]\s*)([^\s,;]+)"
    )
    _SENSITIVE_KEYS = frozenset(
        {
            "api_key",
            "apikey",
            "access_token",
            "auth_token",
            "password",
            "secret",
            "secret_key",
            "token",
        }
    )

    def sanitize_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("source content must be a string")

        sanitized = self._PRIVATE_KEY.sub("[REDACTED:PRIVATE_KEY]", text)
        sanitized = self._BEARER_TOKEN.sub(
            "Bearer [REDACTED:TOKEN]",
            sanitized,
        )
        sanitized = self._KNOWN_TOKEN.sub("[REDACTED:TOKEN]", sanitized)
        sanitized = self._NAMED_SECRET.sub(
            lambda match: (
                f"{match.group(1)}{match.group(2)}[REDACTED:SECRET]"
            ),
            sanitized,
        )
        return sanitized

    def sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(metadata, dict):
            raise TypeError("source metadata must be a dictionary")
        return {
            str(key): self._sanitize_metadata_value(str(key), value)
            for key, value in metadata.items()
        }

    def _sanitize_metadata_value(self, key: str, value: Any) -> Any:
        normalized_key = key.casefold().replace("-", "_")
        if normalized_key in self._SENSITIVE_KEYS:
            return "[REDACTED:SECRET]"
        if isinstance(value, str):
            return self.sanitize_text(value)
        if isinstance(value, dict):
            return {
                str(child_key): self._sanitize_metadata_value(
                    str(child_key),
                    child_value,
                )
                for child_key, child_value in value.items()
            }
        if isinstance(value, list):
            return [self._sanitize_metadata_value(key, item) for item in value]
        if isinstance(value, tuple):
            return [self._sanitize_metadata_value(key, item) for item in value]
        return value
