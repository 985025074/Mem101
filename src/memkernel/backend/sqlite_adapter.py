import json
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, cast

import sqlite_vec

from memkernel.backend.backend import (
    MemoryDecision,
    MemoryPolicy,
    MemoryRecord,
    MemoryState,
    MemoryTier,
    MemoryUsage,
    ScoredMemory,
)
from memkernel.embedding import EmbeddingProvider
from memkernel.extractor import ExtractedResult
from memkernel.provenance import (
    MemorySourceRecord,
    SourceEvent,
    SourceEventRecord,
    SourceLinkType,
    SourceRole,
    SourceType,
)


class SQLiteBackend:
    _QUERYABLE_COLUMNS = frozenset(
        {
            "id",
            "content",
            "created_at",
            "state",
            "superseded_by_id",
            "superseded_at",
            "expires_at",
        }
    )

    def __init__(
        self,
        database_path: str | Path = "memkernel.db",
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self.database_path = Path(database_path)
        self.embedding_provider = embedding_provider
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        # semantic search extenstion
        if self.embedding_provider is not None:
            connection.enable_load_extension(True)
            try:
                sqlite_vec.load(connection)
            finally:
                connection.enable_load_extension(False)

        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            # This can accelerate
            connection.execute("PRAGMA journal_mode = WAL")
            # safe for concurrency
            connection.execute("BEGIN IMMEDIATE")
            # Create Table
            # TODO: We may need more tables
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    state TEXT NOT NULL DEFAULT 'ACTIVE'
                        CHECK (state IN ('ACTIVE', 'SUPERSEDED')),
                    superseded_by_id TEXT,
                    superseded_at TEXT,
                    expires_at TEXT,
                    FOREIGN KEY (superseded_by_id) REFERENCES memories(id)
                        ON DELETE SET NULL
                )
                """
            )
            # add state and  others
            self._migrate_memory_lifecycle(connection)
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_state
                ON memories(state)
                """
            )
            # add table of memory seamantic embedding queried by memory id
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id TEXT PRIMARY KEY,
                    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
                    embedding BLOB NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memories(id)
                        ON DELETE CASCADE
                )
                """
            )
            # Add memory's hierachical level table
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_usage (
                    memory_id TEXT PRIMARY KEY,
                    tier TEXT NOT NULL DEFAULT 'HOT'
                        CHECK (tier IN ('HOT', 'WARM', 'COLD')),
                    importance REAL NOT NULL DEFAULT 0.5
                        CHECK (importance >= 0.0 AND importance <= 1.0),
                    last_accessed_at TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0
                        CHECK (access_count >= 0),
                    last_confirmed_at TEXT,
                    confirmation_count INTEGER NOT NULL DEFAULT 0
                        CHECK (confirmation_count >= 0),
                    pinned INTEGER NOT NULL DEFAULT 0
                        CHECK (pinned IN (0, 1)),
                    FOREIGN KEY (memory_id) REFERENCES memories(id)
                        ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO memory_usage (memory_id, tier)
                SELECT
                    id,
                    CASE state WHEN 'SUPERSEDED' THEN 'WARM' ELSE 'HOT' END
                FROM memories
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_usage_tier
                ON memory_usage(tier)
                """
            )
            # source  event tables
            # Observe at is only used for some old things.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS source_events (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source_type TEXT NOT NULL
                        CHECK (source_type IN ('message', 'tool', 'document')),
                    role TEXT
                        CHECK (
                            role IS NULL OR
                            role IN ('user', 'assistant', 'system', 'tool')
                        ),
                    observed_at TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            # one memory can have many events so we need an big table ,not only a column
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_sources (
                    memory_id TEXT NOT NULL,
                    source_event_id TEXT NOT NULL,
                    evidence_quote TEXT NOT NULL,
                    link_type TEXT NOT NULL
                        CHECK (link_type IN ('DERIVED', 'CONFIRMED')),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (memory_id, source_event_id),
                    FOREIGN KEY (memory_id) REFERENCES memories(id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (source_event_id) REFERENCES source_events(id)
                        ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_sources_source
                ON memory_sources(source_event_id)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_source_events_observed_at
                ON source_events(observed_at)
                """
            )

    @staticmethod
    def _migrate_memory_lifecycle(connection: sqlite3.Connection) -> None:
        """Add lifecycle columns to databases created by earlier versions."""
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(memories)").fetchall()
        }
        if "state" not in columns:
            connection.execute(
                """
                ALTER TABLE memories
                ADD COLUMN state TEXT NOT NULL DEFAULT 'ACTIVE'
                    CHECK (state IN ('ACTIVE', 'SUPERSEDED'))
                """
            )
        if "superseded_by_id" not in columns:
            connection.execute(
                """
                ALTER TABLE memories
                ADD COLUMN superseded_by_id TEXT
                    REFERENCES memories(id) ON DELETE SET NULL
                """
            )
        if "superseded_at" not in columns:
            connection.execute(
                """
                ALTER TABLE memories
                ADD COLUMN superseded_at TEXT
                """
            )
        if "expires_at" not in columns:
            connection.execute(
                """
                ALTER TABLE memories
                ADD COLUMN expires_at TEXT
                """
            )

    @staticmethod
    def _memory_from_row(row: sqlite3.Row) -> MemoryRecord:
        """Change a sqlite row to our memory record"""
        return MemoryRecord(
            id=row["id"],
            content=row["content"],
            created_at=row["created_at"],
            state=cast(MemoryState, row["state"]),
            superseded_by_id=row["superseded_by_id"],
            superseded_at=row["superseded_at"],
            expires_at=row["expires_at"],
        )

    @staticmethod
    def _usage_from_row(row: sqlite3.Row) -> MemoryUsage:
        tier = row["tier"] if row["tier"] is not None else "HOT"
        return MemoryUsage(
            tier=cast(MemoryTier, tier),
            importance=float(
                row["importance"] if row["importance"] is not None else 0.5
            ),
            last_accessed_at=row["last_accessed_at"],
            access_count=int(
                row["access_count"] if row["access_count"] is not None else 0
            ),
            last_confirmed_at=row["last_confirmed_at"],
            confirmation_count=int(
                row["confirmation_count"]
                if row["confirmation_count"] is not None
                else 0
            ),
            pinned=bool(row["pinned"] if row["pinned"] is not None else 0),
        )

    @staticmethod
    def _validate_content(content: str) -> str:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        return content.strip()

    @staticmethod
    def _normalize_timestamp(value: str | None) -> str | None:
        if value is None:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as error:
            raise ValueError("expires_at must be a valid ISO-8601 timestamp") from error
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()

    @classmethod
    def _normalize_policy(cls, policy: MemoryPolicy | None) -> MemoryPolicy:
        selected = policy or MemoryPolicy()
        return MemoryPolicy(
            tier=selected.tier,
            importance=float(selected.importance),
            expires_at=cls._normalize_timestamp(selected.expires_at),
            pinned=selected.pinned,
        )

    @staticmethod
    def _insert_memory_row(
        connection: sqlite3.Connection,
        *,
        memory_id: str,
        content: str,
        embedding: list[float] | None,
        policy: MemoryPolicy,
    ) -> None:
        connection.execute(
            """
            INSERT INTO memories (id, content, expires_at)
            VALUES (?, ?, ?)
            """,
            (memory_id, content, policy.expires_at),
        )
        if embedding is not None:
            connection.execute(
                """
                INSERT INTO memory_embeddings (
                    memory_id, dimensions, embedding
                )
                VALUES (?, ?, ?)
                """,
                (
                    memory_id,
                    len(embedding),
                    sqlite_vec.serialize_float32(embedding),
                ),
            )
        connection.execute(
            """
            INSERT INTO memory_usage (
                memory_id,
                tier,
                importance,
                last_confirmed_at,
                confirmation_count,
                pinned
            )
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, 1, ?)
            """,
            (
                memory_id,
                policy.tier,
                policy.importance,
                int(policy.pinned),
            ),
        )

    def _create_document_embedding(self, content: str) -> list[float]:
        """Embed content that will be stored and searched as a document."""
        if self.embedding_provider is None:
            raise RuntimeError("Semantic search requires an embedding_provider")

        embedding = [
            float(value) for value in self.embedding_provider.embed_document(content)
        ]

        return embedding

    def _create_query_embedding(self, content: str) -> list[float]:
        """Embed content that will be used to search stored documents."""
        if self.embedding_provider is None:
            raise RuntimeError("Semantic search requires an embedding_provider")

        return [float(value) for value in self.embedding_provider.embed_query(content)]

    def insert(
        self,
        content: str,
        *,
        policy: MemoryPolicy | None = None,
    ) -> str:
        content = self._validate_content(content)
        normalized_policy = self._normalize_policy(policy)
        memory_id = str(uuid.uuid4())
        embedding = None
        if self.embedding_provider is not None and normalized_policy.tier != "COLD":
            embedding = self._create_document_embedding(content)

        with self._connect() as connection:
            self._insert_memory_row(
                connection,
                memory_id=memory_id,
                content=content,
                embedding=embedding,
                policy=normalized_policy,
            )

        return memory_id

    def supersede(
        self,
        memory_id: str,
        content: str,
        *,
        policy: MemoryPolicy | None = None,
    ) -> str:
        """Atomically replace an active memory while preserving its history."""
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id must be a non-empty string")
        content = self._validate_content(content)
        normalized_policy = self._normalize_policy(policy)

        new_memory_id = str(uuid.uuid4())
        embedding = None
        if self.embedding_provider is not None and normalized_policy.tier != "COLD":
            embedding = self._create_document_embedding(content)

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT state FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
            if existing is None:
                raise ValueError(f"Memory with id {memory_id} was not found")
            if existing["state"] != "ACTIVE":
                raise ValueError("Only an active memory can be superseded")

            self._insert_memory_row(
                connection,
                memory_id=new_memory_id,
                content=content,
                embedding=embedding,
                policy=normalized_policy,
            )

            # insertion finished.Then lets make the old one outdated
            cursor = connection.execute(
                """
                UPDATE memories
                SET state = 'SUPERSEDED',
                    superseded_by_id = ?,
                    superseded_at = CURRENT_TIMESTAMP
                WHERE id = ? AND state = 'ACTIVE'
                """,
                (new_memory_id, memory_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Memory state changed while it was being superseded")
            connection.execute(
                """
                UPDATE memory_usage
                SET tier = CASE
                    WHEN EXISTS (
                        SELECT 1
                        FROM memory_embeddings
                        WHERE memory_id = ?
                    ) THEN 'WARM'
                    ELSE 'COLD'
                END
                WHERE memory_id = ?
                """,
                (memory_id, memory_id),
            )

        return new_memory_id

    def apply_decisions(
        self,
        source_event: SourceEvent,
        changes: Sequence[tuple[MemoryDecision, str]],
        *,
        policy: MemoryPolicy | None = None,
    ) -> list[MemoryDecision]:
        """Atomically persist one source event and all memory decisions it caused."""
        if not changes:
            return []
        normalized_policy = self._normalize_policy(policy)

        metadata_json = json.dumps(
            source_event.metadata,
            ensure_ascii=False,
            sort_keys=True,
        )
        prepared: list[tuple[MemoryDecision, str, str, list[float] | None]] = []
        for decision, evidence_quote in changes:
            embedding: list[float] | None = None
            if decision.action in {"ADD", "SUPERSEDE"}:
                memory_id = str(uuid.uuid4())
                if (
                    self.embedding_provider is not None
                    and normalized_policy.tier != "COLD"
                ):
                    embedding = self._create_document_embedding(decision.fact)
            else:
                memory_id = cast(str, decision.matched_memory_id)
            prepared.append((decision, evidence_quote, memory_id, embedding))

        completed: list[MemoryDecision] = []
        # recording the supersede in the batch
        latest_batch_replacement: dict[str, str] = {}
        # add source events
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO source_events (
                    id,
                    content,
                    source_type,
                    role,
                    observed_at,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    source_event.id,
                    source_event.content,
                    source_event.source_type,
                    source_event.role,
                    source_event.observed_at,
                    metadata_json,
                ),
            )

            # handle decision
            for decision, evidence_quote, memory_id, embedding in prepared:
                link_type: SourceLinkType
                linked_memory_id = memory_id
                if decision.action == "ADD":
                    self._insert_memory_row(
                        connection,
                        memory_id=memory_id,
                        content=decision.fact,
                        embedding=embedding,
                        policy=normalized_policy,
                    )
                    completed_decision = replace(
                        decision,
                        memory_id=memory_id,
                    )
                    link_type = "DERIVED"
                elif decision.action == "NOOP":
                    # confined the new memory's id if possible
                    confirmed_memory_id = latest_batch_replacement.get(
                        memory_id,
                        memory_id,
                    )
                    matched = connection.execute(
                        "SELECT state FROM memories WHERE id = ?",
                        (confirmed_memory_id,),
                    ).fetchone()
                    # NOOP is a repetion of the existing memory
                    if matched is None or matched["state"] != "ACTIVE":
                        raise RuntimeError("NOOP target changed after reconciliation")
                    completed_decision = replace(
                        decision,
                        memory_id=confirmed_memory_id,
                        matched_memory_id=confirmed_memory_id,
                    )
                    linked_memory_id = confirmed_memory_id
                    link_type = "CONFIRMED"
                elif decision.action == "SUPERSEDE":
                    # TODO: This logic may be not  right.
                    # the request_id we want (for example we want outdate A),it may be outdated by the previous decision
                    # we can get that by latest batch replacement dict
                    # and the newest supersede should outdate that!
                    requested_target_id = cast(str, decision.matched_memory_id)
                    actual_target_id = latest_batch_replacement.get(
                        requested_target_id,
                        requested_target_id,
                    )
                    self._insert_memory_row(
                        connection,
                        memory_id=memory_id,
                        content=decision.fact,
                        embedding=embedding,
                        policy=normalized_policy,
                    )
                    cursor = connection.execute(
                        """
                        UPDATE memories
                        SET state = 'SUPERSEDED',
                            superseded_by_id = ?,
                            superseded_at = CURRENT_TIMESTAMP
                        WHERE id = ? AND state = 'ACTIVE'
                        """,
                        (memory_id, actual_target_id),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError(
                            "SUPERSEDE target changed after reconciliation"
                        )
                    completed_decision = replace(
                        decision,
                        memory_id=memory_id,
                        matched_memory_id=actual_target_id,
                    )
                    latest_batch_replacement[requested_target_id] = memory_id
                    latest_batch_replacement[actual_target_id] = memory_id
                    connection.execute(
                        """
                        UPDATE memory_usage
                        SET tier = CASE
                            WHEN EXISTS (
                                SELECT 1
                                FROM memory_embeddings
                                WHERE memory_id = ?
                            ) THEN 'WARM'
                            ELSE 'COLD'
                        END
                        WHERE memory_id = ?
                        """,
                        (actual_target_id, actual_target_id),
                    )
                    link_type = "DERIVED"
                else:
                    raise ValueError(f"Unsupported memory action: {decision.action}")

                # add memory to event link
                link_cursor = connection.execute(
                    """
                    INSERT INTO memory_sources (
                        memory_id,
                        source_event_id,
                        evidence_quote,
                        link_type
                    )
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (memory_id, source_event_id) DO NOTHING
                    """,
                    (
                        linked_memory_id,
                        source_event.id,
                        evidence_quote,
                        link_type,
                    ),
                )
                if decision.action == "NOOP" and link_cursor.rowcount == 1:
                    connection.execute(
                        """
                        UPDATE memory_usage
                        SET last_confirmed_at = CURRENT_TIMESTAMP,
                            confirmation_count = confirmation_count + 1,
                            tier = CASE
                                WHEN tier = 'WARM' THEN 'HOT'
                                ELSE tier
                            END
                        WHERE memory_id = ?
                        """,
                        (linked_memory_id,),
                    )
                completed.append(completed_decision)

        return completed

    def remember(
        self,
        extracted: ExtractedResult,
        source_event: SourceEvent | None = None,
        *,
        policy: MemoryPolicy | None = None,
    ) -> str:
        if source_event is not None:
            raise TypeError(
                "SQLiteBackend cannot reconcile sourced extracted results directly"
            )
        return self.insert(extracted.content, policy=policy)

    def get(self, memory_id: str) -> MemoryRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    content,
                    created_at,
                    state,
                    superseded_by_id,
                    superseded_at,
                    expires_at
                FROM memories
                WHERE id = ?
                """,
                (memory_id,),
            ).fetchone()

        if row is None:
            return None

        return self._memory_from_row(row)

    def get_history(self, memory_id: str) -> list[MemoryRecord] | None:
        """Return one supersession chain without loading the full memory table."""
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id must be a non-empty string")
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
            if exists is None:
                return None
            rows = connection.execute(
                """
                WITH RECURSIVE
                ancestors(id, depth) AS (
                    SELECT id, 0
                    FROM memories
                    WHERE id = ?
                    UNION ALL
                    SELECT predecessor.id, ancestors.depth + 1
                    FROM memories AS predecessor
                    JOIN ancestors
                      ON predecessor.superseded_by_id = ancestors.id
                ),
                oldest(id) AS (
                    SELECT id
                    FROM ancestors
                    ORDER BY depth DESC
                    LIMIT 1
                ),
                chain(
                    id,
                    content,
                    created_at,
                    state,
                    superseded_by_id,
                    superseded_at,
                    expires_at,
                    depth
                ) AS (
                    SELECT
                        m.id,
                        m.content,
                        m.created_at,
                        m.state,
                        m.superseded_by_id,
                        m.superseded_at,
                        m.expires_at,
                        0
                    FROM memories AS m
                    JOIN oldest ON oldest.id = m.id
                    UNION ALL
                    SELECT
                        successor.id,
                        successor.content,
                        successor.created_at,
                        successor.state,
                        successor.superseded_by_id,
                        successor.superseded_at,
                        successor.expires_at,
                        chain.depth + 1
                    FROM memories AS successor
                    JOIN chain ON chain.superseded_by_id = successor.id
                )
                SELECT
                    id,
                    content,
                    created_at,
                    state,
                    superseded_by_id,
                    superseded_at,
                    expires_at
                FROM chain
                ORDER BY depth ASC
                """,
                (memory_id,),
            ).fetchall()
        return [self._memory_from_row(row) for row in rows]

    def get_sources(self, memory_id: str) -> list[MemorySourceRecord]:
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id must be a non-empty string")

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    s.id,
                    s.content,
                    s.source_type,
                    s.role,
                    s.observed_at,
                    s.created_at AS source_created_at,
                    s.metadata_json,
                    ms.evidence_quote,
                    ms.link_type,
                    ms.created_at AS linked_at
                FROM memory_sources AS ms
                JOIN source_events AS s ON s.id = ms.source_event_id
                WHERE ms.memory_id = ?
                ORDER BY s.observed_at ASC, s.created_at ASC
                """,
                (memory_id,),
            ).fetchall()

        records: list[MemorySourceRecord] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            if not isinstance(metadata, dict):
                raise RuntimeError("Stored source metadata must be a JSON object")
            records.append(
                MemorySourceRecord(
                    source=SourceEventRecord(
                        id=row["id"],
                        content=row["content"],
                        source_type=cast(SourceType, row["source_type"]),
                        role=cast(SourceRole | None, row["role"]),
                        observed_at=row["observed_at"],
                        created_at=row["source_created_at"],
                        metadata=metadata,
                    ),
                    evidence_quote=row["evidence_quote"],
                    link_type=cast(SourceLinkType, row["link_type"]),
                    linked_at=row["linked_at"],
                )
            )
        return records

    def remove(self, memory_id: str) -> bool:
        """Remove a memory and its embedding, returning whether it existed."""
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id must be a non-empty string")

        # embedding uses foreign key,We don't need to delete again
        with self._connect() as connection:
            source_rows = connection.execute(
                """
                SELECT source_event_id
                FROM memory_sources
                WHERE memory_id = ?
                """,
                (memory_id,),
            ).fetchall()
            cursor = connection.execute(
                "DELETE FROM memories WHERE id = ?",
                (memory_id,),
            )
            # remove  memory related evenets
            if cursor.rowcount > 0:
                for row in source_rows:
                    connection.execute(
                        """
                        DELETE FROM source_events
                        WHERE id = ?
                          AND NOT EXISTS (
                              SELECT 1
                              FROM memory_sources
                              WHERE source_event_id = ?
                          )
                        """,
                        (row["source_event_id"], row["source_event_id"]),
                    )

        return cursor.rowcount > 0

    def query_by(self, query_dict: Dict[str, Any]) -> MemoryRecord | None:
        """Return the newest memory that exactly matches every supplied field."""
        if not query_dict:
            raise ValueError("query_dict must contain at least one field")

        invalid_columns = set(query_dict) - self._QUERYABLE_COLUMNS
        if invalid_columns:
            invalid = ", ".join(sorted(invalid_columns))
            allowed = ", ".join(sorted(self._QUERYABLE_COLUMNS))
            raise ValueError(
                f"Unsupported query column(s): {invalid}. Allowed columns: {allowed}"
            )

        conditions: list[str] = []
        values: list[Any] = []
        for key, value in query_dict.items():
            if value is None:
                conditions.append(f"{key} IS NULL")
            else:
                conditions.append(f"{key} = ?")
                values.append(value)

        where_clause = " AND ".join(conditions)
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT
                    id,
                    content,
                    created_at,
                    state,
                    superseded_by_id,
                    superseded_at,
                    expires_at
                FROM memories
                WHERE {where_clause}
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                values,
            ).fetchone()

        if row is None:
            return None

        return self._memory_from_row(row)

    def search_similar(self, content: str, top_k: int = 5) -> list[ScoredMemory]:
        """Search current memories. Kept as a backward-compatible alias."""
        return self.search_current(content, top_k=top_k)

    def search_current(
        self,
        content: str,
        top_k: int = 5,
        *,
        reference_time: datetime | str | None = None,
    ) -> list[ScoredMemory]:
        """Search every searchable active tier(HOT,WARM) for reconciliation callers."""
        return self._search_by_state(
            content,
            top_k=top_k,
            state="ACTIVE",
            tiers=("HOT", "WARM"),
            reference_time=reference_time,
        )

    def search_current_by_tier(
        self,
        content: str,
        *,
        top_k: int = 5,
        tiers: Sequence[MemoryTier],
        reference_time: datetime | str | None = None,
    ) -> list[ScoredMemory]:
        """Search in specified tiers"""
        return self._search_by_state(
            content,
            top_k=top_k,
            state="ACTIVE",
            tiers=tiers,
            reference_time=reference_time,
        )

    def search_history(
        self,
        content: str,
        top_k: int = 5,
        *,
        reference_time: datetime | str | None = None,
    ) -> list[ScoredMemory]:
        return self._search_by_state(
            content,
            top_k=top_k,
            state="SUPERSEDED",
            tiers=("HOT", "WARM"),
            reference_time=reference_time,
        )

    @classmethod
    def _reference_timestamp(
        cls,
        value: datetime | str | None,
    ) -> str:
        if value is None:
            parsed = datetime.now(timezone.utc)
        elif isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str):
            normalized = cls._normalize_timestamp(value)
            if normalized is None:
                raise ValueError("reference_time must not be null")
            return normalized
        else:
            raise ValueError("reference_time must be a datetime or ISO-8601 string")
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()

    def _search_by_state(
        self,
        content: str,
        *,
        top_k: int,
        state: MemoryState,
        tiers: Sequence[MemoryTier] | None = None,
        reference_time: datetime | str | None = None,
    ) -> list[ScoredMemory]:
        content = self._validate_content(content)

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        query_embedding = self._create_query_embedding(content)
        serialized_query = sqlite_vec.serialize_float32(query_embedding)
        normalized_tiers: tuple[MemoryTier, ...] | None = None
        # check tiers
        if tiers is not None:
            normalized_tiers = tuple(dict.fromkeys(tiers))
            if not normalized_tiers:
                return []
            if any(tier not in {"HOT", "WARM", "COLD"} for tier in normalized_tiers):
                raise ValueError("tiers must contain only HOT, WARM, or COLD")

        tier_clause = ""
        parameters: list[Any] = [serialized_query, len(query_embedding), state]
        # COmbine tier in the parametres,which will be passed to SQL
        if normalized_tiers is not None:
            placeholders = ", ".join("?" for _ in normalized_tiers)
            tier_clause = f" AND u.tier IN ({placeholders})"
            parameters.extend(normalized_tiers)
        parameters.extend([self._reference_timestamp(reference_time), top_k])

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    m.id,
                    m.content,
                    m.created_at,
                    m.state,
                    m.superseded_by_id,
                    m.superseded_at,
                    m.expires_at,
                    u.tier,
                    u.importance,
                    u.last_accessed_at,
                    u.access_count,
                    u.last_confirmed_at,
                    u.confirmation_count,
                    u.pinned,
                    vec_distance_cosine(e.embedding, ?) AS distance
                FROM memory_embeddings AS e
                JOIN memories AS m ON m.id = e.memory_id
                JOIN memory_usage AS u ON u.memory_id = m.id
                WHERE e.dimensions = ? AND m.state = ?
                  {tier_clause}
                  AND (
                      m.expires_at IS NULL
                      OR julianday(m.expires_at) IS NULL
                      OR m.expires_at > ?
                  )
                ORDER BY distance ASC
                LIMIT ?
                """,
                parameters,
            ).fetchall()

        return [
            ScoredMemory(
                memory=self._memory_from_row(row),
                similarity=max(-1.0, min(1.0, 1.0 - float(row["distance"]))),
                usage=self._usage_from_row(row),
            )
            for row in rows
        ]

    def get_usage(self, memory_id: str) -> MemoryUsage | None:
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id must be a non-empty string")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    tier,
                    importance,
                    last_accessed_at,
                    access_count,
                    last_confirmed_at,
                    confirmation_count,
                    pinned
                FROM memory_usage
                WHERE memory_id = ?
                """,
                (memory_id,),
            ).fetchone()
        return None if row is None else self._usage_from_row(row)

    def record_access(
        self,
        memory_ids: Sequence[str],
        *,
        promote: bool = True,
    ) -> int:
        """Reinforce only memories that were actually returned to the caller."""
        unique_ids = tuple(dict.fromkeys(memory_ids))
        if not unique_ids:
            return 0
        if any(
            not isinstance(memory_id, str) or not memory_id.strip()
            for memory_id in unique_ids
        ):
            raise ValueError("memory_ids must contain non-empty strings")
        placeholders = ", ".join("?" for _ in unique_ids)
        # only update warm
        tier_update = (
            "CASE WHEN tier = 'WARM' THEN 'HOT' ELSE tier END" if promote else "tier"
        )
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE memory_usage
                SET last_accessed_at = CURRENT_TIMESTAMP,
                    access_count = access_count + 1,
                    tier = {tier_update}
                WHERE memory_id IN ({placeholders})
                """,
                unique_ids,
            )
        return cursor.rowcount

    def run_maintenance(
        self,
        *,
        warm_after_days: float = 30.0,
        cold_after_days: float = 180.0,
        max_hot_memories: int | None = None,
        target_hot_memories: int | None = None,
        reference_time: datetime | str | None = None,
        drop_cold_embeddings: bool = True,
        dry_run: bool = False,
    ) -> dict[str, int]:
        """Demote stale active memories without deleting text or provenance."""

        now_text = self._reference_timestamp(reference_time)
        now_value = datetime.fromisoformat(now_text)
        # exccedd this part,it can be warm
        warm_cutoff = (now_value - timedelta(days=float(warm_after_days))).isoformat()
        # exceed this part, it can be cold
        cold_cutoff = (now_value - timedelta(days=float(cold_after_days))).isoformat()

        # used to check if a memory is expired
        expired_predicate = """
            m.expires_at IS NOT NULL
            AND julianday(m.expires_at) IS NOT NULL
            AND m.expires_at <= ?
        """
        # used to check last time we use it .
        # access ? confrime ?
        activity = """
            CASE
                WHEN u.last_accessed_at IS NULL
                    THEN COALESCE(u.last_confirmed_at, m.created_at)
                WHEN u.last_confirmed_at IS NULL
                    THEN u.last_accessed_at
                WHEN julianday(u.last_accessed_at) >= julianday(u.last_confirmed_at)
                    THEN u.last_accessed_at
                ELSE u.last_confirmed_at
            END
        """
        # ->COLD
        cold_predicate = f"""
            u.tier != 'COLD'
            AND (
                ({expired_predicate})
                OR
                (
                    m.state = 'ACTIVE'
                    AND u.pinned = 0
                    AND julianday({activity}) <= julianday(?)
                )
                OR (
                    m.state = 'SUPERSEDED'
                    AND u.pinned = 0
                    AND julianday(COALESCE(m.superseded_at, {activity}))
                        <= julianday(?)
                )
            )
        """
        # ->WARM
        # Withn cold time but excced HOT time
        warm_predicate = f"""
            m.state = 'ACTIVE'
            AND u.tier = 'HOT'
            AND u.pinned = 0
            AND NOT ({expired_predicate})
            AND julianday({activity}) <= julianday(?)
            AND julianday({activity}) > julianday(?)
        """

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cold_count = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM memories AS m
                    JOIN memory_usage AS u ON u.memory_id = m.id
                    WHERE {cold_predicate}
                    """,
                    (now_text, cold_cutoff, cold_cutoff),
                ).fetchone()[0]
            )
            warm_count = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM memories AS m
                    JOIN memory_usage AS u ON u.memory_id = m.id
                    WHERE {warm_predicate}
                    """,
                    (now_text, warm_cutoff, cold_cutoff),
                ).fetchone()[0]
            )

            # lets update
            connection.execute(
                f"""
                UPDATE memory_usage
                SET tier = 'COLD'
                WHERE memory_id IN (
                    SELECT m.id
                    FROM memories AS m
                    JOIN memory_usage AS u ON u.memory_id = m.id
                    WHERE {cold_predicate}
                )
                """,
                (now_text, cold_cutoff, cold_cutoff),
            )
            connection.execute(
                f"""
                UPDATE memory_usage
                SET tier = 'WARM'
                WHERE memory_id IN (
                    SELECT m.id
                    FROM memories AS m
                    JOIN memory_usage AS u ON u.memory_id = m.id
                    WHERE {warm_predicate}
                )
                """,
                (now_text, warm_cutoff, cold_cutoff),
            )

            capacity_demoted = 0
            # Liimt Hot memories number
            if max_hot_memories is not None:
                target = (
                    max_hot_memories
                    if target_hot_memories is None
                    else target_hot_memories
                )
                hot_count = int(
                    connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM memories AS m
                        JOIN memory_usage AS u ON u.memory_id = m.id
                        WHERE m.state = 'ACTIVE' AND u.tier = 'HOT'
                        """
                    ).fetchone()[0]
                )
                if hot_count > max_hot_memories:
                    demotion_limit = hot_count - target
                    # ordered by importance, them access time, then activity time
                    rows = connection.execute(
                        f"""
                        SELECT m.id
                        FROM memories AS m
                        JOIN memory_usage AS u ON u.memory_id = m.id
                        WHERE m.state = 'ACTIVE'
                          AND u.tier = 'HOT'
                          AND u.pinned = 0
                        ORDER BY
                            u.importance ASC,
                            (u.access_count + u.confirmation_count) ASC,
                            julianday({activity}) ASC,
                            m.id ASC
                        LIMIT ?
                        """,
                        (demotion_limit,),
                    ).fetchall()
                    capacity_ids = tuple(row["id"] for row in rows)
                    # update these to WARM
                    if capacity_ids:
                        placeholders = ", ".join("?" for _ in capacity_ids)
                        connection.execute(
                            f"""
                            UPDATE memory_usage
                            SET tier = 'WARM'
                            WHERE memory_id IN ({placeholders})
                            """,
                            capacity_ids,
                        )
                        capacity_demoted = len(capacity_ids)

            # remove cold  guys
            embeddings_removed = 0
            if dry_run:
                connection.rollback()
            elif drop_cold_embeddings:
                cursor = connection.execute(
                    """
                    DELETE FROM memory_embeddings
                    WHERE memory_id IN (
                        SELECT m.id
                        FROM memories AS m
                        JOIN memory_usage AS u ON u.memory_id = m.id
                        WHERE u.tier = 'COLD'
                    )
                    """
                )
                embeddings_removed = cursor.rowcount

        return {
            "demoted_to_warm": warm_count + capacity_demoted,
            "demoted_to_cold": cold_count,
            "embeddings_removed": embeddings_removed,
        }

    # This is for migration of old datebases,which doesn't have embeddings
    def rebuild_embeddings(
        self,
        *,
        reference_time: datetime | str | None = None,
    ) -> int:
        """Recreate embeddings for searchable memories and retained history."""
        now_text = self._reference_timestamp(reference_time)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT m.id, m.content
                FROM memories AS m
                JOIN memory_usage AS u ON u.memory_id = m.id
                WHERE u.tier != 'COLD'
                  AND (
                      m.expires_at IS NULL
                      OR julianday(m.expires_at) IS NULL
                      OR m.expires_at > ?
                  )
                """,
                (now_text,),
            ).fetchall()
        embeddings = [
            (
                row["id"],
                self._create_document_embedding(row["content"]),
            )
            for row in rows
        ]

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rebuilt_count = 0
            for memory_id, embedding in embeddings:
                cursor = connection.execute(
                    """
                    INSERT INTO memory_embeddings (
                        memory_id,
                        dimensions,
                        embedding
                    )
                    SELECT ?, ?, ?
                    WHERE EXISTS (
                        SELECT 1
                        FROM memories AS m
                        JOIN memory_usage AS u ON u.memory_id = m.id
                        WHERE m.id = ?
                          AND u.tier != 'COLD'
                          AND (
                              m.expires_at IS NULL
                              OR julianday(m.expires_at) IS NULL
                              OR m.expires_at > ?
                          )
                    )
                    ON CONFLICT (memory_id) DO UPDATE SET
                        dimensions = excluded.dimensions,
                        embedding = excluded.embedding
                    """,
                    (
                        memory_id,
                        len(embedding),
                        sqlite_vec.serialize_float32(embedding),
                        memory_id,
                        now_text,
                    ),
                )
                rebuilt_count += cursor.rowcount

            connection.execute(
                """
                DELETE FROM memory_embeddings
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM memories AS m
                    JOIN memory_usage AS u ON u.memory_id = m.id
                    WHERE m.id = memory_embeddings.memory_id
                      AND u.tier != 'COLD'
                      AND (
                          m.expires_at IS NULL
                          OR julianday(m.expires_at) IS NULL
                          OR m.expires_at > ?
                      )
                )
                """,
                (now_text,),
            )

        return rebuilt_count

    def list_memories(self) -> list[MemoryRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    content,
                    created_at,
                    state,
                    superseded_by_id,
                    superseded_at,
                    expires_at
                FROM memories
                ORDER BY created_at DESC
                """
            ).fetchall()

        return [self._memory_from_row(row) for row in rows]
