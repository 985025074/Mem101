import json
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, cast

import sqlite_vec

from memkernel.backend.backend import (
    MemoryDecision,
    MemoryRecord,
    MemoryState,
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
                    FOREIGN KEY (superseded_by_id) REFERENCES memories(id)
                        ON DELETE SET NULL
                )
                """
            )
            # add state and  others
            self._migrate_memory_state(connection)
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
    def _migrate_memory_state(connection: sqlite3.Connection) -> None:
        """Add lifecycle columns to databases created before memory states."""
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
        )

    @staticmethod
    def _validate_content(content: str) -> str:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        return content.strip()

    @staticmethod
    def _insert_memory_row(
        connection: sqlite3.Connection,
        *,
        memory_id: str,
        content: str,
        embedding: list[float] | None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO memories (id, content)
            VALUES (?, ?)
            """,
            (memory_id, content),
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

        embedding = [
            float(value) for value in self.embedding_provider.embed_query(content)
        ]

        return embedding

    def insert(self, content: str) -> str:
        content = self._validate_content(content)
        memory_id = str(uuid.uuid4())
        embedding = None
        if self.embedding_provider is not None:
            embedding = self._create_document_embedding(content)

        with self._connect() as connection:
            self._insert_memory_row(
                connection,
                memory_id=memory_id,
                content=content,
                embedding=embedding,
            )

        return memory_id

    def supersede(self, memory_id: str, content: str) -> str:
        """Atomically replace an active memory while preserving its history."""
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id must be a non-empty string")
        content = self._validate_content(content)

        new_memory_id = str(uuid.uuid4())
        embedding = None
        if self.embedding_provider is not None:
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

        return new_memory_id

    def apply_decisions(
        self,
        source_event: SourceEvent,
        changes: Sequence[tuple[MemoryDecision, str]],
    ) -> list[MemoryDecision]:
        """Atomically persist one source event and all memory decisions it caused."""
        if not changes:
            return []

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
                if self.embedding_provider is not None:
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
                    link_type = "DERIVED"
                else:
                    raise ValueError(f"Unsupported memory action: {decision.action}")

                # add memory to event link
                connection.execute(
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
                completed.append(completed_decision)

        return completed

    def remember(
        self,
        extracted: ExtractedResult,
        source_event: SourceEvent | None = None,
    ) -> str:
        if source_event is not None:
            raise TypeError(
                "SQLiteBackend cannot reconcile sourced extracted results directly"
            )
        return self.insert(extracted.content)

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
                    superseded_at
                FROM memories
                WHERE id = ?
                """,
                (memory_id,),
            ).fetchone()

        if row is None:
            return None

        return self._memory_from_row(row)

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
                    superseded_at
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

    def search_current(self, content: str, top_k: int = 5) -> list[ScoredMemory]:
        return self._search_by_state(content, top_k=top_k, state="ACTIVE")

    def search_history(self, content: str, top_k: int = 5) -> list[ScoredMemory]:
        return self._search_by_state(content, top_k=top_k, state="SUPERSEDED")

    def _search_by_state(
        self,
        content: str,
        *,
        top_k: int,
        state: MemoryState,
    ) -> list[ScoredMemory]:
        content = self._validate_content(content)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        query_embedding = self._create_query_embedding(content)
        serialized_query = sqlite_vec.serialize_float32(query_embedding)

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    m.id,
                    m.content,
                    m.created_at,
                    m.state,
                    m.superseded_by_id,
                    m.superseded_at,
                    vec_distance_cosine(e.embedding, ?) AS distance
                FROM memory_embeddings AS e
                JOIN memories AS m ON m.id = e.memory_id
                WHERE e.dimensions = ? AND m.state = ?
                ORDER BY distance ASC
                LIMIT ?
                """,
                (serialized_query, len(query_embedding), state, top_k),
            ).fetchall()

        return [
            ScoredMemory(
                memory=self._memory_from_row(row),
                similarity=max(-1.0, min(1.0, 1.0 - float(row["distance"]))),
            )
            for row in rows
        ]

    # This is for migration of old datebases,which doesn't have embeddings
    def rebuild_embeddings(self) -> int:
        """Recreate embeddings for every memory using the configured provider."""
        embeddings = [
            (memory.id, self._create_document_embedding(memory.content))
            for memory in self.list_memories()
        ]

        with self._connect() as connection:
            connection.execute("DELETE FROM memory_embeddings")
            connection.executemany(
                """
                INSERT INTO memory_embeddings (memory_id, dimensions, embedding)
                VALUES (?, ?, ?)
                """,
                [
                    (
                        memory_id,
                        len(embedding),
                        sqlite_vec.serialize_float32(embedding),
                    )
                    for memory_id, embedding in embeddings
                ],
            )

        return len(embeddings)

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
                    superseded_at
                FROM memories
                ORDER BY created_at DESC
                """
            ).fetchall()

        return [self._memory_from_row(row) for row in rows]
