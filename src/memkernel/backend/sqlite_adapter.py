import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, cast

import sqlite_vec

from memkernel.backend.backend import MemoryRecord, MemoryState, ScoredMemory
from memkernel.embedding import EmbeddingProvider
from memkernel.extractor import ExtractedResult


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

    def _create_embedding(self, content: str) -> list[float]:
        """Get embedding of a string"""
        if self.embedding_provider is None:
            raise RuntimeError("Semantic search requires an embedding_provider")

        embedding = [float(value) for value in self.embedding_provider.embed(content)]

        return embedding

    def insert(self, content: str) -> str:
        content = self._validate_content(content)
        memory_id = str(uuid.uuid4())
        embedding = None
        if self.embedding_provider is not None:
            embedding = self._create_embedding(content)

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memories (id, content)
                VALUES (?, ?)
                """,
                (memory_id, content),
            )
            # insert embedding of the memory
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

        return memory_id

    def supersede(self, memory_id: str, content: str) -> str:
        """Atomically replace an active memory while preserving its history."""
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id must be a non-empty string")
        content = self._validate_content(content)

        new_memory_id = str(uuid.uuid4())
        embedding = None
        if self.embedding_provider is not None:
            embedding = self._create_embedding(content)

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

            connection.execute(
                """
                INSERT INTO memories (id, content)
                VALUES (?, ?)
                """,
                (new_memory_id, content),
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
                        new_memory_id,
                        len(embedding),
                        sqlite_vec.serialize_float32(embedding),
                    ),
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

    def remember(self, extracted: ExtractedResult) -> str:
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

    def remove(self, memory_id: str) -> bool:
        """Remove a memory and its embedding, returning whether it existed."""
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id must be a non-empty string")

        # embedding uses foreign key,We don't need to delete again
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM memories WHERE id = ?",
                (memory_id,),
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

        query_embedding = self._create_embedding(content)
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

    def rebuild_embeddings(self) -> int:
        """Recreate embeddings for every memory using the configured provider."""
        embeddings = [
            (memory.id, self._create_embedding(memory.content))
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
