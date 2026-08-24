import sqlite3
from typing import Any, Dict
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from memkernel.backend import MemoryRecord
from memkernel.extractor import ExtractedResult


class SQLiteBackend:
    _QUERYABLE_COLUMNS = frozenset({"id", "content", "created_at"})

    def __init__(
        self,
        database_path: str | Path = "memkernel.db",
    ):
        self.database_path = Path(database_path)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")

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
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def remember(self, extracted: ExtractedResult) -> str:
        memory_id = str(uuid.uuid4())

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memories (id, content)
                VALUES (?, ?)
                """,
                (memory_id, extracted.content),
            )

        return memory_id

    def get(self, memory_id: str) -> MemoryRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, content, created_at
                FROM memories
                WHERE id = ?
                """,
                (memory_id,),
            ).fetchone()

        if row is None:
            return None

        return MemoryRecord(
            id=row["id"],
            content=row["content"],
            created_at=row["created_at"],
        )

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
                SELECT id, content, created_at
                FROM memories
                WHERE {where_clause}
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                values,
            ).fetchone()

        if row is None:
            return None

        return MemoryRecord(
            id=row["id"],
            content=row["content"],
            created_at=row["created_at"],
        )

    def list_memories(self) -> list[MemoryRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, content, created_at
                FROM memories
                ORDER BY created_at DESC
                """
            ).fetchall()

        return [
            MemoryRecord(
                id=row["id"],
                content=row["content"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
