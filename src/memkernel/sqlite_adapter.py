import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from memkernel.backend import MemoryRecord
from memkernel.extractor import ExtractedResult


class SQLiteBackend:
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
