#!/usr/bin/env python3
"""Initialize MemKernel's SQLite schema and rebuild all memory embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

from memkernel.database import initialize_database
from memkernel.embedding import OpenAIEmbeddingProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = PROJECT_ROOT / "memkernel.db"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create or migrate MemKernel's SQLite database and rebuild all "
            "stored memory embeddings."
        )
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help=f"SQLite database path (default: {DEFAULT_DATABASE})",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"embedding model (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    embedding_provider = OpenAIEmbeddingProvider(
        OpenAIEmbeddingProvider.get_client(),
        model=args.embedding_model,
    )
    memory_count, embedding_count = initialize_database(
        args.database,
        embedding_provider,
    )
    print(f"Initialized database: {args.database.expanduser().resolve()}")
    print(f"Memories: {memory_count}; embeddings rebuilt: {embedding_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
