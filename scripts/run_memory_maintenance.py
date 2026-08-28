"""Apply non-destructive HOT/WARM/COLD lifecycle maintenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memkernel.backend.sqlite_adapter import SQLiteBackend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Demote stale memories while retaining their text and provenance. "
            "COLD active memories leave semantic recall by default."
        )
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("memkernel.db"),
        help="SQLite database path (default: memkernel.db)",
    )
    parser.add_argument("--warm-after-days", type=float, default=30.0)
    parser.add_argument("--cold-after-days", type=float, default=180.0)
    parser.add_argument(
        "--max-hot-memories",
        type=int,
        help="trigger capacity demotion when HOT exceeds this count",
    )
    parser.add_argument(
        "--target-hot-memories",
        type=int,
        help="demote back to this HOT count (defaults to the maximum)",
    )
    parser.add_argument(
        "--reference-time",
        help="ISO-8601 clock override for reproducible research runs",
    )
    parser.add_argument(
        "--keep-cold-embeddings",
        action="store_true",
        help="retain vectors after demotion to COLD",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report planned demotions without changing the database",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    backend = SQLiteBackend(args.database)
    report = backend.run_maintenance(
        warm_after_days=args.warm_after_days,
        cold_after_days=args.cold_after_days,
        max_hot_memories=args.max_hot_memories,
        target_hot_memories=args.target_hot_memories,
        reference_time=args.reference_time,
        drop_cold_embeddings=not args.keep_cold_embeddings,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
