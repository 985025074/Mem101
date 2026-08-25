#!/usr/bin/env python3
"""Install the project skill into the shared global Agent Skills directory."""

from __future__ import annotations

import argparse
from pathlib import Path


# Keep the canonical source in this project and expose it globally by symlink.
SKILL_NAME = "memkernel-memory"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_SOURCE = PROJECT_ROOT / "skills" / SKILL_NAME
GLOBAL_SKILLS_ROOT = Path.home() / ".agents" / "skills"


def install(target: Path, *, dry_run: bool) -> None:
    source = SKILL_SOURCE.resolve()
    if not (source / "SKILL.md").is_file():
        raise SystemExit(f"error: skill source is missing: {source}")

    if target.is_symlink():
        if target.resolve(strict=False) == source:
            print(f"Skill is already installed: {target} -> {source}")
            return
        raise SystemExit(f"error: a different symlink already exists at {target}")
    if target.exists():
        raise SystemExit(f"error: refusing to overwrite existing path: {target}")

    if dry_run:
        print(f"Would install: {target} -> {source}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=True)
    print(f"Installed: {target} -> {source}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install the MemKernel skill globally as a symlink.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the destination without changing anything",
    )
    args = parser.parse_args()

    target = GLOBAL_SKILLS_ROOT / SKILL_NAME
    install(target, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"Restart or reload the agent to discover {SKILL_NAME}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
