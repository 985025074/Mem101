#!/usr/bin/env python3
"""Small, dependency-free client for MemKernel's HTTP API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_URL = "http://127.0.0.1:8000"


class ClientError(RuntimeError):
    """An HTTP, connection, or response error from MemKernel."""


def json_object(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("metadata must be valid JSON") from error
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("metadata must be a JSON object")
    return payload


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def unit_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between zero and one")
    return parsed


class MemKernelClient:
    def __init__(self, base_url: str, timeout: float):
        normalized_url = base_url.strip().rstrip("/")
        if not normalized_url:
            raise ClientError("MemKernel URL must not be empty")
        self.base_url = normalized_url
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            detail = _format_error_body(body)
            raise ClientError(
                f"MemKernel returned HTTP {error.code}: {detail}"
            ) from error
        except URLError as error:
            raise ClientError(
                f"Could not reach MemKernel at {url}: {error.reason}"
            ) from error
        except TimeoutError as error:
            raise ClientError(
                f"MemKernel request timed out after {self.timeout:g}s"
            ) from error

        if not body:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError as error:
            raise ClientError("MemKernel returned a non-JSON response") from error


def _format_error_body(body: str) -> str:
    if not body:
        return "empty response"
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body.strip()
    if isinstance(payload, dict) and "detail" in payload:
        detail = payload["detail"]
        return (
            detail
            if isinstance(detail, str)
            else json.dumps(detail, ensure_ascii=False)
        )
    return json.dumps(payload, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Call a running MemKernel HTTP service.",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("MEMKERNEL_URL", DEFAULT_URL),
        help=f"service base URL (default: MEMKERNEL_URL or {DEFAULT_URL})",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=10.0,
        help="request timeout in seconds (default: 10)",
    )
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")

    commands = parser.add_subparsers(dest="command", required=True)

    recall = commands.add_parser("recall", help="search relevant memories")
    recall.add_argument("query", help="semantic search query")
    recall.add_argument("--current-top-k", type=int, default=5)
    recall.add_argument("--history-top-k", type=int, default=0)
    recall.add_argument("--threshold", type=float, default=0.5)

    remember = commands.add_parser("remember", help="extract and persist memories")
    remember.add_argument("content", help="source content to remember")
    remember.add_argument(
        "--source-type",
        choices=("message", "tool", "document"),
        default="message",
    )
    remember.add_argument(
        "--role",
        choices=("user", "assistant", "system", "tool", "none"),
        help=(
            "source role (default: user for messages, tool for tools, "
            "none for documents)"
        ),
    )
    remember.add_argument("--observed-at", help="ISO-8601 source timestamp")
    remember.add_argument("--metadata", type=json_object)
    remember.add_argument(
        "--tier",
        choices=("HOT", "WARM", "COLD"),
        help="initial memory tier (default: HOT)",
    )
    remember.add_argument(
        "--importance",
        type=unit_float,
        help="memory importance from 0 to 1 (default: 0.5)",
    )
    remember.add_argument(
        "--expires-at",
        help="ISO-8601 timestamp after which the memory expires",
    )
    remember.add_argument(
        "--pinned",
        action="store_true",
        help="protect the memory from ordinary age and capacity demotion",
    )

    history = commands.add_parser(
        "history",
        help="show a memory's supersession chain",
    )
    history.add_argument("memory_id")

    sources = commands.add_parser("sources", help="show a memory's source evidence")
    sources.add_argument("memory_id")

    commands.add_parser("health", help="check that the service is reachable")
    return parser


def execute(client: MemKernelClient, args: argparse.Namespace) -> Any:
    if args.command == "recall":
        return client.request(
            "POST",
            "/v1/recall",
            {
                "query": args.query,
                "current_top_k": args.current_top_k,
                "history_top_k": args.history_top_k,
                "threshold": args.threshold,
            },
        )

    if args.command == "remember":
        default_role = {
            "message": "user",
            "tool": "tool",
            "document": None,
        }[args.source_type]
        role = default_role if args.role is None else args.role
        payload: dict[str, Any] = {
            "content": args.content,
            "source_type": args.source_type,
            "role": None if role == "none" else role,
            "metadata": args.metadata or {},
        }
        if args.observed_at is not None:
            payload["observed_at"] = args.observed_at
        if args.tier is not None:
            payload["tier"] = args.tier
        if args.importance is not None:
            payload["importance"] = args.importance
        if args.expires_at is not None:
            payload["expires_at"] = args.expires_at
        if args.pinned:
            payload["pinned"] = True
        return client.request("POST", "/v1/memories", payload)

    if args.command == "history":
        memory_id = quote(args.memory_id, safe="")
        return client.request("GET", f"/v1/memories/{memory_id}/history")

    if args.command == "sources":
        memory_id = quote(args.memory_id, safe="")
        return client.request("GET", f"/v1/memories/{memory_id}/sources")

    if args.command == "health":
        return client.request("GET", "/")

    raise ClientError(f"unsupported command: {args.command}")


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = execute(MemKernelClient(args.url, args.timeout), args)
    except ClientError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=None if args.compact else 2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
