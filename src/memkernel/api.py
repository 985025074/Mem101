from __future__ import annotations

import html
import json
import logging
from datetime import datetime
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from memkernel.backend.backend import (
    MemoryAction,
    MemoryDecision,
    MemoryRecord,
    MemoryState,
    MemoryTier,
)
from memkernel.backend.backend_v2 import BackendV2
from memkernel.embedding import OpenAIEmbeddingProvider
from memkernel.extractor.extractor_v2 import ExtractionValidationError, LLMExtractorV2
from memkernel.kernel import MemKernel, PostMemory
from memkernel.provenance import (
    MemorySourceRecord,
    SourceLinkType,
    SourceRole,
    SourceType,
)
from memkernel.retriver import RetrievalResult


logger = logging.getLogger(__name__)


class RecallRequest(BaseModel):
    query: str
    current_top_k: int = Field(default=5, ge=1, le=100, strict=True)
    history_top_k: int = Field(default=0, ge=0, le=100, strict=True)
    # similarity that determines how close the memory we want
    threshold: float = Field(default=0.5, ge=0, le=1.0, strict=True)

    # used to validate query
    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("query must be a non-empty string")
        return query


class RememberRequest(BaseModel):
    content: str
    source_type: SourceType = "message"
    role: SourceRole | None = "user"
    observed_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tier: MemoryTier = "HOT"
    importance: float = Field(default=0.5, ge=0.0, le=1.0, strict=True)
    expires_at: str | None = None
    pinned: bool = Field(default=False, strict=True)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        content = value.strip()
        if not content:
            raise ValueError("content must be a non-empty string")
        return content

    @field_validator("expires_at")
    @classmethod
    def validate_expires_at(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(
                "expires_at must be a valid ISO-8601 timestamp"
            ) from error
        return value


class MemoryDecisionResponse(BaseModel):
    action: MemoryAction
    fact: str
    memory_id: str | None
    matched_memory_id: str | None


# response of Remember endpoint
class RememberResponse(BaseModel):
    decisions: list[MemoryDecisionResponse]


# One single memory, corresponding to sqlite's row
# used for memory query
class MemoryResponse(BaseModel):
    id: str
    content: str
    created_at: str
    state: MemoryState
    superseded_by_id: str | None
    superseded_at: str | None


# wrapper with matching score
class RetrievedMemoryResponse(MemoryResponse):
    score: float


# Recall result
class RecallResponse(BaseModel):
    current: list[RetrievedMemoryResponse]
    history: list[RetrievedMemoryResponse]


# history of a memory
class MemoryHistoryResponse(BaseModel):
    memories: list[MemoryResponse]


# Event
class MemorySourceResponse(BaseModel):
    id: str
    content: str
    source_type: SourceType
    role: SourceRole | None
    observed_at: str
    created_at: str
    metadata: dict[str, Any]
    evidence_quote: str
    link_type: SourceLinkType
    linked_at: str


class MemorySourcesResponse(BaseModel):
    sources: list[MemorySourceResponse]


def _get_kernel(request: Request) -> MemKernel:
    kernel = getattr(request.app.state, "kernel", None)
    if kernel is None:
        raise HTTPException(status_code=503, detail="MemKernel is not configured")
    return kernel


def _memory_response(memory: MemoryRecord) -> MemoryResponse:
    return MemoryResponse(
        id=memory.id,
        content=memory.content,
        created_at=memory.created_at,
        state=memory.state,
        superseded_by_id=memory.superseded_by_id,
        superseded_at=memory.superseded_at,
    )


def _retrieval_response(result: RetrievalResult) -> RetrievedMemoryResponse:
    memory = result.memory
    return RetrievedMemoryResponse(
        id=memory.id,
        content=memory.content,
        created_at=memory.created_at,
        state=memory.state,
        superseded_by_id=memory.superseded_by_id,
        superseded_at=memory.superseded_at,
        score=result.score,
    )


def _decision_response(decision: MemoryDecision) -> MemoryDecisionResponse:
    return MemoryDecisionResponse(
        action=decision.action,
        fact=decision.fact,
        memory_id=decision.memory_id,
        matched_memory_id=decision.matched_memory_id,
    )


def _source_response(source_record: MemorySourceRecord) -> MemorySourceResponse:
    source = source_record.source
    return MemorySourceResponse(
        id=source.id,
        content=source.content,
        source_type=source.source_type,
        role=source.role,
        observed_at=source.observed_at,
        created_at=source.created_at,
        metadata=source.metadata,
        evidence_quote=source_record.evidence_quote,
        link_type=source_record.link_type,
        linked_at=source_record.linked_at,
    )


def _html_cell(value: object | None) -> str:
    if value is None or value == "":
        return '<span class="empty">—</span>'
    return html.escape(str(value), quote=True)


def _debug_table(memory_kernel: MemKernel) -> str:
    memories = memory_kernel.list_memories()
    rows: list[str] = []
    source_link_count = 0

    for memory in memories:
        linked_sources = memory_kernel.get_sources(memory.id) or []
        usage = memory_kernel.get_usage(memory.id)
        source_rows: list[MemorySourceRecord | None] = (
            list(linked_sources) if linked_sources else [None]
        )

        for source_record in source_rows:
            source_link_count += source_record is not None
            if source_record is None:
                source_type = role = observed_at = link_type = None
                evidence_quote = source_content = metadata = None
            else:
                source = source_record.source
                source_type = source.source_type
                role = source.role
                observed_at = source.observed_at
                link_type = source_record.link_type
                evidence_quote = source_record.evidence_quote
                source_content = source.content
                metadata = json.dumps(
                    source.metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                )

            rows.append(
                f"""
                <tr>
                  <td><code>{_html_cell(memory.id)}</code></td>
                  <td><span class="state {memory.state.lower()}">{_html_cell(memory.state)}</span></td>
                  <td>{_html_cell(usage.tier if usage else None)}</td>
                  <td>{_html_cell(usage.importance if usage else None)}</td>
                  <td>{_html_cell(usage.access_count if usage else None)}</td>
                  <td>{_html_cell(usage.confirmation_count if usage else None)}</td>
                  <td>{_html_cell(memory.expires_at)}</td>
                  <td class="text">{_html_cell(memory.content)}</td>
                  <td>{_html_cell(memory.created_at)}</td>
                  <td><code>{_html_cell(memory.superseded_by_id)}</code></td>
                  <td>{_html_cell(memory.superseded_at)}</td>
                  <td>{_html_cell(source_type)}</td>
                  <td>{_html_cell(role)}</td>
                  <td>{_html_cell(observed_at)}</td>
                  <td>{_html_cell(link_type)}</td>
                  <td class="text">{_html_cell(evidence_quote)}</td>
                  <td class="text">{_html_cell(source_content)}</td>
                  <td class="text"><code>{_html_cell(metadata)}</code></td>
                </tr>
                """
            )

    if not rows:
        rows.append(
            '<tr><td colspan="18" class="no-data">No memories stored.</td></tr>'
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="robots" content="noindex, nofollow">
  <title>MemKernel debug view</title>
  <style>
    :root {{ color-scheme: light dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
    body {{ margin: 0; padding: 24px; background: #f6f7f9; color: #17202a; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    p {{ margin: 0 0 18px; color: #59636e; }}
    .table-wrap {{ overflow: auto; border: 1px solid #d8dee4; border-radius: 8px; background: white; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e8ebee; text-align: left; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #eef1f4; white-space: nowrap; }}
    tr:last-child td {{ border-bottom: 0; }}
    tr:hover td {{ background: #f8fafb; }}
    code {{ font-size: 12px; }}
    .text {{ min-width: 240px; max-width: 440px; white-space: pre-wrap; overflow-wrap: anywhere; }}
    .state {{ display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 11px; font-weight: 700; }}
    .active {{ color: #176b3a; background: #dafbe1; }}
    .superseded {{ color: #7d4e00; background: #fff1c2; }}
    .empty, .no-data {{ color: #8c959f; }}
    .no-data {{ padding: 28px; text-align: center; }}
    @media (prefers-color-scheme: dark) {{
      body {{ background: #0d1117; color: #e6edf3; }}
      p, .empty, .no-data {{ color: #8b949e; }}
      .table-wrap {{ border-color: #30363d; background: #161b22; }}
      th {{ background: #21262d; }}
      th, td {{ border-color: #30363d; }}
      tr:hover td {{ background: #1c2128; }}
    }}
  </style>
</head>
<body>
  <h1>MemKernel memories</h1>
  <p>{len(memories)} memories · {source_link_count} source links · newest first</p>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Memory ID</th><th>State</th><th>Tier</th><th>Importance</th>
          <th>Accesses</th><th>Confirmations</th><th>Expires</th>
          <th>Memory</th><th>Created</th>
          <th>Superseded by</th><th>Superseded at</th><th>Source type</th>
          <th>Role</th><th>Observed</th><th>Link</th><th>Evidence</th>
          <th>Source content</th><th>Metadata</th>
        </tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </div>
</body>
</html>"""


def create_app(
    *,
    kernel: MemKernel | None = None,
) -> FastAPI:
    application = FastAPI(title="MemKernel")
    if kernel is not None:
        application.state.kernel = kernel

    @application.get("/")
    def home() -> dict[str, str]:
        logger.debug("GET /")
        return {"message": "Hello, MemKernel"}

    @application.get(
        "/debug/memories",
        response_class=HTMLResponse,
        tags=["debug"],
    )
    def debug_memories(
        memory_kernel: Annotated[MemKernel, Depends(_get_kernel)],
    ) -> HTMLResponse:
        logger.debug("GET /debug/memories")
        return HTMLResponse(
            _debug_table(memory_kernel),
            headers={"Cache-Control": "no-store"},
        )

    @application.post("/v1/recall", response_model=RecallResponse)
    def recall(
        request: RecallRequest,
        memory_kernel: Annotated[MemKernel, Depends(_get_kernel)],
    ) -> RecallResponse:
        logger.debug("POST /v1/recall request=%s", request)
        try:
            results = memory_kernel.recall(
                request.query,
                current_top_k=request.current_top_k,
                history_top_k=request.history_top_k,
                threshold=request.threshold,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        return RecallResponse(
            current=[_retrieval_response(item) for item in results.current],
            history=[_retrieval_response(item) for item in results.history],
        )

    @application.post("/v1/memories", response_model=RememberResponse)
    def remember(
        request: RememberRequest,
        memory_kernel: Annotated[MemKernel, Depends(_get_kernel)],
    ) -> RememberResponse:
        logger.debug(
            "POST /v1/memories source_type=%s role=%s observed_at=%s "
            "tier=%s importance=%s expires_at=%s pinned=%s "
            "content_length=%d metadata_keys=%s",
            request.source_type,
            request.role,
            request.observed_at,
            request.tier,
            request.importance,
            request.expires_at,
            request.pinned,
            len(request.content),
            sorted(request.metadata),
        )
        try:
            decisions = memory_kernel.remember(
                PostMemory(
                    date=request.observed_at,
                    content=request.content,
                    source_type=request.source_type,
                    role=request.role,
                    metadata=request.metadata,
                    tier=request.tier,
                    importance=request.importance,
                    expires_at=request.expires_at,
                    pinned=request.pinned,
                )
            )
        except ExtractionValidationError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return RememberResponse(
            decisions=[_decision_response(decision) for decision in decisions]
        )

    @application.get(
        "/v1/memories/{memory_id}/history",
        response_model=MemoryHistoryResponse,
    )
    def memory_history(
        memory_id: str,
        memory_kernel: Annotated[MemKernel, Depends(_get_kernel)],
    ) -> MemoryHistoryResponse:
        logger.debug(
            "GET /v1/memories/{memory_id}/history memory_id=%s",
            memory_id,
        )
        history = memory_kernel.get_history(memory_id)
        if history is None:
            raise HTTPException(status_code=404, detail="Memory was not found")
        return MemoryHistoryResponse(
            memories=[_memory_response(memory) for memory in history]
        )

    @application.get(
        "/v1/memories/{memory_id}/sources",
        response_model=MemorySourcesResponse,
    )
    def memory_sources(
        memory_id: str,
        memory_kernel: Annotated[MemKernel, Depends(_get_kernel)],
    ) -> MemorySourcesResponse:
        logger.debug(
            "GET /v1/memories/{memory_id}/sources memory_id=%s",
            memory_id,
        )
        sources = memory_kernel.get_sources(memory_id)
        if sources is None:
            raise HTTPException(status_code=404, detail="Memory was not found")
        return MemorySourcesResponse(
            sources=[_source_response(source) for source in sources]
        )

    return application


app = create_app(
    kernel=MemKernel(
        extractor=LLMExtractorV2(),
        memory_backend=BackendV2(
            embedding_provider=OpenAIEmbeddingProvider(
                OpenAIEmbeddingProvider.get_client()
            )
        ),
    ),
)


__all__ = [
    "MemoryHistoryResponse",
    "MemorySourceResponse",
    "MemorySourcesResponse",
    "MemoryDecisionResponse",
    "MemoryResponse",
    "PostMemory",
    "RecallRequest",
    "RecallResponse",
    "RememberRequest",
    "RememberResponse",
    "RetrievedMemoryResponse",
    "app",
    "create_app",
]
