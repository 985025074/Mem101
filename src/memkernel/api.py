from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from memkernel.backend.backend import (
    MemoryAction,
    MemoryDecision,
    MemoryRecord,
    MemoryState,
)
from memkernel.backend.backend_v2 import BackendV2
from memkernel.extractor.extractor_v2 import ExtractionValidationError, LLMExtractorV2
from memkernel.kernel import MemKernel, PostMemory
from memkernel.provenance import (
    MemorySourceRecord,
    SourceLinkType,
    SourceRole,
    SourceType,
)
from memkernel.retriver import RetrievalResult


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

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        content = value.strip()
        if not content:
            raise ValueError("content must be a non-empty string")
        return content


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


def create_app(
    *,
    kernel: MemKernel | None = None,
) -> FastAPI:
    application = FastAPI(title="MemKernel")
    if kernel is not None:
        application.state.kernel = kernel

    @application.get("/")
    def home() -> dict[str, str]:
        return {"message": "Hello, MemKernel"}

    @application.post("/v1/recall", response_model=RecallResponse)
    def recall(
        request: RecallRequest,
        memory_kernel: Annotated[MemKernel, Depends(_get_kernel)],
    ) -> RecallResponse:
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
        try:
            decisions = memory_kernel.remember(
                PostMemory(
                    date=request.observed_at,
                    content=request.content,
                    source_type=request.source_type,
                    role=request.role,
                    metadata=request.metadata,
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
        sources = memory_kernel.get_sources(memory_id)
        if sources is None:
            raise HTTPException(status_code=404, detail="Memory was not found")
        return MemorySourcesResponse(
            sources=[_source_response(source) for source in sources]
        )

    return application


kernel = MemKernel(
    extractor=LLMExtractorV2(),
    memory_backend=BackendV2(),
)

app = create_app(kernel=kernel)


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
