import logging
import math
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Protocol

from memkernel.backend.backend import MemoryTier, ScoredMemory
from memkernel.retriver import RecallResults, RetrievalResult


logger = logging.getLogger(__name__)


class SearchBackend(Protocol):
    def search_current(self, content: str, top_k: int = 5) -> list[ScoredMemory]: ...

    def search_current_by_tier(
        self,
        content: str,
        *,
        top_k: int = 5,
        tiers: Sequence[MemoryTier],
        reference_time: datetime | str | None = None,
    ) -> list[ScoredMemory]: ...

    def search_history(self, content: str, top_k: int = 5) -> list[ScoredMemory]: ...


class SemanticRetriever:
    """Tier-aware semantic retrieval with bounded access/recency reranking."""

    def __init__(
        self,
        memory_backend: SearchBackend,
        *,
        candidate_multiplier: int = 5,
        decay_half_life_days: float = 30.0,
        decay_enabled: bool = True,
        record_access: bool = True,
        clock: Callable[[], datetime] | None = None,
    ):
        self.memory_backend = memory_backend
        # How much we select from
        self.candidate_multiplier = candidate_multiplier
        # How old the memory we want if the staleness is big, it will lower the weights of old memory
        self.decay_half_life_days = float(decay_half_life_days)
        self.decay_enabled = bool(decay_enabled)
        # Update memory activity  time  we used
        self.record_access_enabled = bool(record_access)
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> list[RetrievalResult]:
        """Retrieve current memories using the legacy convenience API."""
        return self.retrieve_current(query, top_k=top_k, threshold=threshold)

    def recall(
        self,
        query: str,
        *,
        current_top_k: int = 5,
        history_top_k: int = 0,
        threshold: float = 0.5,
    ) -> RecallResults:
        """Retrieve current and historical memories with independent limits."""
        if (
            isinstance(history_top_k, bool)
            or not isinstance(history_top_k, int)
            or history_top_k < 0
        ):
            raise ValueError("history_top_k must be a non-negative integer")

        current = self.retrieve_current(
            query,
            top_k=current_top_k,
            threshold=threshold,
        )
        history = (
            self.retrieve_history(
                query,
                top_k=history_top_k,
                threshold=threshold,
            )
            if history_top_k > 0
            else []
        )
        return RecallResults(current=current, history=history)

    def retrieve_current(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> list[RetrievalResult]:
        self._validate_request(query, top_k=top_k, threshold=threshold)
        normalized_query = query.strip()
        return self._retrieve_current_by_tier(
            normalized_query,
            top_k=top_k,
            threshold=threshold,
        )

    def retrieve_history(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> list[RetrievalResult]:
        return self._retrieve(
            self.memory_backend.search_history,
            query,
            top_k=top_k,
            threshold=threshold,
            promote=False,
        )

    def _retrieve_current_by_tier(
        self,
        query: str,
        *,
        top_k: int,
        threshold: float,
    ) -> list[RetrievalResult]:
        now = self._now()
        candidate_limit = max(top_k, top_k * self.candidate_multiplier)
        # Got hot ones
        matches = self.memory_backend.search_current_by_tier(
            query,
            top_k=candidate_limit,
            tiers=("HOT",),
            reference_time=now,
        )
        eligible_hot = self._eligible(matches, threshold=threshold, now=now)
        # If not eligible then lets go to warm place
        if len(eligible_hot) < top_k:
            warm = self.memory_backend.search_current_by_tier(
                query,
                top_k=candidate_limit,
                tiers=("WARM",),
                reference_time=now,
            )
            matches = self._deduplicate([*matches, *warm])

        return self._finalize(
            matches,
            top_k=top_k,
            threshold=threshold,
            now=now,
            promote=True,
        )

    def _retrieve(
        self,
        search: Callable[..., list[ScoredMemory]],
        query: str,
        *,
        top_k: int,
        threshold: float,
        promote: bool,
    ) -> list[RetrievalResult]:
        self._validate_request(query, top_k=top_k, threshold=threshold)

        matches = search(
            query.strip(),
            top_k=top_k,
        )
        return self._finalize(
            matches,
            top_k=top_k,
            threshold=threshold,
            now=self._now(),
            promote=promote,
        )

    @staticmethod
    def _validate_request(query: str, *, top_k: int, threshold: float) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be a number")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

    def _finalize(
        self,
        matches: Sequence[ScoredMemory],
        *,
        top_k: int,
        threshold: float,
        now: datetime,
        promote: bool,
    ) -> list[RetrievalResult]:
        eligible = self._eligible(matches, threshold=threshold, now=now)
        ranked = sorted(
            eligible,
            key=lambda item: (
                item.rank_score if item.rank_score is not None else item.similarity
            ),
            reverse=True,
        )[:top_k]

        # renew these memories
        self._record_access(
            [match.memory.id for match in ranked],
            promote=promote,
        )

        return [
            RetrievalResult(
                memory=match.memory,
                score=match.similarity,
                rank_score=match.rank_score,
            )
            for match in ranked
        ]

    def _eligible(
        self,
        matches: Sequence[ScoredMemory],
        *,
        threshold: float,
        now: datetime,
    ) -> list[ScoredMemory]:
        """remove expired ones, and"""
        eligible: list[ScoredMemory] = []
        for match in matches:
            if match.similarity < float(threshold):
                continue
            if self._is_expired(match, now):
                continue
            rank_score = self._rank_score(match, now)
            eligible.append(
                ScoredMemory(
                    memory=match.memory,
                    similarity=match.similarity,
                    usage=match.usage,
                    rank_score=rank_score,
                )
            )
        return eligible

    def _rank_score(self, match: ScoredMemory, now: datetime) -> float | None:
        usage = match.usage
        if not self.decay_enabled or usage is None:
            return match.rank_score

        # parse time
        anchors = [
            parsed
            for parsed in (
                self._parse_timestamp(usage.last_accessed_at),
                self._parse_timestamp(usage.last_confirmed_at),
            )
            if parsed is not None
        ]
        created_at = self._parse_timestamp(match.memory.created_at)
        if not anchors and created_at is not None:
            anchors.append(created_at)
        # get latest active  times
        anchor = max(anchors) if anchors else None
        is_neutral = (
            anchor is None
            and usage.access_count == 0
            and usage.confirmation_count == 0
            and usage.importance == 0.5
            and not usage.pinned
        )
        if is_neutral:
            return match.rank_score

        age_days = 0.0
        # TODO: These  paramters need to be confirmed or experimend in the future.
        if anchor is not None:
            age_days = max(0.0, (now - anchor).total_seconds() / 86_400.0)
        staleness = 1.0 - math.pow(
            0.5,
            age_days / self.decay_half_life_days,
        )
        access_strength = min(
            1.0,
            math.log1p(max(usage.access_count, 0)) / math.log1p(20),
        )
        confirmation_strength = min(
            1.0,
            math.log1p(max(usage.confirmation_count, 0)) / math.log1p(10),
        )
        # According to above parameters do weighted calculation
        factor = (
            1.0
            + 0.4 * (usage.importance - 0.5)
            + 0.2 * access_strength
            + 0.1 * confirmation_strength
            + (0.15 if usage.pinned else 0.0)
            - 0.35 * staleness
        )
        factor = max(0.3, min(1.5, factor))
        return match.similarity * factor

    def _record_access(self, memory_ids: Sequence[str], *, promote: bool) -> None:
        if not self.record_access_enabled or not memory_ids:
            return
        callback = getattr(self.memory_backend, "record_access", None)
        if not callable(callback):
            return
        try:
            callback(memory_ids, promote=promote)
        except Exception:
            logger.warning("Failed to record memory access", exc_info=True)

    @classmethod
    def _is_expired(cls, match: ScoredMemory, now: datetime) -> bool:
        expires_at = cls._parse_timestamp(match.memory.expires_at)
        return expires_at is not None and expires_at <= now

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _deduplicate(matches: Sequence[ScoredMemory]) -> list[ScoredMemory]:
        by_id: dict[str, ScoredMemory] = {}
        for match in matches:
            existing = by_id.get(match.memory.id)
            if existing is None or match.similarity > existing.similarity:
                by_id[match.memory.id] = match
        return list(by_id.values())
