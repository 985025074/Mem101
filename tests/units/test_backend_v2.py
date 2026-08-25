import json
import sqlite3
import uuid

import pytest

from memkernel.backend.backend import MemoryDecision, ScoredMemory
from memkernel.backend.backend_v2 import (
    BATCH_RECONCILE_PROMPT,
    BackendV2,
    RECONCILE_PROMPT,
)
from memkernel.extractor.extractor import SimpleExtractedResult
from memkernel.extractor.extractor_v2 import (
    ExtractedFact,
    JsonExtractedResult,
)
from memkernel.provenance import SourceEvent


class RecordingAI:
    def __init__(self, response: str):
        self.response = response
        self.instruction: str | None = None
        self.input_text: str | None = None
        self.call_count = 0

    def get_client(self) -> object:
        return object()

    def get_ai_response(
        self, client: object, inst: str, input_text: str
    ) -> str:
        self.call_count += 1
        self.instruction = inst
        self.input_text = input_text
        input_payload = json.loads(input_text)
        if "comparisons" in input_payload:
            response_payload = json.loads(self.response)
            relation = response_payload["relation"]
            return json.dumps(
                {
                    "comparisons": [
                        {
                            "comparison_id": comparison["comparison_id"],
                            "relation": relation,
                        }
                        for comparison in input_payload["comparisons"]
                    ]
                }
            )
        return self.response


class StaticEmbeddingProvider:
    def embed_document(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class MappingEmbeddingProvider:
    def __init__(self, embeddings: dict[str, list[float]]):
        self.embeddings = embeddings

    def embed_document(self, text: str) -> list[float]:
        return self.embeddings[text]

    def embed_query(self, text: str) -> list[float]:
        return self.embeddings[text]


class CandidateAwareAI(RecordingAI):
    def __init__(self, equivalent_content: str):
        super().__init__('{"relation": "DISTINCT"}')
        self.equivalent_content = equivalent_content

    def get_ai_response(
        self, client: object, inst: str, input_text: str
    ) -> str:
        self.call_count += 1
        self.instruction = inst
        self.input_text = input_text
        payload = json.loads(input_text)
        if "comparisons" in payload:
            return json.dumps(
                {
                    "comparisons": [
                        {
                            "comparison_id": comparison["comparison_id"],
                            "relation": "EQUIVALENT"
                            if comparison["existing_fact"]
                            == self.equivalent_content
                            else "DISTINCT",
                        }
                        for comparison in payload["comparisons"]
                    ]
                }
            )
        relation = (
            "EQUIVALENT"
            if payload["existing_fact"] == self.equivalent_content
            else "DISTINCT"
        )
        return json.dumps({"relation": relation})


def extracted_result(*facts: str) -> JsonExtractedResult:
    payload = {
        "facts": [
            {"content": fact, "evidence": fact}
            for fact in facts
        ]
    }
    return JsonExtractedResult(
        json.dumps(payload),
        payload,
        tuple(ExtractedFact(content=fact, evidence=fact) for fact in facts),
    )


def sourced_result(*facts: tuple[str, str]) -> JsonExtractedResult:
    payload = {
        "facts": [
            {"content": content, "evidence": evidence}
            for content, evidence in facts
        ]
    }
    return JsonExtractedResult(
        json.dumps(payload),
        payload,
        tuple(
            ExtractedFact(content=content, evidence=evidence)
            for content, evidence in facts
        ),
    )


def source_event(source_id: str, content: str) -> SourceEvent:
    return SourceEvent(
        id=source_id,
        content=content,
        source_type="message",
        role="user",
        observed_at="2026-08-24T12:00:00+00:00",
    )


def remember_facts(backend: BackendV2, *facts: str) -> list[MemoryDecision]:
    source_content = " ".join(facts) or "No durable facts."
    return backend.remember(
        extracted_result(*facts),
        source_event=source_event(str(uuid.uuid4()), source_content),
    )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"relation": "EQUIVALENT"}', "EQUIVALENT"),
        ('{"relation": "SUPERSEDES"}', "SUPERSEDES"),
        ('{"relation": "DISTINCT"}', "DISTINCT"),
    ],
)
def test_classify_relationship_parses_llm_decision(
    tmp_path, response: str, expected: str
) -> None:
    ai = RecordingAI(response)
    backend = BackendV2(tmp_path / "memory.db", ai_provider=ai)

    result = backend._classify_relationship(
        "User likes programming in Rust.",
        "The user enjoys Rust programming.",
    )

    assert result == expected
    assert ai.instruction == RECONCILE_PROMPT
    assert json.loads(ai.input_text) == {
        "new_fact": "User likes programming in Rust.",
        "existing_fact": "The user enjoys Rust programming.",
    }


@pytest.mark.parametrize(
    "response",
    [
        "yes",
        "{}",
        "null",
        "[]",
        '{"equivalent": true}',
        '{"relation": "UNKNOWN"}',
        '{"relation": ["DISTINCT"]}',
        '{"relation": "EQUIVALENT", "reason": "same"}',
    ],
)
def test_classify_relationship_rejects_invalid_llm_output(
    tmp_path, response: str
) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI(response),
    )

    with pytest.raises(ValueError, match="LLM comparison response"):
        backend._classify_relationship("User likes Rust.", "User enjoys Rust.")


@pytest.mark.parametrize(("a", "b"), [("", "fact"), ("fact", "  ")])
def test_classify_relationship_rejects_empty_facts(tmp_path, a: str, b: str) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"relation": "EQUIVALENT"}'),
    )

    with pytest.raises(ValueError, match="non-empty string"):
        backend._classify_relationship(a, b)


def test_remember_adds_a_new_fact(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"relation": "DISTINCT"}'),
        embedding_provider=StaticEmbeddingProvider(),
    )

    decisions = remember_facts(backend, "User likes Rust.")

    assert decisions == [
        MemoryDecision(
            action="ADD",
            fact="User likes Rust.",
            memory_id=decisions[0].memory_id,
        )
    ]
    assert backend.get(decisions[0].memory_id).content == "User likes Rust."


def test_sourced_add_persists_exact_evidence(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"relation": "DISTINCT"}'),
        embedding_provider=StaticEmbeddingProvider(),
    )
    source = source_event("source-1", "I like Rust")

    decisions = backend.remember(
        sourced_result(("User likes Rust.", "I like Rust")),
        source_event=source,
    )

    assert decisions[0].action == "ADD"
    sources = backend.get_sources(decisions[0].memory_id)
    assert len(sources) == 1
    assert sources[0].source.id == "source-1"
    assert sources[0].source.content == "I like Rust"
    assert sources[0].evidence_quote == "I like Rust"
    assert sources[0].link_type == "DERIVED"


def test_sourced_noop_adds_confirming_evidence(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"relation": "EQUIVALENT"}'),
        embedding_provider=StaticEmbeddingProvider(),
    )
    memory_id = backend.sqlite_backend.insert("User likes Rust.")

    decision = backend.remember(
        sourced_result(("The user enjoys Rust.", "Rust is my favorite")),
        source_event=source_event("source-1", "Rust is my favorite"),
    )[0]
    backend.remember(
        sourced_result(("User likes Rust.", "I still like Rust")),
        source_event=source_event("source-2", "I still like Rust"),
    )

    assert decision.action == "NOOP"
    assert decision.memory_id == memory_id
    sources = backend.get_sources(memory_id)
    assert sorted(source.source.id for source in sources) == [
        "source-1",
        "source-2",
    ]
    assert [source.link_type for source in sources] == ["CONFIRMED", "CONFIRMED"]


def test_remember_returns_noop_for_equivalent_fact(tmp_path) -> None:
    ai = RecordingAI('{"relation": "EQUIVALENT"}')
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=ai,
        embedding_provider=StaticEmbeddingProvider(),
    )
    existing_id = backend.sqlite_backend.insert("User likes Rust.")

    decisions = remember_facts(
        backend,
        "The user enjoys programming in Rust.",
    )

    assert decisions == [
        MemoryDecision(
            action="NOOP",
            fact="The user enjoys programming in Rust.",
            memory_id=existing_id,
            matched_memory_id=existing_id,
        )
    ]
    assert len(backend.list_memories()) == 1
    assert ai.call_count == 1


def test_remember_supersedes_an_outdated_memory(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"relation": "SUPERSEDES"}'),
        embedding_provider=StaticEmbeddingProvider(),
    )
    old_id = backend.sqlite_backend.insert("User lives in Shanghai.")

    decisions = remember_facts(backend, "User now lives in Beijing.")

    assert decisions == [
        MemoryDecision(
            action="SUPERSEDE",
            fact="User now lives in Beijing.",
            memory_id=decisions[0].memory_id,
            matched_memory_id=old_id,
        )
    ]
    new_id = decisions[0].memory_id
    assert new_id is not None

    old_memory = backend.get(old_id)
    new_memory = backend.get(new_id)
    assert old_memory is not None
    assert old_memory.state == "SUPERSEDED"
    assert old_memory.superseded_by_id == new_id
    assert old_memory.superseded_at is not None
    assert new_memory is not None
    assert new_memory.state == "ACTIVE"

    assert [result.memory.id for result in backend.search_current("where user lives")] == [
        new_id
    ]
    assert [result.memory.id for result in backend.search_history("where user lived")] == [
        old_id
    ]
    assert len(backend.list_memories()) == 2


def test_sourced_supersede_keeps_old_and_new_evidence_separate(tmp_path) -> None:
    ai = RecordingAI('{"relation": "DISTINCT"}')
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=ai,
        embedding_provider=StaticEmbeddingProvider(),
    )
    old_decision = backend.remember(
        sourced_result(("User lives in Shanghai.", "I live in Shanghai")),
        source_event=source_event("old-source", "I live in Shanghai"),
    )[0]
    ai.response = '{"relation": "SUPERSEDES"}'

    new_decision = backend.remember(
        sourced_result(("User now lives in Beijing.", "I moved to Beijing")),
        source_event=source_event("new-source", "I moved to Beijing"),
    )[0]

    assert new_decision.action == "SUPERSEDE"
    assert new_decision.matched_memory_id == old_decision.memory_id
    assert [
        source.source.id
        for source in backend.get_sources(old_decision.memory_id)
    ] == ["old-source"]
    assert [
        source.source.id
        for source in backend.get_sources(new_decision.memory_id)
    ] == ["new-source"]


def test_sourced_batch_chains_repeated_supersessions(tmp_path) -> None:
    database_path = tmp_path / "memory.db"
    backend = BackendV2(
        database_path,
        ai_provider=RecordingAI('{"relation": "SUPERSEDES"}'),
        embedding_provider=StaticEmbeddingProvider(),
        candidate_limit=1,
    )
    old_id = backend.sqlite_backend.insert("User lives in Shanghai.")
    source = source_event(
        "source-rollback",
        "I moved to Beijing and then to Shenzhen",
    )

    decisions = backend.remember(
        sourced_result(
            ("User lives in Beijing.", "moved to Beijing"),
            ("User lives in Shenzhen.", "then to Shenzhen"),
        ),
        source_event=source,
    )

    old_memory = backend.get(old_id)
    assert old_memory is not None
    beijing_id = decisions[0].memory_id
    shenzhen_id = decisions[1].memory_id
    assert beijing_id is not None
    assert shenzhen_id is not None
    beijing_memory = backend.get(beijing_id)
    shenzhen_memory = backend.get(shenzhen_id)

    assert old_memory.state == "SUPERSEDED"
    assert old_memory.superseded_by_id == beijing_id
    assert beijing_memory is not None
    assert beijing_memory.state == "SUPERSEDED"
    assert beijing_memory.superseded_by_id == shenzhen_id
    assert shenzhen_memory is not None
    assert shenzhen_memory.state == "ACTIVE"
    assert decisions[1].matched_memory_id == beijing_id
    connection = sqlite3.connect(database_path)
    source_count = connection.execute(
        "SELECT COUNT(*) FROM source_events"
    ).fetchone()[0]
    connection.close()
    assert source_count == 1


def test_reconciliation_compares_only_with_current_memories(tmp_path) -> None:
    ai = RecordingAI('{"relation": "SUPERSEDES"}')
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=ai,
        embedding_provider=StaticEmbeddingProvider(),
        candidate_limit=5,
    )
    old_id = backend.sqlite_backend.insert("User lived in Shanghai.")
    current_id = backend.sqlite_backend.supersede(
        old_id,
        "User lives in Beijing.",
    )

    decisions = remember_facts(backend, "User now lives in Shenzhen.")

    assert decisions[0].action == "SUPERSEDE"
    assert decisions[0].matched_memory_id == current_id
    assert ai.call_count == 1
    assert ai.instruction == BATCH_RECONCILE_PROMPT
    assert json.loads(ai.input_text)["comparisons"][0]["existing_fact"] == (
        "User lives in Beijing."
    )


def test_remember_adds_distinct_but_similar_fact(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"relation": "DISTINCT"}'),
        embedding_provider=StaticEmbeddingProvider(),
    )
    existing_id = backend.sqlite_backend.insert("User likes Rust.")

    decisions = remember_facts(backend, "User dislikes Rust.")

    assert decisions[0].action == "ADD"
    assert decisions[0].memory_id != existing_id
    assert len(backend.list_memories()) == 2


def test_remember_checks_more_than_the_first_candidate(tmp_path) -> None:
    equivalent = "The user enjoys Rust programming."
    new_fact = "User likes programming in Rust."
    embeddings = {
        "User attended a Rust conference.": [1.0, 0.0],
        equivalent: [0.95, 0.05],
        new_fact: [1.0, 0.0],
    }
    ai = CandidateAwareAI(equivalent)
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=ai,
        embedding_provider=MappingEmbeddingProvider(embeddings),
    )
    backend.sqlite_backend.insert("User attended a Rust conference.")
    equivalent_id = backend.sqlite_backend.insert(equivalent)

    decisions = remember_facts(backend, new_fact)

    assert decisions[0].action == "NOOP"
    assert decisions[0].matched_memory_id == equivalent_id
    assert ai.call_count == 1
    assert len(backend.list_memories()) == 2


def test_remember_batches_multiple_fact_comparisons_into_one_llm_call(
    tmp_path,
) -> None:
    ai = RecordingAI('{"relation": "DISTINCT"}')
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=ai,
        embedding_provider=StaticEmbeddingProvider(),
        candidate_limit=1,
    )
    backend.sqlite_backend.insert("User likes Python.")

    decisions = remember_facts(
        backend,
        "User likes Rust.",
        "User likes green tea.",
    )

    assert [decision.action for decision in decisions] == ["ADD", "ADD"]
    assert ai.call_count == 1
    assert ai.instruction == BATCH_RECONCILE_PROMPT
    payload = json.loads(ai.input_text)
    assert len(payload["comparisons"]) == 2


def test_remember_handles_multiple_facts(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"relation": "DISTINCT"}'),
        embedding_provider=StaticEmbeddingProvider(),
    )

    decisions = remember_facts(
        backend,
        "User likes Rust.",
        "User likes green tea.",
    )

    assert [decision.action for decision in decisions] == ["ADD", "ADD"]
    assert len(backend.list_memories()) == 2


def test_remember_requires_v2_extracted_result(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"relation": "DISTINCT"}'),
        embedding_provider=StaticEmbeddingProvider(),
    )

    with pytest.raises(TypeError, match="JsonExtractedResult"):
        backend.remember(
            SimpleExtractedResult("User likes Rust."),
            source_event=source_event("source", "User likes Rust."),
        )


def test_remember_requires_embedding_provider(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"relation": "DISTINCT"}'),
    )

    with pytest.raises(RuntimeError, match="embedding_provider"):
        remember_facts(backend, "User likes Rust.")


def test_remember_returns_empty_decisions_for_no_facts(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"relation": "DISTINCT"}'),
        embedding_provider=StaticEmbeddingProvider(),
    )

    assert remember_facts(backend) == []


def test_empty_sourced_extraction_does_not_store_source(tmp_path) -> None:
    database_path = tmp_path / "memory.db"
    backend = BackendV2(
        database_path,
        ai_provider=RecordingAI('{"relation": "DISTINCT"}'),
        embedding_provider=StaticEmbeddingProvider(),
    )

    assert backend.remember(
        sourced_result(),
        source_event=source_event("unused-source", "Hello"),
    ) == []

    connection = sqlite3.connect(database_path)
    source_count = connection.execute(
        "SELECT COUNT(*) FROM source_events"
    ).fetchone()[0]
    connection.close()
    assert source_count == 0


def test_decide_does_not_write_before_decision_is_applied(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"relation": "DISTINCT"}'),
        embedding_provider=StaticEmbeddingProvider(),
    )

    decision = backend._decide("User likes Rust.", [])

    assert decision == MemoryDecision(action="ADD", fact="User likes Rust.")
    assert backend.list_memories() == []

    completed = backend._apply_decision(decision)

    assert completed.action == "ADD"
    assert completed.memory_id is not None
    assert backend.get(completed.memory_id).content == "User likes Rust."


def test_apply_noop_resolves_to_the_matched_memory(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"relation": "EQUIVALENT"}'),
        embedding_provider=StaticEmbeddingProvider(),
    )
    existing_id = backend.sqlite_backend.insert("User likes Rust.")
    existing_memory = backend.get(existing_id)
    assert existing_memory is not None

    decision = backend._decide(
        "The user enjoys Rust programming.",
        [ScoredMemory(memory=existing_memory, similarity=1.0)],
    )

    assert decision == MemoryDecision(
        action="NOOP",
        fact="The user enjoys Rust programming.",
        matched_memory_id=existing_id,
    )

    completed = backend._apply_decision(decision)

    assert completed.memory_id == existing_id
    assert len(backend.list_memories()) == 1


def test_apply_supersede_requires_a_matched_memory(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"relation": "SUPERSEDES"}'),
        embedding_provider=StaticEmbeddingProvider(),
    )

    with pytest.raises(ValueError, match="matched_memory_id"):
        backend._apply_decision(
            MemoryDecision(action="SUPERSEDE", fact="User moved.")
        )


def test_apply_decision_rejects_an_already_applied_decision(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"relation": "DISTINCT"}'),
        embedding_provider=StaticEmbeddingProvider(),
    )
    completed = backend._apply_decision(
        MemoryDecision(action="ADD", fact="User likes Rust.")
    )

    with pytest.raises(ValueError, match="already been applied"):
        backend._apply_decision(completed)
