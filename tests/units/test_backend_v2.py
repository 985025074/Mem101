import json

import pytest

from memkernel.backend.backend import MemoryDecision, ScoredMemory
from memkernel.backend.backend_v2 import (
    BackendV2,
    COMPARE_PROMPT,
    RECONCILE_PROMPT,
)
from memkernel.extractor.extractor import SimpleExtractedResult
from memkernel.extractor.extractor_v2 import JsonExtractedResult


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
        return self.response


class StaticEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


class MappingEmbeddingProvider:
    def __init__(self, embeddings: dict[str, list[float]]):
        self.embeddings = embeddings

    def embed(self, text: str) -> list[float]:
        return self.embeddings[text]


class CandidateAwareAI(RecordingAI):
    def __init__(self, equivalent_content: str):
        super().__init__('{"relation": "DISTINCT"}')
        self.equivalent_content = equivalent_content

    def get_ai_response(
        self, client: object, inst: str, input_text: str
    ) -> str:
        self.call_count += 1
        payload = json.loads(input_text)
        relation = (
            "EQUIVALENT"
            if payload["existing_fact"] == self.equivalent_content
            else "DISTINCT"
        )
        return json.dumps({"relation": relation})


def extracted_result(*facts: str) -> JsonExtractedResult:
    payload = {"facts": list(facts)}
    return JsonExtractedResult(json.dumps(payload), payload)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"equivalent": true}', True),
        ('{"equivalent": false}', False),
    ],
)
def test_compare_two_things_preserves_boolean_contract(
    tmp_path, response: str, expected: bool
) -> None:
    ai = RecordingAI(response)
    backend = BackendV2(tmp_path / "memory.db", ai_provider=ai)

    result = backend._compare_two_things(
        "User likes programming in Rust.",
        "The user enjoys Rust programming.",
    )

    assert result is expected
    assert ai.instruction == COMPARE_PROMPT
    assert json.loads(ai.input_text) == {
        "fact_a": "User likes programming in Rust.",
        "fact_b": "The user enjoys Rust programming.",
    }


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


def test_compare_two_things_skips_llm_for_exact_normalized_match(tmp_path) -> None:
    ai = RecordingAI('{"equivalent": false}')
    backend = BackendV2(tmp_path / "memory.db", ai_provider=ai)

    assert backend._compare_two_things(" User likes Rust. ", "user likes rust.")
    assert ai.call_count == 0


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

    decisions = backend.remember(extracted_result("User likes Rust."))

    assert decisions == [
        MemoryDecision(
            action="ADD",
            fact="User likes Rust.",
            memory_id=decisions[0].memory_id,
        )
    ]
    assert backend.get(decisions[0].memory_id).content == "User likes Rust."


def test_remember_returns_noop_for_equivalent_fact(tmp_path) -> None:
    ai = RecordingAI('{"relation": "EQUIVALENT"}')
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=ai,
        embedding_provider=StaticEmbeddingProvider(),
    )
    existing_id = backend.sqlite_backend.insert("User likes Rust.")

    decisions = backend.remember(
        extracted_result("The user enjoys programming in Rust.")
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

    decisions = backend.remember(extracted_result("User now lives in Beijing."))

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

    decisions = backend.remember(extracted_result("User now lives in Shenzhen."))

    assert decisions[0].action == "SUPERSEDE"
    assert decisions[0].matched_memory_id == current_id
    assert ai.call_count == 1
    assert json.loads(ai.input_text)["existing_fact"] == "User lives in Beijing."


def test_remember_adds_distinct_but_similar_fact(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"relation": "DISTINCT"}'),
        embedding_provider=StaticEmbeddingProvider(),
    )
    existing_id = backend.sqlite_backend.insert("User likes Rust.")

    decisions = backend.remember(extracted_result("User dislikes Rust."))

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

    decisions = backend.remember(extracted_result(new_fact))

    assert decisions[0].action == "NOOP"
    assert decisions[0].matched_memory_id == equivalent_id
    assert ai.call_count == 2
    assert len(backend.list_memories()) == 2


def test_remember_handles_multiple_facts(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"relation": "DISTINCT"}'),
        embedding_provider=StaticEmbeddingProvider(),
    )

    decisions = backend.remember(
        extracted_result("User likes Rust.", "User likes green tea.")
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
        backend.remember(SimpleExtractedResult("User likes Rust."))


def test_remember_requires_embedding_provider(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"relation": "DISTINCT"}'),
    )

    with pytest.raises(RuntimeError, match="embedding_provider"):
        backend.remember(extracted_result("User likes Rust."))


def test_remember_returns_empty_decisions_for_no_facts(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"relation": "DISTINCT"}'),
        embedding_provider=StaticEmbeddingProvider(),
    )

    assert backend.remember(extracted_result()) == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"facts": "User likes Rust."},
        {"facts": [""]},
        {"facts": [123]},
    ],
)
def test_remember_rejects_invalid_fact_lists(tmp_path, payload) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"relation": "DISTINCT"}'),
        embedding_provider=StaticEmbeddingProvider(),
    )
    extracted = JsonExtractedResult(json.dumps(payload), payload)

    with pytest.raises(ValueError, match="facts|fact"):
        backend.remember(extracted)


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
