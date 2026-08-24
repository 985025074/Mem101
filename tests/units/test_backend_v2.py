import json

import pytest

from memkernel.backend.backend import MemoryDecision
from memkernel.backend.backend_v2 import BackendV2, COMPARE_PROMPT
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
        super().__init__('{"equivalent": false}')
        self.equivalent_content = equivalent_content

    def get_ai_response(
        self, client: object, inst: str, input_text: str
    ) -> str:
        self.call_count += 1
        payload = json.loads(input_text)
        equivalent = payload["fact_b"] == self.equivalent_content
        return json.dumps({"equivalent": equivalent})


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
def test_compare_two_things_parses_llm_decision(
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
        '{"equivalent": "true"}',
        '{"equivalent": true, "reason": "same"}',
    ],
)
def test_compare_two_things_rejects_invalid_llm_output(
    tmp_path, response: str
) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI(response),
    )

    with pytest.raises(ValueError, match="LLM comparison response"):
        backend._compare_two_things("User likes Rust.", "User enjoys Rust.")


@pytest.mark.parametrize(("a", "b"), [("", "fact"), ("fact", "  ")])
def test_compare_two_things_rejects_empty_facts(tmp_path, a: str, b: str) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"equivalent": true}'),
    )

    with pytest.raises(ValueError, match="non-empty string"):
        backend._compare_two_things(a, b)


def test_remember_adds_a_new_fact(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"equivalent": false}'),
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
    ai = RecordingAI('{"equivalent": true}')
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


def test_remember_adds_distinct_but_similar_fact(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"equivalent": false}'),
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
        ai_provider=RecordingAI('{"equivalent": false}'),
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
        ai_provider=RecordingAI('{"equivalent": false}'),
        embedding_provider=StaticEmbeddingProvider(),
    )

    with pytest.raises(TypeError, match="JsonExtractedResult"):
        backend.remember(SimpleExtractedResult("User likes Rust."))


def test_remember_requires_embedding_provider(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"equivalent": false}'),
    )

    with pytest.raises(RuntimeError, match="embedding_provider"):
        backend.remember(extracted_result("User likes Rust."))


def test_remember_returns_empty_decisions_for_no_facts(tmp_path) -> None:
    backend = BackendV2(
        tmp_path / "memory.db",
        ai_provider=RecordingAI('{"equivalent": false}'),
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
        ai_provider=RecordingAI('{"equivalent": false}'),
        embedding_provider=StaticEmbeddingProvider(),
    )
    extracted = JsonExtractedResult(json.dumps(payload), payload)

    with pytest.raises(ValueError, match="facts|fact"):
        backend.remember(extracted)
