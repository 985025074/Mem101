import json

import pytest

from memkernel.backend.backend_v2 import BackendV2, COMPARE_PROMPT


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
