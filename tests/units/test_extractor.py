import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest

from memkernel.extractor.extractor_v2 import EXTRACTOR_PROMPT, LLMExtractorV2


class RecordingAI:
    def __init__(self, response: str):
        self.response = response
        self.instruction: str | None = None
        self.input_text: str | None = None

    def get_client(self) -> object:
        return object()

    def get_ai_response(
        self, client: object, inst: str, input_text: str
    ) -> str:
        self.instruction = inst
        self.input_text = input_text
        return self.response


def parse_facts(content: str) -> list[str]:
    payload = json.loads(content)

    assert set(payload) == {"facts"}
    assert isinstance(payload["facts"], list)
    assert all(isinstance(fact, str) and fact.strip() for fact in payload["facts"])

    return payload["facts"]


def test_extractor_v2_uses_its_prompt_and_returns_json() -> None:
    ai = RecordingAI('{"facts": ["User likes Rust."]}')
    extractor = LLMExtractorV2(llm=ai)

    result = extractor.extract("I like Rust.")

    assert ai.instruction == EXTRACTOR_PROMPT
    assert ai.input_text == "I like Rust."
    assert parse_facts(result.content) == ["User likes Rust."]


@dataclass(frozen=True)
class ExtractionCase:
    name: str
    input_text: str
    required_term_groups: tuple[tuple[str, ...], ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    expected_empty: bool = False
    minimum_fact_count: int = 1
    require_cjk_output: bool = False


tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

EXTRACTION_CASES = (
    ExtractionCase(
        name="small-talk",
        input_text="Hello! How are you today?",
        expected_empty=True,
        minimum_fact_count=0,
    ),
    ExtractionCase(
        name="preference-and-project",
        input_text="I like Rust and I am building a small memory system.",
        required_term_groups=(("rust",), ("memory system",)),
        minimum_fact_count=2,
    ),
    ExtractionCase(
        name="multiple-personal-facts",
        input_text=(
            "My name is Alice. I work as a backend engineer, and I prefer tea "
            "over coffee."
        ),
        required_term_groups=(("alice",), ("backend",), ("tea",)),
        minimum_fact_count=3,
    ),
    ExtractionCase(
        name="negated-preference",
        input_text="I do not like coffee, but I enjoy green tea.",
        required_term_groups=(
            ("does not like coffee", "doesn't like coffee", "dislikes coffee"),
            ("green tea",),
        ),
    ),
    ExtractionCase(
        name="corrected-preference",
        input_text=(
            "I used to prefer Vim, but I switched to Neovim and now prefer "
            "Neovim."
        ),
        required_term_groups=(("neovim",), ("prefer",)),
    ),
    ExtractionCase(
        name="relative-date",
        input_text="Tomorrow I will fly to Shanghai for a conference.",
        required_term_groups=((tomorrow,), ("shanghai",), ("conference",)),
    ),
    ExtractionCase(
        name="generic-knowledge",
        input_text="The Earth orbits the Sun.",
        expected_empty=True,
        minimum_fact_count=0,
    ),
    ExtractionCase(
        name="secret-filtering",
        input_text="My API key is mk-secret-9f4c2. Please remember it.",
        forbidden_terms=("mk-secret-9f4c2",),
        expected_empty=True,
        minimum_fact_count=0,
    ),
    ExtractionCase(
        name="chinese-input",
        input_text="我喜欢 Rust，正在开发一个小型记忆系统。",
        required_term_groups=(("rust",), ("记忆",)),
        minimum_fact_count=2,
        require_cjk_output=True,
    ),
)


@pytest.mark.skipif(
    os.getenv("RUN_LLM_TESTS") != "1",
    reason="Set RUN_LLM_TESTS=1 to run tests that call the configured LLM.",
)
@pytest.mark.parametrize("case", EXTRACTION_CASES, ids=lambda case: case.name)
def test_extractor_v2_live(case: ExtractionCase) -> None:
    facts = parse_facts(LLMExtractorV2().extract(case.input_text).content)
    combined = " ".join(facts).casefold()

    if case.expected_empty:
        assert facts == []
    else:
        assert len(facts) >= case.minimum_fact_count

    for alternatives in case.required_term_groups:
        assert any(term.casefold() in combined for term in alternatives), (
            f"Expected one of {alternatives!r} in extracted facts: {facts!r}"
        )

    for forbidden in case.forbidden_terms:
        assert forbidden.casefold() not in combined

    if case.require_cjk_output:
        assert any("\u4e00" <= character <= "\u9fff" for character in combined)
