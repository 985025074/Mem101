import json

from openai import OpenAI
import pytest

from memkernel.extractor import LLMExtractor
from unittest.mock import Mock

from memkernel.extractor.extractor_v2 import (
    ExtractionValidationError,
    ExtractedFact,
    JsonExtractedResult,
    LLMExtractorV2,
)
from memkernel.provenance import SourceEvent


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.input_text: str | None = None

    def get_ai_response(self, client: OpenAI, inst: str, input_text: str) -> str:
        self.input_text = input_text
        return self.response

    @staticmethod
    def get_client() -> OpenAI:
        return Mock(spec=OpenAI)


def test_extractor_easy():
    extractor = LLMExtractor()
    result = extractor.extract("I like Kokona.")


def test_extractor_v2_easy():
    payload = {
        "facts": [
            {
                "content": "User likes Rust.",
                "evidence": "I like Rust",
            }
        ]
    }
    llm = FakeLLM(json.dumps(payload))
    extractor = LLMExtractorV2(llm)
    source = SourceEvent(
        id="source-id",
        content="I like Rust",
        source_type="message",
        role="user",
        observed_at="2026-08-24T12:00:00+00:00",
    )

    result = extractor.extract_with_source(source)

    assert isinstance(result, JsonExtractedResult)
    assert result.parsed_dict == payload
    assert result.facts == (
        ExtractedFact(content="User likes Rust.", evidence="I like Rust"),
    )
    assert json.loads(llm.input_text) == {
        "source": {
            "content": "I like Rust",
            "source_type": "message",
            "role": "user",
            "observed_at": "2026-08-24T12:00:00+00:00",
        }
    }


def test_extractor_rejects_evidence_not_found_in_source() -> None:
    payload = {
        "facts": [
            {
                "content": "User likes Rust.",
                "evidence": "User enjoys Rust",
            }
        ]
    }
    extractor = LLMExtractorV2(FakeLLM(json.dumps(payload)))

    with pytest.raises(ExtractionValidationError, match="exact"):
        extractor.extract("I like Rust")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"facts": "not-a-list"},
        {"facts": ["User likes Rust."]},
        {"facts": [{"content": "User likes Rust."}]},
    ],
)
def test_extractor_rejects_invalid_fact_schema(payload: object) -> None:
    extractor = LLMExtractorV2(FakeLLM(json.dumps(payload)))

    with pytest.raises(ExtractionValidationError):
        extractor.extract("I like Rust")
