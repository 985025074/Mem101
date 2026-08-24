from dataclasses import dataclass
from typing import Protocol

from memkernel.ai import AIProvider, DeepSeekAI


class ExtractedResult(Protocol):
    # content: str
    # in protocol if we want to define a field use method instead of @dataclass
    @property
    def content(self) -> str: ...


@dataclass(slots=True, frozen=True)
class SimpleExtractedResult:
    content: str


class Extractor(Protocol):
    def extract(self, given_info: str) -> ExtractedResult: ...


# Prototype
class LLMExtractor:
    def __init__(self, llm: AIProvider = DeepSeekAI()):
        self.llm: AIProvider = llm
        # TODO: Do this
        self.llm_prompt = "please extract useful information from the given message."
        self.client = self.llm.get_client()

    def extract(self, given_info: str) -> ExtractedResult:
        result = self.llm.get_ai_response(
            self.client, inst=self.llm_prompt, input_text=given_info
        )
        return SimpleExtractedResult(result)
