import json

from openai import OpenAI

from memkernel.extractor import LLMExtractor
from unittest.mock import Mock

from memkernel.extractor.extractor_v2 import JsonExtractedResult, LLMExtractorV2


class FakeLLM:
    def get_ai_response(self, client: OpenAI, inst: str, input_text: str) -> str:
        return input_text

    @staticmethod
    def get_client() -> OpenAI:
        return Mock(spec=OpenAI)


def test_extractor_easy():
    extractor = LLMExtractor()
    result = extractor.extract("I like Kokona.")


def test_extractor_v2_easy():
    dict_ = {"1234": 1234, "2234": 2234}
    dumped_dict = json.dumps(dict_)

    extractor = LLMExtractorV2(FakeLLM())

    result = extractor.extract(dumped_dict)
    assert isinstance(result, JsonExtractedResult)
    assert dict_ == result.parsed_dict
