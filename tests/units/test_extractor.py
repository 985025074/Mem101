from memkernel.extractor import LLMExtractor


def test_extractor_easy():
    extractor = LLMExtractor()
    result = extractor.extract("I like Kokona.")
