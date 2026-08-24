from memkernel.extractor.extractor_v2 import LLMExtractorV2

ex = LLMExtractorV2()
result = ex.extract(
    "Today I get up late.Eat a hamberger. And i got to the library.I like rust.  I know our teacher is John. Last day our teacher guided us a tour."
)

print(result.content)
