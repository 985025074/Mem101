import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from memkernel.backend.backend_v2 import BackendV2
from memkernel.embedding import OpenAIEmbeddingProvider
from memkernel.extractor.extractor_v2 import LLMExtractorV2


def configure_logging() -> None:
    package_logger = logging.getLogger("memkernel")
    package_logger.setLevel(logging.DEBUG)
    package_logger.propagate = False

    if package_logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    package_logger.addHandler(handler)


def main() -> None:
    configure_logging()
    embedding_provider = OpenAIEmbeddingProvider(OpenAIEmbeddingProvider.get_client())
    extractor = LLMExtractorV2()

    with TemporaryDirectory() as temporary_directory:
        backend = BackendV2(
            memory_path=Path(temporary_directory) / "memkernel-test.db",
            embedding_provider=embedding_provider,
        )

        for message in ("我是男的", "我是女的", "我喜欢西瓜"):
            extracted = extractor.extract(message)
            print(f"Input: {message}")
            print(f"Extracted facts: {extracted.parsed_dict['facts']}")
            backend.remember(extracted)

        print("Stored memories:")
        for memory in backend.list_memories():
            print(f"- {memory.content}")


if __name__ == "__main__":
    main()
