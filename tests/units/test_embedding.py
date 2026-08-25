from types import SimpleNamespace
from unittest.mock import Mock, call

from memkernel.embedding import OpenAIEmbeddingProvider


def test_nomic_provider_adds_retrieval_task_prefixes() -> None:
    client = Mock()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(embedding=[1.0, 0.0])]
    )
    provider = OpenAIEmbeddingProvider(client)

    assert provider.embed_document("Jenny likes apples.") == [1.0, 0.0]
    assert provider.embed_query("Who likes apples?") == [1.0, 0.0]

    assert client.embeddings.create.call_args_list == [
        call(
            model="nomic-embed-text:latest",
            input="search_document: Jenny likes apples.",
        ),
        call(
            model="nomic-embed-text:latest",
            input="search_query: Who likes apples?",
        ),
    ]
