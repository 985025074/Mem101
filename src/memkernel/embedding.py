from collections.abc import Sequence
from os import environ
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI


class EmbeddingProvider(Protocol):
    def embed_document(self, text: str) -> Sequence[float]: ...

    def embed_query(self, text: str) -> Sequence[float]: ...


class OpenAIEmbeddingProvider:
    DOCUMENT_PREFIX = "search_document: "
    QUERY_PREFIX = "search_query: "

    @staticmethod
    def get_client():
        load_dotenv()
        client = OpenAI(
            base_url=environ.get("EMBEDDING_BASE"),
            api_key="ollama",  # Required by the SDK, ignored
        )
        return client

    def __init__(self, client: OpenAI, model: str = "nomic-embed-text:latest"):
        self.client = client
        self.model = model

    def _embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def embed_document(self, text: str) -> list[float]:
        return self._embed(f"{self.DOCUMENT_PREFIX}{text}")

    def embed_query(self, text: str) -> list[float]:
        return self._embed(f"{self.QUERY_PREFIX}{text}")
