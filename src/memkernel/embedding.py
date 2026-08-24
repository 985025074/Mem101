from collections.abc import Sequence
from os import environ
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> Sequence[float]: ...


class OpenAIEmbeddingProvider:
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

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding
