from os import environ
from typing import Protocol
from openai import OpenAI
from dotenv import load_dotenv
from openai.types import ReasoningEffort


class AIProvider(Protocol):
    def get_ai_response(self, client: OpenAI, inst: str, input_text: str) -> str: ...
    @staticmethod
    def get_client() -> OpenAI: ...


class DeepSeekAI:
    model = "deepseek-v4-flash"
    effort: ReasoningEffort = "max"

    @staticmethod
    def get_client():
        load_dotenv()
        client = OpenAI(
            api_key=environ.get("API_KEY"), base_url=environ.get("API_BASE")
        )
        return client

    def get_ai_response(self, client: OpenAI, inst: str, input_text: str):
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": inst},
                {"role": "user", "content": input_text},
            ],
            reasoning_effort=self.effort,
        )
        if response.choices[0].message.content is None:
            raise Exception("LLM backend error.")
        return response.choices[0].message.content
