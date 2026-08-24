import json
from typing import Any, Dict
from dataclasses import dataclass

from memkernel.ai import AIProvider, DeepSeekAI
from memkernel.utility.time_helper import now
from .extractor import ExtractedResult


# replace time with real time.
# TODO: add last 20 messages to prompt to help performance?
EXTRACTOR_PROMPT = f"""
You are MemKernel's memory extractor. Convert the input message into a small
set of durable facts that may be useful in future conversations.

Extract only information explicitly supported by the input, including:
- stable preferences, dislikes, and personal details;
- relationships, roles, skills, and ongoing work;
- plans, goals, commitments, and decisions;
- meaningful events, experiences, constraints, and corrections.

Rules:
1. Write one atomic, self-contained fact per list item.
2. Resolve pronouns only when the referenced person or object is unambiguous.
3. Preserve important qualifiers, negations, and time expressions.
4. Do not invent details, explanations, implications, or personality traits.
5. Ignore greetings, small talk, generic knowledge, and temporary requests
   that will not be useful later.
6. Do not store passwords, API keys, authentication tokens, or other secrets.
7. Remove duplicate or semantically equivalent facts.
8. Write facts in the same language as the input.
9. You should replace some ambiguous time to real time of it. Today is {now()}.

Return valid JSON only, using exactly this schema:
    {{"facts": ["first fact", "second fact"]}}

If the input contains nothing worth remembering, return:
    {{"facts": []}}

Examples:
Input: Hi, how are you?
Output: {{"facts": []}}

Input: I like Rust and I am building a small memory system.
Output: {{"facts": ["User likes Rust.", "User is building a small memory system."]}}
""".strip()


@dataclass(slots=True, frozen=True)
class JsonExtractedResult:
    content: str
    parsed_dict: Dict[Any, Any]


class LLMExtractorV2:
    def __init__(self, llm: AIProvider = DeepSeekAI()):
        self.llm: AIProvider = llm
        self.llm_prompt = EXTRACTOR_PROMPT
        self.client = self.llm.get_client()

    def extract(self, given_info: str) -> ExtractedResult:
        result = self.llm.get_ai_response(
            self.client, inst=self.llm_prompt, input_text=given_info
        )
        parsed_dict = json.loads(result)
        return JsonExtractedResult(
            result,
            parsed_dict,
        )
