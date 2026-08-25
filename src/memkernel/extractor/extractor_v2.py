from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from memkernel.ai import AIProvider, DeepSeekAI
from memkernel.provenance import SourceEvent


EXTRACTOR_PROMPT = """
You are MemKernel's evidence-bound memory extractor. Convert the supplied
source event into a small set of durable facts that may be useful later.

Extract only information asserted or confirmed by source.content. The optional
recent_context contains earlier messages, ordered oldest to newest. Use it only
to resolve references, pronouns, ellipsis, confirmations, and corrections in
source.content. Never extract a fact supported only by recent_context.

Durable information includes:
- stable preferences, dislikes, and personal details;
- relationships, roles, skills, and ongoing work;
- plans, goals, commitments, and decisions;
- meaningful events, experiences, constraints, and corrections.

Rules:
1. Write one atomic, self-contained fact per item.
2. For every fact, copy a non-empty evidence quote exactly as it appears in
   source.content. Never paraphrase the evidence quote.
3. Resolve pronouns and references using recent_context only when the referent
   is unambiguous. Otherwise, omit the fact.
4. Preserve qualifiers, negations, quantities, and time expressions.
5. Resolve relative time using source.observed_at, not the current clock.
6. Do not invent details, implications, explanations, or personality traits.
7. Ignore greetings, generic knowledge, and temporary requests that will not
   be useful later.
8. Do not extract passwords, keys, authentication tokens, or redacted values.
9. Remove duplicate or semantically equivalent facts.
10. Write facts in the same language as source.content.

Return valid JSON only, using exactly this schema:
{"facts": [{"content": "one durable fact", "evidence": "exact quote"}]}

If the source contains nothing worth remembering, return:
{"facts": []}
""".strip()


MAX_RECENT_CONTEXT_MESSAGES = 6
RECENT_CONTEXT_ROLES = {"user", "assistant", "system", "tool"}


class ExtractionValidationError(ValueError):
    """The extractor returned output that cannot be safely persisted."""


@dataclass(frozen=True, slots=True)
class ExtractedFact:
    content: str
    evidence: str


@dataclass(frozen=True, slots=True)
class JsonExtractedResult:
    content: str
    parsed_dict: dict[str, Any]
    facts: tuple[ExtractedFact, ...] = ()


def parse_extracted_facts(
    payload: object,
    *,
    source_content: str,
) -> tuple[ExtractedFact, ...]:
    """we will make sure the final json's evidence from source content"""
    if not isinstance(payload, dict) or set(payload) != {"facts"}:
        raise ExtractionValidationError(
            'Extractor response must be an object containing only "facts"'
        )

    raw_facts = payload["facts"]
    #
    if not isinstance(raw_facts, list):
        raise ExtractionValidationError('Extractor response "facts" must be a list')

    facts: list[ExtractedFact] = []
    for raw_fact in raw_facts:
        # check if key is right. and it is a dict
        if not isinstance(raw_fact, dict) or set(raw_fact) != {
            "content",
            "evidence",
        }:
            raise ExtractionValidationError(
                "Each extracted fact must contain only content and evidence"
            )

        content = raw_fact["content"]
        evidence = raw_fact["evidence"]

        if not isinstance(content, str) or not content.strip():
            raise ExtractionValidationError(
                "Extracted fact content must be a non-empty string"
            )
        if not isinstance(evidence, str) or not evidence.strip():
            raise ExtractionValidationError(
                "Extracted fact evidence must be a non-empty string"
            )
        # make sure evidence come from source
        if evidence not in source_content:
            raise ExtractionValidationError(
                "Extracted evidence must be an exact source-content substring"
            )

        facts.append(
            ExtractedFact(
                content=content.strip(),
                evidence=evidence,
            )
        )

    return tuple(facts)


def parse_recent_context(metadata: dict[str, Any]) -> list[dict[str, str]]:
    """Validate and bound context supplied for reference resolution."""
    raw_context = metadata.get("recent_context", [])
    if not isinstance(raw_context, list):
        raise ExtractionValidationError(
            "source metadata recent_context must be a list"
        )

    recent_context: list[dict[str, str]] = []
    for message in raw_context[-MAX_RECENT_CONTEXT_MESSAGES:]:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ExtractionValidationError(
                "Every recent_context message must contain only role and content"
            )
        role = message["role"]
        content = message["content"]
        if role not in RECENT_CONTEXT_ROLES:
            raise ExtractionValidationError(
                "recent_context contains an unsupported role"
            )
        if not isinstance(content, str) or not content.strip():
            raise ExtractionValidationError(
                "recent_context content must be a non-empty string"
            )
        recent_context.append({"role": role, "content": content.strip()})

    return recent_context


class LLMExtractorV2:
    def __init__(self, llm: AIProvider = DeepSeekAI()):
        self.llm: AIProvider = llm
        self.llm_prompt = EXTRACTOR_PROMPT
        self.client = self.llm.get_client()

    # TODO: more role. and add recent info as context
    def extract(self, given_info: str) -> JsonExtractedResult:
        """Extract from a plain user message using current UTC as event time."""
        source = SourceEvent(
            id="",
            content=given_info,
            source_type="message",
            role="user",
            observed_at=datetime.now(timezone.utc).isoformat(),
        )
        return self.extract_with_source(source)

    def extract_with_source(self, source: SourceEvent) -> JsonExtractedResult:
        input_payload = {
            "recent_context": parse_recent_context(source.metadata),
            "source": {
                "content": source.content,
                "source_type": source.source_type,
                "role": source.role,
                "observed_at": source.observed_at,
            }
        }
        result = self.llm.get_ai_response(
            self.client,
            inst=self.llm_prompt,
            input_text=json.dumps(input_payload, ensure_ascii=False),
        )
        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, TypeError) as error:
            raise ExtractionValidationError(
                "Extractor response must be valid JSON"
            ) from error

        # This function will make sure evidence from source
        facts = parse_extracted_facts(
            parsed,
            source_content=source.content,
        )
        return JsonExtractedResult(
            content=result,
            parsed_dict=parsed,
            facts=facts,
        )
