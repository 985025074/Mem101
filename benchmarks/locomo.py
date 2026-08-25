"""Minimal LoCoMo runner for MemKernel.

The pipeline and JSON output follow the public memory-benchmarks project:
https://github.com/mem0ai/memory-benchmarks
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


DATASET_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)
DEFAULT_DATASET_PATH = Path("datasets/locomo/locomo10.json")
DEFAULT_CATEGORIES = (1, 2, 3, 4)
CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


# This
class TextGenerator(Protocol):
    model: str

    def generate(self, system: str, user: str) -> str: ...


class ProviderTextGenerator:
    """Adapt MemKernel's AI provider to the benchmark's tiny interface."""

    def __init__(self, provider: Any):
        self.provider = provider
        self.client = provider.get_client()
        self.model = str(getattr(provider, "model", type(provider).__name__))

    def generate(self, system: str, user: str) -> str:
        return self.provider.get_ai_response(
            self.client,
            inst=system,
            input_text=user,
        )


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """load dateset to  a dict"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("LoCoMo dataset must be a non-empty JSON list")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError("Every LoCoMo conversation must be a JSON object")
    return data


def ensure_dataset(path: Path) -> list[dict[str, Any]]:
    """Load LoCoMo, downloading its official JSON file when necessary."""
    if path.exists():
        print(f"[LoCoMo] Using dataset: {path}", flush=True)
        return load_dataset(path)

    print(f"[LoCoMo] Downloading dataset to {path}...", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "MemKernel-LoCoMo/0.1"},
    )
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            temporary_path.write_bytes(response.read())
        data = load_dataset(temporary_path)
        temporary_path.replace(path)
        print(f"[LoCoMo] Dataset downloaded: {len(data)} conversations", flush=True)
        return data
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_locomo_date(value: str) -> str | None:
    """Convert dates such as '1:56 pm on 8 May, 2023' to UTC ISO-8601."""
    for date_format in (
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %b, %Y",
    ):
        try:
            parsed = datetime.strptime(value.strip(), date_format)
        except (AttributeError, ValueError):
            continue
        return parsed.replace(tzinfo=timezone.utc).isoformat()
    return None


def get_sorted_sessions(
    conversation: dict[str, Any],
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """Sort sessions. We only accept seesion_number. Input data by time,if no time,then use the number"""
    sessions: list[tuple[str, str, list[dict[str, Any]]]] = []
    for key, turns in conversation.items():
        if not re.fullmatch(r"session_\d+", key) or not isinstance(turns, list):
            continue
        sessions.append((key, str(conversation.get(f"{key}_date_time", "")), turns))

    def sort_key(item: tuple[str, str, list[dict[str, Any]]]) -> tuple[int, Any]:
        parsed = parse_locomo_date(item[1])
        if parsed is not None:
            return (0, parsed)
        return (1, int(item[0].removeprefix("session_")))

    return sorted(sessions, key=sort_key)


def _turn_content(turn: dict[str, Any]) -> str:
    # Currently we cant do image,so we just
    text = str(turn.get("text", "")).strip()
    query = str(turn.get("query", "")).strip()
    caption = str(turn.get("blip_caption", "")).strip()
    if query and caption:
        image_text = f"[Shared image about {query}: {caption}]"
    elif query:
        image_text = f"[Shared image about {query}]"
    elif caption:
        image_text = f"[Shared image: {caption}]"
    else:
        image_text = ""
    return " ".join(part for part in (text, image_text) if part)


def iter_source_chunks(
    entry: dict[str, Any],
    conversation_index: int,
    chunk_size: int,
) -> Iterator[dict[str, Any]]:
    if isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    conversation = entry["conversation"]
    primary_speaker = str(conversation["speaker_a"])
    for session_key, raw_date, turns in get_sorted_sessions(conversation):
        # Keep chunks inside one session so every source has one observed_at.
        observed_at = parse_locomo_date(raw_date)
        prepared_turns: list[dict[str, Any]] = []
        for turn in turns:
            content = _turn_content(turn)
            if not content:
                continue
            speaker = str(turn.get("speaker", "Unknown"))
            dia_id = str(turn["dia_id"]) if turn.get("dia_id") else None
            prefix = f"[{dia_id}] " if dia_id else ""
            line = f"{prefix}{speaker}: {content}"
            prepared_turns.append(
                {
                    "content": line,
                    "dia_id": dia_id,
                    "role": "user"
                    if speaker == primary_speaker
                    else "assistant",
                }
            )

        for start in range(0, len(prepared_turns), chunk_size):
            chunk = prepared_turns[start : start + chunk_size]
            roles = {turn["role"] for turn in chunk}
            dia_ids = [turn["dia_id"] for turn in chunk if turn["dia_id"]]
            yield {
                "content": "\n".join(turn["content"] for turn in chunk),
                "role": next(iter(roles)) if len(roles) == 1 else None,
                "observed_at": observed_at,
                "metadata": {
                    "benchmark": "locomo",
                    "conversation_index": conversation_index,
                    "session": session_key,
                    "chunk_index": start // chunk_size,
                    "turn_count": len(chunk),
                    "dia_id": dia_ids[0] if len(dia_ids) == 1 else None,
                    "dia_ids": dia_ids,
                    "turns": [
                        {
                            "dia_id": turn["dia_id"],
                            "content": turn["content"],
                        }
                        for turn in chunk
                    ],
                },
            }


def build_evidence_lookup(entry: dict[str, Any]) -> dict[str, str]:
    """set up dia_id (DX:Y ) to somthing like \"[D1:3,date] Alice :xxx \""""
    lookup: dict[str, str] = {}
    conversation = entry["conversation"]
    for _, raw_date, turns in get_sorted_sessions(conversation):
        for turn in turns:
            dia_id = turn.get("dia_id")
            if dia_id:
                lookup[str(dia_id)] = (
                    f"[{dia_id}, {raw_date}] {turn.get('speaker', '')}: "
                    f"{_turn_content(turn)}"
                )
    return lookup


def _answer_prompt(question: str, memories: Sequence[str], reference_date: str) -> str:
    memory_block = "\n".join(f"- {memory}" for memory in memories) or "(none)"
    return f"""Answer the question using only the retrieved memories.
Preserve names, dates, quantities, and negations. If the memories are
insufficient, answer that the information is unknown. Return only the answer.

Conversation reference date: {reference_date or "unknown"}
Question: {question}
Memories:
{memory_block}
"""


def _judge_prompt(
    question: str,
    reference_answer: str,
    generated_answer: str,
    evidence: Sequence[str],
) -> str:
    evidence_block = "\n".join(evidence) or "(not available)"
    return f"""Decide whether the generated answer is semantically correct.
Accept equivalent wording and an answer containing at least one correct item
when the reference is a list. Extra correct detail is allowed. Use the source
evidence to accept a better-supported answer when the reference is incomplete.

Question: {question}
Reference answer: {reference_answer}
Generated answer: {generated_answer}
Source evidence:
{evidence_block}

Return JSON only:
{{"verdict":"correct" or "incorrect","reason":"short explanation"}}
"""


def _parse_judgment(raw: str) -> tuple[str, str]:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate)
    start, end = candidate.find("{"), candidate.rfind("}")
    if start >= 0 and end >= start:
        candidate = candidate[start : end + 1]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return "error", "Judge did not return valid JSON"
    verdict = str(payload.get("verdict", "")).casefold()
    if verdict not in {"correct", "incorrect"}:
        return "error", "Judge returned an invalid verdict"
    return verdict, str(payload.get("reason", ""))


def _source_ids(kernel: Any, memory_id: str) -> list[str]:
    sources = kernel.get_sources(memory_id) or []
    source_ids: set[str] = set()
    for linked_source in sources:
        metadata = linked_source.source.metadata
        if metadata.get("dia_id"):
            source_ids.add(str(metadata["dia_id"]))

        evidence = linked_source.evidence_quote
        for turn in metadata.get("turns", []):
            if not isinstance(turn, dict) or not turn.get("dia_id"):
                continue
            turn_content = str(turn.get("content", ""))
            if evidence in turn_content or turn_content in evidence:
                source_ids.add(str(turn["dia_id"]))

    return sorted(source_ids)


def _metric_bucket(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    judgments = [item.get("judgment") for item in items]
    passed = sum(
        1 for judgment in judgments if judgment and judgment["verdict"] == "correct"
    )
    failed = sum(
        1 for judgment in judgments if judgment and judgment["verdict"] == "incorrect"
    )
    errors = sum(
        1 for judgment in judgments if judgment and judgment["verdict"] == "error"
    )
    search_latencies = [
        item["retrieval"]["latency_ms"]
        for item in items
        if item.get("retrieval", {}).get("latency_ms") is not None
    ]
    return {
        "total": len(items),
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "accuracy": round(100 * passed / (passed + failed), 2)
        if passed + failed
        else 0.0,
        "avg_search_latency_ms": round(sum(search_latencies) / len(search_latencies), 2)
        if search_latencies
        else None,
    }


def compute_metrics(evaluations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({str(item["group"]) for item in evaluations})
    evidence_recalls = [
        item["retrieval"]["query_debug"]["evidence_recall"]
        for item in evaluations
        if item["retrieval"].get("query_debug", {}).get("evidence_recall") is not None
    ]
    return {
        "overall": _metric_bucket(evaluations),
        "by_group": {
            group: _metric_bucket(
                [item for item in evaluations if item["group"] == group]
            )
            for group in groups
        },
        "retrieval": {
            "questions_with_evidence": len(evidence_recalls),
            "evidence_hit_rate": round(
                sum(value > 0 for value in evidence_recalls) / len(evidence_recalls),
                4,
            )
            if evidence_recalls
            else None,
            "mean_evidence_recall": round(
                sum(evidence_recalls) / len(evidence_recalls), 4
            )
            if evidence_recalls
            else None,
        },
    }


def create_runtime(
    embedding_model: str,
) -> tuple[Callable[[Path], Any], TextGenerator]:
    """Construct the real MemKernel factory only when the CLI starts a run."""
    from memkernel.ai import DeepSeekAI
    from memkernel.backend.backend_v2 import BackendV2
    from memkernel.embedding import OpenAIEmbeddingProvider
    from memkernel.extractor.extractor_v2 import LLMExtractorV2
    from memkernel.kernel import MemKernel

    ai_provider = DeepSeekAI()
    embedding_provider = OpenAIEmbeddingProvider(
        OpenAIEmbeddingProvider.get_client(),
        model=embedding_model,
    )

    def kernel_factory(database_path: Path) -> MemKernel:
        backend = BackendV2(
            memory_path=database_path,
            ai_provider=ai_provider,
            embedding_provider=embedding_provider,
        )
        return MemKernel(
            extractor=LLMExtractorV2(ai_provider),
            memory_backend=backend,
        )

    return kernel_factory, ProviderTextGenerator(ai_provider)


def run_benchmark(
    dataset: Sequence[dict[str, Any]],
    *,
    conversation_indices: Sequence[int],
    max_questions: int | None,
    chunk_size: int,
    top_k: int,
    threshold: float,
    predict_only: bool,
    kernel_factory: Callable[[Path], Any],
    generator: TextGenerator,
) -> dict[str, Any]:
    from memkernel.kernel import PostMemory

    evaluations: list[dict[str, Any]] = []
    total_ingestion_ms = 0.0
    total_turns = 0
    total_chunks = 0

    # We set up tempdir
    with tempfile.TemporaryDirectory(prefix="memkernel-locomo-") as temp_dir:
        root = Path(temp_dir)

        for conversation_position, conversation_index in enumerate(
            conversation_indices,
            start=1,
        ):
            # Each conversation's dict
            entry = dataset[conversation_index]
            # for each conversation set up a independent database
            kernel = kernel_factory(root / f"conversation-{conversation_index}.db")
            source_chunks = list(
                iter_source_chunks(
                    entry,
                    conversation_index,
                    chunk_size,
                )
            )
            conversation_turn_count = sum(
                int(source["metadata"]["turn_count"])
                for source in source_chunks
            )
            print(
                f"[LoCoMo] Conversation {conversation_position}/"
                f"{len(conversation_indices)} (index {conversation_index}): "
                f"ingesting {conversation_turn_count} turns in "
                f"{len(source_chunks)} chunks",
                flush=True,
            )

            ingestion_start = time.monotonic()
            for chunk_position, source in enumerate(source_chunks, start=1):
                print(
                    f"\r[LoCoMo] Conversation {conversation_index}: "
                    f"chunk {chunk_position}/{len(source_chunks)}",
                    end="",
                    flush=True,
                )
                kernel.remember(
                    PostMemory(
                        date=source["observed_at"],
                        content=source["content"],
                        source_type="message",
                        role=source["role"],
                        metadata=source["metadata"],
                    )
                )
            print(flush=True)
            # record time we use
            ingestion_ms = (time.monotonic() - ingestion_start) * 1000
            total_ingestion_ms += ingestion_ms
            total_turns += conversation_turn_count
            total_chunks += len(source_chunks)

            questions = entry.get("qa", entry.get("qa_pairs", []))
            # By default we dont test 5
            # select specified category 's problem
            selected_questions = [
                (index, qa)
                for index, qa in enumerate(questions)
                if int(qa.get("category", 0)) in DEFAULT_CATEGORIES
            ]
            # select questions
            if max_questions is not None:
                selected_questions = selected_questions[:max_questions]

            conversation = entry["conversation"]
            sessions = get_sorted_sessions(conversation)
            reference_date = sessions[-1][1] if sessions else ""
            evidence_lookup = build_evidence_lookup(entry)
            memory_count = len(kernel.list_memories())
            print(
                f"[LoCoMo] Conversation {conversation_index}: ingestion complete "
                f"({memory_count} memories, {ingestion_ms / 1000:.1f}s)",
                flush=True,
            )

            # Solbe problem(recall ) Here
            for question_position, (question_index, qa) in enumerate(
                selected_questions,
                start=1,
            ):
                question = str(qa["question"])
                print(
                    f"[LoCoMo] Conversation {conversation_index}: question "
                    f"{question_position}/{len(selected_questions)} retrieving",
                    flush=True,
                )
                search_start = time.monotonic()
                recalled = kernel.recall(
                    question,
                    current_top_k=top_k,
                    history_top_k=0,
                    threshold=threshold,
                )
                search_ms = (time.monotonic() - search_start) * 1000

                retrieved: list[dict[str, Any]] = []
                found_source_ids: set[str] = set()
                for rank, result in enumerate(recalled.current, start=1):
                    source_ids = _source_ids(kernel, result.memory.id)
                    # found evidence id
                    found_source_ids.update(source_ids)
                    retrieved.append(
                        {
                            "rank": rank,
                            "memory": result.memory.content,
                            "score": result.score,
                            "id": result.memory.id,
                            "created_at": result.memory.created_at,
                            "source_ids": source_ids,
                        }
                    )

                # real evidence id
                expected_source_ids = {str(item) for item in qa.get("evidence", [])}
                # recall percentage
                evidence_recall = (
                    len(found_source_ids & expected_source_ids)
                    / len(expected_source_ids)
                    if expected_source_ids
                    else None
                )
                category = int(qa["category"])
                # evaluation for each question
                evaluation: dict[str, Any] = {
                    "id": f"conv{conversation_index}_q{question_index}",
                    "group": CATEGORY_NAMES.get(category, "unknown"),
                    "question": question,
                    "ground_truth": str(qa["answer"]),
                    "conversation_idx": conversation_index,
                    "ingestion": {
                        "items_processed": len(source_chunks),
                        "items_failed": 0,
                        "total_memories_created": memory_count,
                        "latency_ms": round(ingestion_ms, 2),
                    },
                    "retrieval": {
                        "query": question,
                        "latency_ms": round(search_ms, 2),
                        "results": retrieved,
                        "total_results": len(retrieved),
                        "query_debug": {
                            "expected_source_ids": sorted(expected_source_ids),
                            "evidence_recall": evidence_recall,
                        },
                    },
                }

                # use LLM's judge
                if not predict_only:
                    print(
                        f"[LoCoMo] Conversation {conversation_index}: question "
                        f"{question_position}/{len(selected_questions)} answering",
                        flush=True,
                    )
                    answer_start = time.monotonic()
                    generated_answer = generator.generate(
                        "You answer questions from retrieved conversational memory.",
                        _answer_prompt(
                            question,
                            [item["memory"] for item in retrieved],
                            reference_date,
                        ),
                    ).strip()
                    answer_ms = (time.monotonic() - answer_start) * 1000

                    # right answer
                    reference_answer = str(qa["answer"])

                    # We just care about first part
                    if category == 3 and ";" in reference_answer:
                        reference_answer = reference_answer.split(";", 1)[0].strip()
                    # Judge our answer with real source
                    evidence = [
                        evidence_lookup[source_id]
                        for source_id in expected_source_ids
                        if source_id in evidence_lookup
                    ]
                    print(
                        f"[LoCoMo] Conversation {conversation_index}: question "
                        f"{question_position}/{len(selected_questions)} judging",
                        flush=True,
                    )
                    judge_start = time.monotonic()
                    raw_judgment = generator.generate(
                        "You evaluate conversational memory answers. Return JSON only.",
                        _judge_prompt(
                            question,
                            reference_answer,
                            generated_answer,
                            evidence,
                        ),
                    )
                    judge_ms = (time.monotonic() - judge_start) * 1000
                    verdict, reason = _parse_judgment(raw_judgment)
                    evaluation["generation"] = {
                        "model": generator.model,
                        "answer": generated_answer,
                        "latency_ms": round(answer_ms, 2),
                    }
                    evaluation["judgment"] = {
                        "model": generator.model,
                        "verdict": verdict,
                        "score": 1.0 if verdict == "correct" else 0.0,
                        "reasoning": reason,
                        "latency_ms": round(judge_ms, 2),
                    }
                    print(
                        f"[LoCoMo] Conversation {conversation_index}: question "
                        f"{question_position}/{len(selected_questions)} -> {verdict}",
                        flush=True,
                    )

                evaluations.append(evaluation)

    return {
        "schema_version": "1.0",
        "metadata": {
            "eval_type": "locomo",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "models": {
                "answerer": None if predict_only else generator.model,
                "judge": None if predict_only else generator.model,
            },
            "capabilities": {
                "has_answer_sessions": not predict_only,
                "has_ingestion_debug": True,
                "has_ground_truth_evidence": True,
            },
            "dataset": {
                "name": "locomo10",
                "total_items": len(evaluations),
                "categories": [CATEGORY_NAMES[item] for item in DEFAULT_CATEGORIES],
            },
            "config": {
                "conversation_indices": list(conversation_indices),
                "max_questions_per_conversation": max_questions,
                "chunk_size": chunk_size,
                "top_k": top_k,
                "threshold": threshold,
                "predict_only": predict_only,
                "ingested_turns": total_turns,
                "ingested_chunks": total_chunks,
                "ingestion_latency_ms": round(total_ingestion_ms, 2),
            },
        },
        "metrics": compute_metrics(evaluations),
        "evaluations": evaluations,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small LoCoMo evaluation")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--conversation", type=int, action="append")
    parser.add_argument("--all-conversations", action="store_true")
    parser.add_argument("--max-questions", type=int, default=5)
    parser.add_argument("--all-questions", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--embedding-model",
        default="nomic-embed-text:latest",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = ensure_dataset(args.dataset)
    # Args check
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be between 0 and 1")
    if args.max_questions <= 0:
        raise SystemExit("--max-questions must be positive")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")

    #  the conv we run
    if args.all_conversations:
        conversation_indices = list(range(len(dataset)))
    else:
        conversation_indices = sorted(set(args.conversation or [0]))
    if any(index < 0 or index >= len(dataset) for index in conversation_indices):
        raise SystemExit(f"Conversation index must be between 0 and {len(dataset) - 1}")

    kernel_factory, generator = create_runtime(args.embedding_model)
    result = run_benchmark(
        dataset,
        conversation_indices=conversation_indices,
        max_questions=None if args.all_questions else args.max_questions,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        threshold=args.threshold,
        predict_only=args.predict_only,
        kernel_factory=kernel_factory,
        generator=generator,
    )

    output = args.output or Path("benchmark-results") / (
        f"locomo-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    overall = result["metrics"]["overall"]
    if args.predict_only:
        print(f"LoCoMo retrieval completed for {overall['total']} questions")
    else:
        print(
            f"LoCoMo: {overall['passed']}/{overall['passed'] + overall['failed']} "
            f"correct ({overall['accuracy']:.2f}%)"
        )
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
