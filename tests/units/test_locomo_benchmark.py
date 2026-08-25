from types import SimpleNamespace

from benchmarks.locomo import (
    _source_ids,
    build_evidence_lookup,
    compute_metrics,
    get_sorted_sessions,
    iter_source_chunks,
    parse_locomo_date,
)


def sample_entry() -> dict:
    return {
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_2": [
                {
                    "speaker": "Bob",
                    "text": "Welcome back.",
                    "dia_id": "D2:1",
                }
            ],
            "session_2_date_time": "2:00 pm on 9 May, 2023",
            "session_1": [
                {
                    "speaker": "Alice",
                    "text": "I moved to Paris.",
                    "dia_id": "D1:1",
                }
            ],
            "session_1_date_time": "1:56 pm on 8 May, 2023",
        },
        "qa": [],
    }


def test_locomo_sessions_and_turns_are_chronological() -> None:
    entry = sample_entry()

    sessions = get_sorted_sessions(entry["conversation"])
    chunks = list(iter_source_chunks(entry, conversation_index=3, chunk_size=8))

    assert [session[0] for session in sessions] == ["session_1", "session_2"]
    assert chunks[0]["content"] == "[D1:1] Alice: I moved to Paris."
    assert chunks[0]["role"] == "user"
    assert chunks[1]["role"] == "assistant"
    assert chunks[0]["metadata"]["dia_id"] == "D1:1"


def test_locomo_batches_turns_without_crossing_sessions() -> None:
    entry = sample_entry()
    entry["conversation"]["session_1"].append(
        {
            "speaker": "Bob",
            "text": "Congratulations!",
            "dia_id": "D1:2",
        }
    )

    chunks = list(iter_source_chunks(entry, conversation_index=0, chunk_size=8))

    assert len(chunks) == 2
    assert chunks[0]["role"] is None
    assert chunks[0]["metadata"]["dia_ids"] == ["D1:1", "D1:2"]
    assert chunks[0]["metadata"]["turn_count"] == 2
    assert chunks[1]["metadata"]["session"] == "session_2"


def test_chunk_source_ids_are_resolved_from_exact_evidence() -> None:
    linked_source = SimpleNamespace(
        evidence_quote="I moved to Paris.",
        source=SimpleNamespace(
            metadata={
                "turns": [
                    {"dia_id": "D1:1", "content": "[D1:1] Alice: I moved to Paris."},
                    {"dia_id": "D1:2", "content": "[D1:2] Bob: Congratulations!"},
                ]
            }
        ),
    )
    kernel = SimpleNamespace(get_sources=lambda _: [linked_source])

    assert _source_ids(kernel, "memory-id") == ["D1:1"]


def test_locomo_date_and_evidence_parsing() -> None:
    entry = sample_entry()

    assert parse_locomo_date("1:56 pm on 8 May, 2023") == (
        "2023-05-08T13:56:00+00:00"
    )
    assert build_evidence_lookup(entry)["D1:1"].endswith(
        "Alice: I moved to Paris."
    )


def test_locomo_metrics_include_quality_and_retrieval() -> None:
    evaluations = [
        {
            "group": "single-hop",
            "retrieval": {
                "latency_ms": 10.0,
                "query_debug": {"evidence_recall": 0.5},
            },
            "judgment": {"verdict": "correct"},
        },
        {
            "group": "single-hop",
            "retrieval": {
                "latency_ms": 20.0,
                "query_debug": {"evidence_recall": 0.0},
            },
            "judgment": {"verdict": "incorrect"},
        },
    ]

    metrics = compute_metrics(evaluations)

    assert metrics["overall"]["accuracy"] == 50.0
    assert metrics["overall"]["avg_search_latency_ms"] == 15.0
    assert metrics["retrieval"]["evidence_hit_rate"] == 0.5
    assert metrics["retrieval"]["mean_evidence_recall"] == 0.25


def test_locomo_predict_only_metrics_do_not_require_judgments() -> None:
    metrics = compute_metrics(
        [
            {
                "group": "single-hop",
                "retrieval": {
                    "latency_ms": 10.0,
                    "query_debug": {"evidence_recall": None},
                },
            }
        ]
    )

    assert metrics["overall"]["total"] == 1
    assert metrics["overall"]["accuracy"] == 0.0
    assert metrics["retrieval"]["evidence_hit_rate"] is None
