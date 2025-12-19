"""Benchmark suites for the DML playground."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class BenchmarkQuery(TypedDict, total=False):
    id: str
    question: str
    answer: Optional[str]
    ground_truth_ids: Optional[List[str]]
    meta: Dict[str, Any]


class BenchmarkDocument(TypedDict, total=False):
    id: str
    text: str
    meta: Dict[str, Any]


class BenchmarkSuite(TypedDict, total=False):
    id: str
    name: str
    description: str
    k: int
    queries: List[BenchmarkQuery]
    documents: List[BenchmarkDocument]


def _build_doc(doc_id: str, text: str, *, meta: Optional[Dict[str, Any]] = None) -> BenchmarkDocument:
    return {"id": doc_id, "text": text, "meta": meta or {}}


def _needle_doc(idx: int, fact: str) -> BenchmarkDocument:
    padding = " ".join(["lorem ipsum"] * 30)
    needle_sentence = f"Needle fact {idx}: {fact}."
    return _build_doc(
        f"needle-{idx}",
        f"{padding} {needle_sentence} {padding}",
        meta={"needle_fact": fact},
    )


doc_qa_suite: BenchmarkSuite = {
    "id": "doc_qa_mini",
    "name": "Doc QA mini",
    "description": "Small QA set over short product and policy notes.",
    "k": 3,
    "documents": [
        _build_doc(
            "roadmap",
            "Upcoming release: v2.1 ships on October 20 with Slack integration and improved analytics dashboard.",
            meta={"topic": "product"},
        ),
        _build_doc(
            "support-hours",
            "Customer support operates Monday to Friday, 8am to 6pm Pacific time, excluding federal holidays.",
            meta={"topic": "support"},
        ),
        _build_doc(
            "security",
            "Security posture: SOC2 Type II certified, data encrypted at rest with AES-256, backups retained for 30 days.",
            meta={"topic": "security"},
        ),
        _build_doc(
            "pricing",
            "Pricing: Starter at $29/month, Growth at $99/month, and Enterprise with custom quotes including priority support.",
            meta={"topic": "pricing"},
        ),
        _build_doc(
            "sla",
            "Service level agreement guarantees 99.9% uptime with credits available after 15 minutes of consecutive downtime.",
            meta={"topic": "sla"},
        ),
    ],
    "queries": [
        {
            "id": "roadmap-date",
            "question": "When is v2.1 scheduled to ship and what key feature is included?",
            "answer": "v2.1 ships on October 20 and includes Slack integration.",
            "ground_truth_ids": ["roadmap"],
        },
        {
            "id": "support-hours",
            "question": "What are the weekday support hours?",
            "answer": "Support runs Monday to Friday, 8am to 6pm Pacific time.",
            "ground_truth_ids": ["support-hours"],
        },
        {
            "id": "sla-credits",
            "question": "When do uptime credits start applying?",
            "answer": "Credits are available after 15 minutes of consecutive downtime under the 99.9% uptime SLA.",
            "ground_truth_ids": ["sla"],
        },
        {
            "id": "security-encryption",
            "question": "How is data stored at rest?",
            "answer": "Data is encrypted at rest with AES-256 as part of the SOC2 Type II posture.",
            "ground_truth_ids": ["security"],
        },
        {
            "id": "pricing-tiers",
            "question": "List the pricing tiers.",
            "answer": "Starter at $29/month, Growth at $99/month, Enterprise is custom with priority support.",
            "ground_truth_ids": ["pricing"],
        },
    ],
}


needle_suite: BenchmarkSuite = {
    "id": "needle_toy",
    "name": "Needle-in-a-Haystack (toy)",
    "description": "Synthetic long docs with embedded needle facts.",
    "k": 2,
    "documents": [
        _needle_doc(1, "The admin password is riven-42"),
        _needle_doc(2, "Launch site is Pad 39A"),
        _needle_doc(3, "Primary contact is Dr. Imani Lee"),
        _needle_doc(4, "Backup database endpoint is db.internal:9002"),
        _needle_doc(5, "API key prefix is sk_live_NEEDLE"),
    ],
    "queries": [
        {
            "id": "needle-password",
            "question": "What is the admin password?",
            "answer": "The admin password is riven-42.",
            "ground_truth_ids": ["needle-1"],
            "meta": {"needle_fact": "riven-42"},
        },
        {
            "id": "needle-pad",
            "question": "Which launch site is used?",
            "answer": "The launch site is Pad 39A.",
            "ground_truth_ids": ["needle-2"],
            "meta": {"needle_fact": "Pad 39A"},
        },
        {
            "id": "needle-contact",
            "question": "Name the primary contact person.",
            "answer": "The primary contact is Dr. Imani Lee.",
            "ground_truth_ids": ["needle-3"],
            "meta": {"needle_fact": "Imani Lee"},
        },
        {
            "id": "needle-endpoint",
            "question": "What is the backup database endpoint?",
            "answer": "The backup database endpoint is db.internal:9002.",
            "ground_truth_ids": ["needle-4"],
            "meta": {"needle_fact": "db.internal:9002"},
        },
        {
            "id": "needle-key",
            "question": "Provide the API key prefix noted in the docs.",
            "answer": "The API key prefix is sk_live_NEEDLE.",
            "ground_truth_ids": ["needle-5"],
            "meta": {"needle_fact": "sk_live_NEEDLE"},
        },
    ],
}


BENCHMARK_SUITES: List[BenchmarkSuite] = [doc_qa_suite, needle_suite]
