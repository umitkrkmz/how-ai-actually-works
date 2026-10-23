"""rag_evaluator.py - Ground-Truth Retrieval & Faithfulness Metrics from Scratch.

This module implements empirical evaluation mathematics without third-party frameworks:
1. Precision@K (P@K): Fraction of retrieved top-K documents that are relevant.
2. Recall@K (R@K): Fraction of all relevant documents that were retrieved in top-K.
3. Reciprocal Rank (RR) & Mean Reciprocal Rank (MRR): Position of the first relevant document.
4. Hit Rate@K: Fraction of queries with at least one relevant document in top-K.
5. Context Faithfulness: Quantifying whether claims/tokens in the generated answer
   are strictly grounded in the retrieved context (hallucination detection).

Zero external dependencies. Pure Python standard library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set


@dataclass(frozen=True)
class QueryEvalSample:
    query_id: str
    query_text: str
    expected_relevant_ids: set[str]
    retrieved_ids: list[str]
    retrieved_context: str
    generated_answer: str


def precision_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Computes Precision@K: (Number of relevant docs in top-K) / K."""
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    top_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant)
    return relevant_retrieved / k


def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Computes Recall@K: (Number of relevant docs in top-K) / (Total relevant docs)."""
    if not relevant:
        return 1.0
    top_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant)
    return relevant_retrieved / len(relevant)


def reciprocal_rank_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Computes Reciprocal Rank (RR): 1 / rank of first relevant item in top-K.

    Returns 0.0 if no relevant document appears in top-K.
    """
    top_k = retrieved[:k]
    for rank, doc_id in enumerate(top_k, 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Computes Hit Rate@K: 1.0 if at least one relevant document is in top-K, else 0.0."""
    top_k = retrieved[:k]
    return 1.0 if any(doc_id in relevant for doc_id in top_k) else 0.0


def compute_retrieval_batch_metrics(
    samples: Sequence[QueryEvalSample], k: int
) -> dict[str, float]:
    """Computes aggregate macro-averaged metrics across an evaluation batch."""
    if not samples:
        return {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "hit_rate": 0.0}

    total_p = 0.0
    total_r = 0.0
    total_rr = 0.0
    total_hits = 0.0
    n = len(samples)

    for s in samples:
        total_p += precision_at_k(s.retrieved_ids, s.expected_relevant_ids, k)
        total_r += recall_at_k(s.retrieved_ids, s.expected_relevant_ids, k)
        total_rr += reciprocal_rank_at_k(s.retrieved_ids, s.expected_relevant_ids, k)
        total_hits += hit_rate_at_k(s.retrieved_ids, s.expected_relevant_ids, k)

    return {
        f"precision@{k}": total_p / n,
        f"recall@{k}": total_r / n,
        f"mrr@{k}": total_rr / n,
        f"hit_rate@{k}": total_hits / n,
    }


def compute_context_faithfulness(
    generated_answer: str,
    retrieved_context: str,
    stop_words: Set[str] | None = None,
) -> float:
    """Computes the fraction of substantive words in generated answer grounded in context.

    Faithfulness = |Tokens(Answer) INTERSECT Tokens(Context)| / |Tokens(Answer)|.
    A score of 1.0 means every substantive answer word also occurs in the context. It is a cheap proxy:
    - False alarms: a harmless word that is not in the context lowers the score.
    - Blind spot: an answer that recombines context words into a false claim (swapped numbers,
      a negation, the wrong entity) still scores 1.0.
    Checking that each claim is *entailed* by the context needs an entailment model or a judge.
    """
    if stop_words is None:
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "and", "or", "it", "this", "that", "by", "with",
        }

    answer_tokens = [
        t for t in re.findall(r"\b[a-z0-9_]+\b", generated_answer.lower())
        if t not in stop_words
    ]
    if not answer_tokens:
        return 1.0

    context_token_set = set(re.findall(r"\b[a-z0-9_]+\b", retrieved_context.lower()))
    grounded_count = sum(1 for token in answer_tokens if token in context_token_set)
    return grounded_count / len(answer_tokens)


def run_verification_suite() -> None:
    """Rigorous mathematical verification of retrieval metrics and hallucination scoring."""
    eps = 1e-9

    # 1. Single Query Metric Verification (Hand-calculated)
    # Target relevant: {"doc_a", "doc_b"}
    # Retrieved: ["doc_x", "doc_a", "doc_y", "doc_b"]
    retrieved = ["doc_x", "doc_a", "doc_y", "doc_b"]
    relevant = {"doc_a", "doc_b"}

    # k=1: ["doc_x"] -> 0 relevant
    assert abs(precision_at_k(retrieved, relevant, k=1) - 0.0) < eps
    assert abs(recall_at_k(retrieved, relevant, k=1) - 0.0) < eps
    assert abs(reciprocal_rank_at_k(retrieved, relevant, k=1) - 0.0) < eps
    assert abs(hit_rate_at_k(retrieved, relevant, k=1) - 0.0) < eps

    # k=2: ["doc_x", "doc_a"] -> 1 relevant ("doc_a" at rank 2)
    # P@2 = 1/2 = 0.5
    # R@2 = 1/2 = 0.5
    # RR = 1/2 = 0.5
    # HitRate = 1.0
    assert abs(precision_at_k(retrieved, relevant, k=2) - 0.5) < eps
    assert abs(recall_at_k(retrieved, relevant, k=2) - 0.5) < eps
    assert abs(reciprocal_rank_at_k(retrieved, relevant, k=2) - 0.5) < eps
    assert abs(hit_rate_at_k(retrieved, relevant, k=2) - 1.0) < eps

    # k=4: ["doc_x", "doc_a", "doc_y", "doc_b"] -> 2 relevant ("doc_a", "doc_b")
    # P@4 = 2/4 = 0.5
    # R@4 = 2/2 = 1.0
    # RR = 1/2 = 0.5 (first hit is still at rank 2)
    assert abs(precision_at_k(retrieved, relevant, k=4) - 0.5) < eps
    assert abs(recall_at_k(retrieved, relevant, k=4) - 1.0) < eps
    assert abs(reciprocal_rank_at_k(retrieved, relevant, k=4) - 0.5) < eps

    # 2. Batch MRR Verification
    # Query 1: First hit at rank 1 -> RR = 1.0
    # Query 2: First hit at rank 2 -> RR = 0.5
    # Query 3: No hit in top 3     -> RR = 0.0
    # MRR = (1.0 + 0.5 + 0.0) / 3 = 1.5 / 3 = 0.5
    batch = [
        QueryEvalSample("q1", "q1", {"d1"}, ["d1", "d2"], "ctx", "ans"),
        QueryEvalSample("q2", "q2", {"d2"}, ["d3", "d2"], "ctx", "ans"),
        QueryEvalSample("q3", "q3", {"d9"}, ["d4", "d5"], "ctx", "ans"),
    ]
    batch_metrics = compute_retrieval_batch_metrics(batch, k=2)
    assert abs(batch_metrics["mrr@2"] - 0.5) < eps
    assert abs(batch_metrics["hit_rate@2"] - (2 / 3)) < eps

    # 3. Context Faithfulness & Hallucination Verification
    context = (
        "Postgres database configuration: max_connections = 250. "
        "Idle connection timeout is 3000 milliseconds."
    )
    # Grounded answer: every factual token is in context
    grounded_answer = "Postgres max_connections is 250 and idle connection timeout is 3000 milliseconds."
    faith_grounded = compute_context_faithfulness(grounded_answer, context)
    assert faith_grounded == 1.0, f"Expected 1.0 for grounded answer, got {faith_grounded}"

    # A false alarm: 'limit' is a harmless paraphrase, but it is not in the context, so the score drops
    partially_grounded_answer = "The max_connections limit is 250 and idle timeout is 3000 milliseconds."
    faith_partial = compute_context_faithfulness(partially_grounded_answer, context)
    # 6 out of 7 tokens match ('limit' was not in context): 6/7 = 0.8571
    assert abs(faith_partial - (6 / 7)) < eps

    # The blind spot: swapped numbers make a false answer from context words only, and it scores 1.0
    swapped_answer = "Postgres max_connections is 3000 and idle connection timeout is 250 milliseconds."
    assert compute_context_faithfulness(swapped_answer, context) == 1.0

    # Hallucinated answer: injects completely invented facts (e.g., redis, 6379, encryption)
    hallucinated_answer = (
        "Postgres max_connections is 250, but Redis cache runs on port 6379 with AES encryption."
    )
    faith_hallucinated = compute_context_faithfulness(hallucinated_answer, context)
    assert faith_hallucinated < 0.60, (
        f"Expected faithfulness < 0.60 for hallucinated terms, got {faith_hallucinated:.2f}"
    )

    print("[OK] RAG evaluation metrics: P@K, R@K, MRR, HitRate, and lexical faithfulness (with its blind spot) verified.")


if __name__ == "__main__":
    print("Executing Ground-Truth RAG Evaluator Verification...")
    run_verification_suite()

    # Demonstrative printout
    print("\n--- Example Batch Evaluation Output (k=3) ---")
    mock_batch = [
        QueryEvalSample(
            query_id="q_auth",
            query_text="What port is auth running on?",
            expected_relevant_ids={"doc_auth"},
            retrieved_ids=["doc_auth", "doc_cache", "doc_db"],
            retrieved_context="Auth service listens on port 8080.",
            generated_answer="Auth service runs on port 8080.",
        ),
        QueryEvalSample(
            query_id="q_db",
            query_text="What is max_connections?",
            expected_relevant_ids={"doc_db"},
            retrieved_ids=["doc_proxy", "doc_db", "doc_auth"],
            retrieved_context="Database max_connections is set to 250.",
            generated_answer="Database max_connections is 250.",
        ),
    ]
    results = compute_retrieval_batch_metrics(mock_batch, k=3)
    for m, val in results.items():
        print(f"  {m:15s}: {val:.4f}")
