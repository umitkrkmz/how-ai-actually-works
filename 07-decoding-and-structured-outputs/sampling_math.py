"""sampling_math.py - Next-Token Probability & Sampling Mathematics from Scratch.

This module implements the exact statistical mechanics of LLM text generation:
1. Raw Logits to Probabilities:
   - Temperature Scaling: Divides unnormalized logits by scalar T.
   - Numerically Stable Softmax: Subtracts max(z) to prevent IEEE-754 floating-point overflow.
2. Truncation Algorithms:
   - Top-K Filtering: Hard cutoff preserving only the highest K probability candidates.
   - Top-P (Nucleus) Filtering: Dynamic cutoff preserving the smallest set of tokens
     whose cumulative probability exceeds threshold p.
3. Multinomial Sampling:
   - Cumulative distribution sampling with deterministic seeding.

Zero external dependencies. Pure Python standard library.
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Computes numerically stable softmax with temperature scaling.

    Formula:
      z'_i = z_i / T
      M = max(z')
      P_i = exp(z'_i - M) / sum(exp(z'_j - M))

    Subtracting M prevents exp() overflow without altering the resulting probabilities.
    """
    if temperature <= 0.0:
        raise ValueError("Temperature must be strictly positive (T > 0). For greedy, use T -> 0.")

    # 1. Scale by temperature
    scaled = [z / temperature for z in logits]

    # 2. Subtract maximum value for numerical stability
    max_val = max(scaled)
    exp_shifted = [math.exp(z - max_val) for z in scaled]

    # 3. Normalize to sum to 1.0
    sum_exp = sum(exp_shifted)
    return [e / sum_exp for e in exp_shifted]


def apply_top_k(probabilities: Sequence[float], k: int) -> list[float]:
    """Truncates distribution to top K tokens and renormalizes the remaining probabilities."""
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if k >= len(probabilities):
        return list(probabilities)

    # Sort indices by descending probability
    indexed = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)
    kept_indices = {idx for idx, _ in indexed[:k]}

    filtered = [p if idx in kept_indices else 0.0 for idx, p in enumerate(probabilities)]
    total = sum(filtered)
    return [p / total for p in filtered]


def apply_top_p(probabilities: Sequence[float], p: float) -> list[float]:
    """Applies nucleus (Top-P) sampling by retaining the smallest subset summing to >= p."""
    if not (0.0 < p <= 1.0):
        raise ValueError("p must be in the range (0.0, 1.0].")

    # Sort indices by descending probability
    indexed = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)

    kept_indices = set()
    cumulative = 0.0

    for idx, prob in indexed:
        kept_indices.add(idx)
        cumulative += prob
        if cumulative >= p:
            break

    filtered = [prob if idx in kept_indices else 0.0 for idx, prob in enumerate(probabilities)]
    total = sum(filtered)
    return [prob / total for prob in filtered]


def sample_token(probabilities: Sequence[float], rng: random.Random | None = None) -> int:
    """Draws a discrete token index from a categorical probability distribution."""
    r = (rng or random).random()
    cumulative = 0.0
    for idx, p in enumerate(probabilities):
        cumulative += p
        if r <= cumulative:
            return idx
    return len(probabilities) - 1


def run_verification_suite() -> None:
    """Verifies numerical stability, temperature limits, top-k/top-p truncation, and sampling."""
    eps = 1e-6

    # 1. Numerically Stable Softmax: Handles extreme logits without OverflowError
    extreme_logits = [1000.0, 1005.0, 995.0]  # math.exp(1000) overflows standard float
    probs = softmax(extreme_logits, temperature=1.0)
    assert abs(sum(probs) - 1.0) < eps, f"Probabilities must sum to 1.0, got {sum(probs)}"
    assert probs[1] > probs[0] > probs[2], "Order of probabilities must mirror logits"

    # 2. Temperature Limit Asymptotes:
    # As T -> 0, softmax collapses deterministically to Argmax (Greedy)
    raw_logits = [2.0, 5.0, 1.0]  # Index 1 has the maximum logit
    greedy_probs = softmax(raw_logits, temperature=0.001)
    assert abs(greedy_probs[1] - 1.0) < eps, f"T->0 must collapse to argmax, got {greedy_probs[1]}"
    assert greedy_probs[0] < eps and greedy_probs[2] < eps

    # As T -> inf, softmax approaches Uniform distribution [1/3, 1/3, 1/3]
    uniform_probs = softmax(raw_logits, temperature=1000.0)
    assert abs(uniform_probs[0] - 1 / 3) < 0.01
    assert abs(uniform_probs[1] - 1 / 3) < 0.01
    assert abs(uniform_probs[2] - 1 / 3) < 0.01

    # 3. Top-K Verification
    # Distribution: [0.50, 0.30, 0.15, 0.05]
    p_dist = [0.50, 0.30, 0.15, 0.05]
    top_2 = apply_top_k(p_dist, k=2)
    # Indices 2 and 3 should be zeroed out
    assert top_2[2] == 0.0 and top_2[3] == 0.0
    # Remaining indices (0 and 1) must sum to 1.0 and preserve relative ratio (0.50 / 0.30 = 5/3)
    assert abs(sum(top_2) - 1.0) < eps
    assert abs((top_2[0] / top_2[1]) - (0.50 / 0.30)) < eps

    # 4. Top-P (Nucleus) Verification
    # Distribution: [0.60, 0.25, 0.10, 0.05]
    # For p = 0.70: First token (0.60) < 0.70; second token brings sum to 0.85 (>= 0.70)
    # Result should keep indices 0 and 1, dropping indices 2 and 3
    p_dist2 = [0.60, 0.25, 0.10, 0.05]
    top_p_result = apply_top_p(p_dist2, p=0.70)
    assert top_p_result[2] == 0.0 and top_p_result[3] == 0.0
    assert abs(sum(top_p_result) - 1.0) < eps
    assert abs(top_p_result[0] - (0.60 / 0.85)) < eps

    # 5. Deterministic Sampling Verification
    seeded_rng = random.Random(42)
    sample_counts = {0: 0, 1: 0}
    test_probs = [0.80, 0.20]
    for _ in range(1000):
        t = sample_token(test_probs, seeded_rng)
        sample_counts[t] += 1
    # Empirical ratio should roughly approximate 80/20 within random variance
    assert 750 <= sample_counts[0] <= 850, f"Expected ~800, got {sample_counts[0]}"

    print("[OK] Sampling math: stable softmax, T asymptotes, Top-K, Top-P, and sampling verified.")


if __name__ == "__main__":
    print("Executing Next-Token Sampling Math Verification...")
    run_verification_suite()

    # Demonstrative printout
    logits_demo = [1.2, 4.8, 2.1, 0.3]
    vocab_demo = ["apple", "banana", "cherry", "date"]

    print("\nLogits:", dict(zip(vocab_demo, logits_demo)))
    print("Softmax (T=1.0):", [round(x, 4) for x in softmax(logits_demo, 1.0)])
    print("Softmax (T=0.3):", [round(x, 4) for x in softmax(logits_demo, 0.3)])
    print("Softmax (T=2.0):", [round(x, 4) for x in softmax(logits_demo, 2.0)])

    base_p = softmax(logits_demo, 1.0)
    print("Top-K (k=2)    :", [round(x, 4) for x in apply_top_k(base_p, 2)])
    print("Top-P (p=0.85) :", [round(x, 4) for x in apply_top_p(base_p, 0.85)])
