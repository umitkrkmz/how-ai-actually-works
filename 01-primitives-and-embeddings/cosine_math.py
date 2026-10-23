"""cosine_math.py - Vector Mathematics and Similarity Search from First Principles.

This module demonstrates the bare-metal mechanics of vector embeddings:
1. Dot Product: Sum of coordinate-wise products (projection).
2. Euclidean Norm (Magnitude): Length of a vector in N-dimensional space.
3. Cosine Similarity: Cosine of the angle between two vectors.
4. Euclidean Distance: Straight-line distance between two points in N-dimensional space.
5. Unit Normalization: Projecting any vector onto the unit hypersphere so that
   dot_product(u, v) becomes mathematically identical to cosine_similarity(u, v).
6. Exact Flat Vector Search: How vector databases actually perform naive nearest neighbor queries.

Zero external dependencies. Pure Python standard library.
"""

from __future__ import annotations

import math
from typing import Sequence


def dot_product(u: Sequence[float], v: Sequence[float]) -> float:
    """Compute the algebraic dot product of two vectors: sum(u_i * v_i).

    Geometric intuition: Measures how much two vectors point in the same direction,
    scaled by their respective lengths.
    """
    if len(u) != len(v):
        raise ValueError(f"Vector dimensions must match: len(u)={len(u)}, len(v)={len(v)}")
    return sum(ui * vi for ui, vi in zip(u, v))


def norm(u: Sequence[float]) -> float:
    """Compute the Euclidean L2 norm (magnitude/length) of a vector: sqrt(sum(u_i^2))."""
    sum_sq = sum(ui * ui for ui in u)
    if sum_sq == 0.0:
        return 0.0
    return math.sqrt(sum_sq)


def normalize(u: Sequence[float]) -> list[float]:
    """Normalize a vector to unit length (L2 norm = 1.0).

    A vector lying on the unit hypersphere preserves its angle (direction)
    while standardizing its scale.
    """
    mag = norm(u)
    if mag == 0.0:
        raise ValueError("Cannot normalize a zero vector.")
    return [ui / mag for ui in u]


def cosine_similarity(u: Sequence[float], v: Sequence[float]) -> float:
    """Compute the cosine of the angle between two vectors: (u . v) / (||u|| * ||v||).

    Returns a value in [-1.0, 1.0]:
      1.0 : Vectors point in the exact same direction (angle = 0 deg).
      0.0 : Vectors are orthogonal / perpendicular (angle = 90 deg, zero correlation).
     -1.0 : Vectors point in opposite directions (angle = 180 deg).
    """
    mag_u = norm(u)
    mag_v = norm(v)
    if mag_u == 0.0 or mag_v == 0.0:
        raise ValueError("Cosine similarity is undefined for zero vectors.")
    return dot_product(u, v) / (mag_u * mag_v)


def cosine_distance(u: Sequence[float], v: Sequence[float]) -> float:
    """Compute cosine distance: 1.0 - cosine_similarity(u, v).

    Range: [0.0, 2.0]. 0.0 means identical direction; 2.0 means opposite direction.
    """
    return 1.0 - cosine_similarity(u, v)


def euclidean_distance(u: Sequence[float], v: Sequence[float]) -> float:
    """Compute the Euclidean L2 distance between two points: sqrt(sum((u_i - v_i)^2))."""
    if len(u) != len(v):
        raise ValueError(f"Vector dimensions must match: len(u)={len(u)}, len(v)={len(v)}")
    return math.sqrt(sum((ui - vi) ** 2 for ui, vi in zip(u, v)))


class VectorIndexNaive:
    """A bare-metal, in-memory exact flat vector search index.

    This 25-line class is the conceptual core of what vector databases do during
    'Flat / Exact' index queries (like Chroma or FAISS IndexFlatIP).
    """

    def __init__(self, normalize_vectors: bool = True) -> None:
        self.normalize_vectors = normalize_vectors
        self.entries: list[tuple[str, list[float], dict]] = []

    def insert(self, doc_id: str, vector: Sequence[float], metadata: dict | None = None) -> None:
        """Store a vector. If normalize_vectors=True, vector is stored as a unit vector."""
        vec = normalize(vector) if self.normalize_vectors else list(vector)
        self.entries.append((doc_id, vec, metadata or {}))

    def search(self, query_vector: Sequence[float], top_k: int = 3) -> list[tuple[str, float, dict]]:
        """Perform exact brute-force search across all stored vectors.

        When vectors are normalized, dot_product(q, v) is identical to cosine_similarity.
        Time complexity: O(N * D) where N is collection size and D is vector dimensionality.
        """
        q = normalize(query_vector) if self.normalize_vectors else list(query_vector)
        scored: list[tuple[str, float, dict]] = []

        for doc_id, vec, meta in self.entries:
            # When pre-normalized, cosine similarity reduces to a pure dot product:
            score = dot_product(q, vec) if self.normalize_vectors else cosine_similarity(q, vec)
            scored.append((doc_id, score, meta))

        # Sort descending by similarity score
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


def run_verification_suite() -> None:
    """Execute rigorous mathematical invariants and test assertions."""
    eps = 1e-9

    # 1. Orthogonal Vectors (90 degrees apart -> cos(theta) = 0.0)
    v_x = [1.0, 0.0, 0.0]
    v_y = [0.0, 1.0, 0.0]
    sim_ortho = cosine_similarity(v_x, v_y)
    assert abs(sim_ortho - 0.0) < eps, f"Expected 0.0, got {sim_ortho}"

    # 2. Parallel Vectors with Different Magnitudes (0 degrees apart -> cos(theta) = 1.0)
    v_a = [1.0, 2.0, 3.0]
    v_b = [2.0, 4.0, 6.0]  # Exact scalar multiple of v_a
    sim_parallel = cosine_similarity(v_a, v_b)
    assert abs(sim_parallel - 1.0) < eps, f"Expected 1.0, got {sim_parallel}"

    # 3. Diametrically Opposite Vectors (180 degrees apart -> cos(theta) = -1.0)
    v_c = [-1.0, -2.0, -3.0]
    sim_opposite = cosine_similarity(v_a, v_c)
    assert abs(sim_opposite - (-1.0)) < eps, f"Expected -1.0, got {sim_opposite}"

    # 4. Unit Normalization Equivalence:
    # dot_product(normalize(u), normalize(v)) == cosine_similarity(u, v)
    u_raw = [3.5, -1.2, 8.4, 0.1]
    v_raw = [-0.5, 4.2, 1.1, 9.0]
    u_norm = normalize(u_raw)
    v_norm = normalize(v_raw)

    assert abs(norm(u_norm) - 1.0) < eps, "Normalized vector length must be 1.0"
    assert abs(norm(v_norm) - 1.0) < eps, "Normalized vector length must be 1.0"

    dot_of_normalized = dot_product(u_norm, v_norm)
    direct_cosine = cosine_similarity(u_raw, v_raw)
    assert abs(dot_of_normalized - direct_cosine) < eps, (
        f"dot(u_norm, v_norm) {dot_of_normalized} != cosine(u, v) {direct_cosine}"
    )

    # 5. Geometric Relationship between Euclidean Distance and Cosine Similarity:
    # For unit vectors: ||u - v||^2 = ||u||^2 + ||v||^2 - 2(u . v) = 1 + 1 - 2*cos(theta) = 2 - 2*cos(theta)
    dist_sq = euclidean_distance(u_norm, v_norm) ** 2
    expected_dist_sq = 2.0 - 2.0 * dot_of_normalized
    assert abs(dist_sq - expected_dist_sq) < eps, (
        f"Euclidean distance identity failed: {dist_sq} vs {expected_dist_sq}"
    )

    # 6. Verify Exact Search Engine behavior
    # Note on Cosine Similarity: It strictly measures angular alignment, NOT magnitude.
    # A document with coordinates [0.99, 0.01] is more aligned with [1.0, 0.0] than [0.90, 0.05]
    # because the ratio of off-axis components is smaller.
    index = VectorIndexNaive(normalize_vectors=True)
    # Mock embeddings: 4-dimensional semantic space [Tech, Finance, Biology, Cooking]
    docs = [
        ("doc_kernel", [0.99, 0.01, 0.00, 0.00], {"topic": "OS Kernel Architecture"}),
        ("doc_compiler", [0.88, 0.10, 0.00, 0.01], {"topic": "LLVM Optimization"}),
        ("doc_stocks", [0.05, 0.95, 0.00, 0.02], {"topic": "Equity Portfolio Management"}),
        ("doc_crispr", [0.01, 0.01, 0.98, 0.00], {"topic": "Gene Editing with CRISPR"}),
        ("doc_sourdough", [0.00, 0.02, 0.15, 0.92], {"topic": "Wild Yeast Fermentation"}),
    ]
    for doc_id, vec, meta in docs:
        index.insert(doc_id, vec, meta)

    # Query clearly geared toward systems programming: high 'Tech', zero other topics
    query_tech = [1.00, 0.00, 0.00, 0.00]
    results = index.search(query_tech, top_k=2)

    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert results[0][0] == "doc_kernel", f"Expected top match doc_kernel, got {results[0][0]}"
    assert results[1][0] == "doc_compiler", f"Expected 2nd match doc_compiler, got {results[1][0]}"
    assert results[0][1] > results[1][1], "doc_kernel should score higher than doc_compiler"

    print("[OK] All vector math invariants and retrieval assertions verified successfully.")


if __name__ == "__main__":
    print("Running bare-metal vector mathematics verification...")
    run_verification_suite()

    # Demonstrative printout for manual inspection
    sample_a = [1.0, 2.0, 3.0]
    sample_b = [2.0, 4.0, 5.0]
    print(f"\nVector A: {sample_a}")
    print(f"Vector B: {sample_b}")
    print(f"Norm(A): {norm(sample_a):.4f}")
    print(f"Norm(B): {norm(sample_b):.4f}")
    print(f"Dot Product: {dot_product(sample_a, sample_b):.4f}")
    print(f"Cosine Similarity: {cosine_similarity(sample_a, sample_b):.6f}")
    print(f"Euclidean Distance: {euclidean_distance(sample_a, sample_b):.6f}")
