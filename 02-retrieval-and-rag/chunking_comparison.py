"""chunking_comparison.py - Structural vs. Naive Fixed-Size Chunking.

This module demonstrates the structural failure modes of naive chunking:
1. Fixed-Size Character Chunking: Slices text at arbitrary character boundaries,
   severing identifiers, words, and semantic clauses in half.
2. Sliding Window Chunking: Mitigates word-severing with overlap, but duplicates tokens
   and still separates subjects from predicates.
3. Delimiter / Structural Chunking: Splits along natural semantic boundaries
   (double newlines, AST boundaries, Markdown headers, or function signatures).

Zero external dependencies. Pure Python standard library.
"""

from __future__ import annotations

import re


def naive_fixed_chunk(text: str, chunk_size: int) -> list[str]:
    """Slices text purely by character count without regard for words or boundaries."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i : i + chunk_size])
    return chunks


def sliding_window_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Slices text by characters with a sliding overlap window."""
    chunks = []
    start = 0
    stride = max(1, chunk_size - overlap)
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        if start + chunk_size >= len(text):
            break
        start += stride
    return chunks


def structural_delimiter_chunk(text: str, delimiter_pattern: str = r"\n\s*\n") -> list[str]:
    """Splits text on natural structural boundaries (e.g. paragraphs, function defs)."""
    raw_blocks = re.split(delimiter_pattern, text)
    return [block.strip() for block in raw_blocks if block.strip()]


def score_term_density(query: str, chunk: str) -> float:
    """Lexical overlap score: query-term occurrences divided by sqrt(chunk length).

    Not BM25: no inverse document frequency and no term-frequency saturation. Just enough to rank chunks.
    """
    query_terms = set(re.findall(r"\b[a-z0-9_]+\b", query.lower()))
    chunk_terms = re.findall(r"\b[a-z0-9_]+\b", chunk.lower())
    if not chunk_terms or not query_terms:
        return 0.0

    matches = sum(1 for term in chunk_terms if term in query_terms)
    # Simple term density score
    return matches / (len(chunk_terms) ** 0.5)


def run_verification_suite() -> None:
    """Proves that naive fixed chunking severs critical security clauses, while structural chunking succeeds."""

    # Realistic infrastructure policy document
    security_policy = (
        "[POLICY_FIREWALL_902]\n"
        "All incoming ingress traffic on public interface eth0 is blocked by default.\n"
        "EXCEPTION: Automated deployment runners connecting via certificate-based mTLS "
        "on port 9443 are granted temporary ingress.\n"
        "All other packets are dropped immediately with an ICMP unreachable reply."
    )

    query = "What is the EXCEPTION rule for POLICY_FIREWALL_902?"

    # --- 1. Naive Fixed-Size Chunking (chunk_size = 95 characters) ---
    fixed_chunks = naive_fixed_chunk(security_policy, chunk_size=95)

    # Prove that the word 'EXCEPTION' or its rule was severed right through
    # Let's see: character 95 lands right in the middle of the text!
    assert len(fixed_chunks) >= 3, f"Expected at least 3 chunks, got {len(fixed_chunks)}"

    # Check Chunk 0: Has POLICY_FIREWALL_902, but does it have the complete EXCEPTION clause?
    has_policy_in_chunk0 = "POLICY_FIREWALL_902" in fixed_chunks[0]
    has_exception_in_chunk0 = "EXCEPTION" in fixed_chunks[0]
    has_port_in_chunk0 = "9443" in fixed_chunks[0]

    # In naive chunking, Chunk 0 contains the policy ID, but cuts off BEFORE port 9443 and the mTLS rule!
    assert has_policy_in_chunk0 is True
    assert has_port_in_chunk0 is False, (
        "Chunk 0 should have severed the port 9443 rule from the policy header"
    )

    # In Chunk 1: It might contain 'port 9443', but DOES NOT contain 'POLICY_FIREWALL_902'!
    has_policy_in_chunk1 = "POLICY_FIREWALL_902" in fixed_chunks[1]
    has_port_in_chunk1 = "9443" in fixed_chunks[1]
    assert has_policy_in_chunk1 is False, "Chunk 1 lost the policy subject header"
    assert has_port_in_chunk1 is True, "Chunk 1 contains the rule detail without the header"

    # Now simulate retrieval: neither chunk contains BOTH the policy name AND the complete answer!
    scores_fixed = [score_term_density(query, c) for c in fixed_chunks]
    best_fixed_idx = scores_fixed.index(max(scores_fixed))
    best_fixed_chunk = fixed_chunks[best_fixed_idx]

    # Neither fixed chunk can satisfy the prompt alone without truncation
    assert not ("POLICY_FIREWALL_902" in best_fixed_chunk and "port 9443" in best_fixed_chunk), (
        "Naive fixed chunking unexpectedly kept both subject and predicate together"
    )

    # --- 2. Structural Delimiter-Aware Chunking ---
    # Treats the entire coherent policy block as a unified chunk
    structural_chunks = structural_delimiter_chunk(security_policy, delimiter_pattern=r"\n\s*\n")
    assert len(structural_chunks) == 1, "Policy is a single coherent block"

    best_structural_chunk = structural_chunks[0]
    # Structural chunk preserves both the identifier and the exception rule intact!
    assert "POLICY_FIREWALL_902" in best_structural_chunk
    assert "port 9443" in best_structural_chunk
    assert "EXCEPTION:" in best_structural_chunk

    print("[OK] Chunking comparison verified: Naive chunking severed context; structural chunking preserved integrity.")


if __name__ == "__main__":
    print("Running Chunking Strategy Invariant Checks...")
    run_verification_suite()

    # Demonstrative printout of the severed chunks
    sample_doc = (
        "def authenticate_admin(user_id: str, token: str) -> bool:\n"
        "    # CRITICAL: Root bypass token must be disabled in staging/prod\n"
        "    if token == 'SUPERUSER_SECRET_OVERRIDE':\n"
        "        return True\n"
        "    return verify_jwt_signature(token)\n"
    )

    print("\n--- Strategy 1: Naive Fixed-Size Chunking (chunk_size=70 chars) ---")
    fixed = naive_fixed_chunk(sample_doc, chunk_size=70)
    for i, c in enumerate(fixed):
        print(f"Chunk {i} ({len(c)} chars):\n{repr(c)}")

    print("\n--- Strategy 2: Structural Chunking (by function definition) ---")
    structural = structural_delimiter_chunk(sample_doc, delimiter_pattern=r"(?=\ndef )")
    for i, c in enumerate(structural):
        print(f"Chunk {i} ({len(c)} chars):\n{c}")
