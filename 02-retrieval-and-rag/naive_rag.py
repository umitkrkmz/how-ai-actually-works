"""naive_rag.py - Bare-Metal End-to-End RAG from Scratch.

This module demonstrates the complete mechanics of Naive Retrieval-Augmented Generation (RAG):
1. Ingestion & Chunking: Split documents into digestible text snippets.
2. Embedding Generation: Compute normalized TF-IDF vectors (a lexical stand-in for a learned embedding).
3. Flat Exact Retrieval: Compute cosine similarity between query and all stored chunks.
4. Prompt Augmentation: Synthesize the final LLM prompt injecting the top-k retrieved chunks.
5. Deterministic Generation: A regex stands in for the LLM, showing that the answer can only come from the prompt.

Zero external dependencies. Pure Python standard library.
"""

from __future__ import annotations

import math
import re
from typing import Sequence


def tokenize(text: str) -> list[str]:
    """Lowercases and extracts alphanumeric words."""
    return re.findall(r"\b[a-z0-9_]+\b", text.lower())


def compute_tf(tokens: list[str]) -> dict[str, float]:
    """Computes normalized term frequency for a tokenized text."""
    if not tokens:
        return {}
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = len(tokens)
    return {t: count / total for t, count in counts.items()}


class BareMetalVectorStore:
    """A pure Python vector store implementing exact flat search over normalized vectors."""

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.chunks: list[dict] = []
        self.vectors: list[list[float]] = []

    def build_vocabulary(self, raw_chunks: list[str]) -> None:
        """Builds a global vocabulary and IDF table across all corpus chunks."""
        doc_count = len(raw_chunks)
        doc_frequencies: dict[str, int] = {}

        for chunk_text in raw_chunks:
            unique_terms = set(tokenize(chunk_text))
            for term in unique_terms:
                doc_frequencies[term] = doc_frequencies.get(term, 0) + 1

        # Sorted for deterministic index ordering
        all_terms = sorted(doc_frequencies.keys())
        self.vocabulary = {term: idx for idx, term in enumerate(all_terms)}

        # Standard smoothed IDF: ln((N + 1) / (df + 1)) + 1
        self.idf = {
            term: math.log((doc_count + 1) / (df + 1)) + 1.0
            for term, df in doc_frequencies.items()
        }

    def _embed(self, text: str) -> list[float]:
        """Converts text into a unit-normalized TF-IDF vector."""
        tokens = tokenize(text)
        tf = compute_tf(tokens)
        vec = [0.0] * len(self.vocabulary)

        for term, term_tf in tf.items():
            if term in self.vocabulary:
                idx = self.vocabulary[term]
                vec[idx] = term_tf * self.idf[term]

        # Normalize to unit length (L2 norm) so dot_product == cosine_similarity
        magnitude = math.sqrt(sum(x * x for x in vec))
        if magnitude > 0.0:
            vec = [x / magnitude for x in vec]
        return vec

    def add_chunks(self, chunks: list[dict]) -> None:
        """Adds structured chunk dictionaries and computes their embeddings."""
        self.chunks.extend(chunks)
        for chunk in chunks:
            self.vectors.append(self._embed(chunk["text"]))

    def search(self, query: str, top_k: int = 2) -> list[tuple[dict, float]]:
        """Performs exact flat cosine search (query dot product against all chunks)."""
        q_vec = self._embed(query)
        scored: list[tuple[dict, float]] = []

        for chunk, v_chunk in zip(self.chunks, self.vectors):
            # Because vectors are pre-normalized, dot product is identical to cosine similarity:
            sim = sum(qi * vi for qi, vi in zip(q_vec, v_chunk))
            scored.append((chunk, sim))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


def chunk_document(doc_id: str, text: str, chunk_size_words: int = 25, overlap_words: int = 5) -> list[dict]:
    """Splits a document into overlapping word chunks to prevent cutting words in half."""
    words = text.split()
    chunks = []
    start = 0
    chunk_index = 0
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunk_text = " ".join(words[start:end]).strip()
        if chunk_text:
            chunks.append({
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "text": chunk_text,
            })
            chunk_index += 1
        if end == len(words):
            break
        start += max(1, chunk_size_words - overlap_words)
    return chunks


def synthesize_rag_prompt(query: str, retrieved_chunks: list[tuple[dict, float]]) -> str:
    """Assembles the final augmented prompt injecting retrieved context."""
    context_blocks = []
    for rank, (chunk, score) in enumerate(retrieved_chunks, 1):
        context_blocks.append(
            f"[Source: {chunk['doc_id']} | Chunk {chunk['chunk_index']} | Score: {score:.3f}]\n{chunk['text']}"
        )
    formatted_context = "\n\n".join(context_blocks)

    prompt = (
        "You are an exact technical assistant. Answer the question using ONLY the provided context snippets.\n"
        "If the answer is not contained within the context, say 'I do not have enough information.'\n\n"
        f"--- BEGIN CONTEXT ---\n{formatted_context}\n--- END CONTEXT ---\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
    return prompt


def simulate_llm_generation(prompt: str) -> str:
    """A deterministic stand-in for the LLM generation step.

    Demonstrates how generation is strictly bounded by the text present in the prompt.
    """
    if "max_connections" in prompt.lower() and "database" in prompt.lower():
        match = re.search(r"max_connections\s*=\s*([0-9]+)", prompt)
        if match:
            return f"The maximum database connection limit is {match.group(1)}."
    elif "port" in prompt.lower() and "redis" in prompt.lower():
        match = re.search(r"redis.*?port\s*([0-9]+)", prompt, re.DOTALL | re.IGNORECASE)
        if match:
            return f"Redis is configured on port {match.group(1)}."
    return "I do not have enough information."


def run_verification_suite() -> None:
    """Verifies naive RAG pipeline execution, indexing invariants, and retrieval accuracy."""
    raw_docs = [
        (
            "doc_database_config",
            "Database Connection Pool Settings: By default, the application opens 10 idle connections. "
            "Under peak load, max_connections = 250 to prevent postgres process exhaustion. "
            "Connection timeout is set to 3000ms before returning an HTTP 504 gateway timeout.",
        ),
        (
            "doc_cache_config",
            "Redis Cluster Infrastructure: In-memory caching runs on dedicated internal nodes. "
            "Redis is bound to internal port 6379 with TLS enabled. "
            "Keys expire automatically after 86400 seconds unless flagged as persistent.",
        ),
        (
            "doc_auth_guide",
            "Authentication Service Guidelines: All incoming user requests must present an OAuth 2.0 Bearer token. "
            "Tokens are validated against the internal JWKS endpoint at /auth/.well-known/jwks.json.",
        ),
    ]

    # 1. Chunk documents
    all_chunks = []
    for doc_id, text in raw_docs:
        all_chunks.extend(chunk_document(doc_id, text, chunk_size_words=20, overlap_words=4))

    assert len(all_chunks) >= 4, f"Expected at least 4 chunks across 3 docs, got {len(all_chunks)}"

    # 2. Build index & vector store
    store = BareMetalVectorStore()
    store.build_vocabulary([c["text"] for c in all_chunks])
    store.add_chunks(all_chunks)

    assert len(store.vocabulary) > 20, "Vocabulary size should exceed 20 terms"
    assert len(store.vectors) == len(all_chunks), "Vector count must equal chunk count"

    # Verify unit-length invariant on vectors
    for v in store.vectors:
        mag = math.sqrt(sum(x * x for x in v))
        assert abs(mag - 1.0) < 1e-9, f"Vector must be unit normalized, got magnitude {mag}"

    # 3. Query retrieval test
    query = "What is the max_connections limit under peak load for database?"
    retrieved = store.search(query, top_k=2)

    assert len(retrieved) == 2
    top_chunk, top_score = retrieved[0]
    assert top_chunk["doc_id"] == "doc_database_config", (
        f"Expected top chunk from doc_database_config, got {top_chunk['doc_id']}"
    )
    assert "max_connections" in top_chunk["text"], "Top chunk must contain target fact"
    assert top_score > 0.3, f"Similarity score should be substantial, got {top_score}"

    # 4. Prompt synthesis test
    prompt = synthesize_rag_prompt(query, retrieved)
    assert "--- BEGIN CONTEXT ---" in prompt
    assert "max_connections = 250" in prompt
    assert f"Question: {query}" in prompt

    # 5. Answer extraction test
    answer = simulate_llm_generation(prompt)
    assert answer == "The maximum database connection limit is 250.", (
        f"Unexpected generated answer: {answer}"
    )

    print("[OK] Naive RAG pipeline: chunking, embedding, retrieval, prompt assembly, and extraction passed.")


if __name__ == "__main__":
    print("Executing Bare-Metal Naive RAG Verification...")
    run_verification_suite()

    # Manual walkthrough output
    sample_query = "What port is Redis running on?"
    store = BareMetalVectorStore()
    docs = [
        ("doc_db", "Postgres runs on port 5432 with 10 pool connections."),
        ("doc_cache", "Redis caching service listens on internal port 6379."),
        ("doc_proxy", "Nginx reverse proxy listens on public port 443 with SSL termination."),
    ]
    chunks = []
    for d_id, text in docs:
        chunks.extend(chunk_document(d_id, text, chunk_size_words=15, overlap_words=3))

    store.build_vocabulary([c["text"] for c in chunks])
    store.add_chunks(chunks)
    matches = store.search(sample_query, top_k=1)

    print(f"\nUser Query: '{sample_query}'")
    print(f"Top Retrieved Document: {matches[0][0]['doc_id']}")
    print(f"Similarity Score: {matches[0][1]:.4f}")
    print(f"Retrieved Chunk: \"{matches[0][0]['text']}\"")
    prompt = synthesize_rag_prompt(sample_query, matches)
    print("\nSynthesized LLM Prompt:\n" + "=" * 40)
    print(prompt)
    print("=" * 40)
    print("Simulated answer (regex, no LLM):", simulate_llm_generation(prompt))
