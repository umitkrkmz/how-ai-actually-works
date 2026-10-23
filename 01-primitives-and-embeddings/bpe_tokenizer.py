"""bpe_tokenizer.py - Byte Pair Encoding (BPE) Tokenizer from Scratch.

This module demonstrates the bare-metal mechanics of modern LLM tokenizers:
1. The Dilemma: Word-level tokenization produces massive vocabularies and breaks
   on Out-Of-Vocabulary (OOV) words. Character-level tokenization inflates sequence
   length by 4-5x (exploding quadratic self-attention costs: O(L^2)).
2. The Solution: Subword tokenization via Byte Pair Encoding (BPE).
3. The Mechanism:
   - Deconstruct raw text into individual characters (or bytes) with an end-of-word token.
   - Count frequencies of all adjacent pairs.
   - Greedily merge the single most frequent pair into a new token.
   - Repeat for N merge operations to build a subword vocabulary.
   - Map tokens to unique integer IDs.

Zero external dependencies. Pure Python standard library.
"""

from __future__ import annotations

import collections
import re
from typing import Dict, List, Tuple


class BytePairEncoder:
    """A minimal, self-contained Byte Pair Encoding (BPE) tokenizer."""

    def __init__(self) -> None:
        self.merges: list[tuple[str, str]] = []
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.end_of_word = "</w>"

    def _get_stats(self, splits: dict[tuple[str, ...], int]) -> dict[tuple[str, str], int]:
        """Count frequency of all adjacent token pairs across the current vocabulary splits."""
        pairs: dict[tuple[str, str], int] = collections.defaultdict(int)
        for word_tuple, freq in splits.items():
            for i in range(len(word_tuple) - 1):
                pairs[(word_tuple[i], word_tuple[i + 1])] += freq
        return pairs

    def _merge_pair(
        self, pair: tuple[str, str], splits: dict[tuple[str, ...], int]
    ) -> dict[tuple[str, ...], int]:
        """Replace all occurrences of `pair` with a single merged token across splits."""
        first, second = pair
        new_splits: dict[tuple[str, ...], int] = {}

        for word_tuple, freq in splits.items():
            new_tuple: list[str] = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == first and word_tuple[i + 1] == second:
                    new_tuple.append(first + second)
                    i += 2
                else:
                    new_tuple.append(word_tuple[i])
                    i += 1
            new_splits[tuple(new_tuple)] = freq
        return new_splits

    def train(self, corpus: str, num_merges: int = 10) -> None:
        """Train BPE on raw text by discovering the top `num_merges` subword pairs."""
        # 1. Clean and split corpus into whitespace-delimited words
        words = re.findall(r"\S+", corpus)
        word_counts = collections.Counter(words)

        # 2. Decompose words into character tuples appended with end-of-word marker
        splits: dict[tuple[str, ...], int] = {
            tuple(list(word) + [self.end_of_word]): count for word, count in word_counts.items()
        }

        # 3. Seed base vocabulary with all unique single characters + end marker
        base_tokens = set()
        for word_tuple in splits:
            for token in word_tuple:
                base_tokens.add(token)

        self.merges.clear()

        # 4. Iteratively find most frequent pair and merge
        for _ in range(num_merges):
            stats = self._get_stats(splits)
            if not stats:
                break
            best_pair = max(stats, key=stats.get)
            splits = self._merge_pair(best_pair, splits)
            self.merges.append(best_pair)

        # 5. Build final vocabulary (base tokens + all merged combinations)
        all_vocab: set[str] = set(base_tokens)
        for pair in self.merges:
            all_vocab.add(pair[0] + pair[1])

        # Sort for deterministic integer assignments
        sorted_vocab = sorted(list(all_vocab))
        self.token_to_id = {token: idx for idx, token in enumerate(sorted_vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def tokenize_word(self, word: str) -> list[str]:
        """Tokenize a single word by iteratively applying learned merge rules."""
        tokens: list[str] = list(word) + [self.end_of_word]

        for pair in self.merges:
            first, second = pair
            new_tokens: list[str] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == first and tokens[i + 1] == second:
                    new_tokens.append(first + second)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str) -> list[int]:
        """Convert a raw text string into a list of integer token IDs."""
        words = re.findall(r"\S+", text)
        token_ids: list[int] = []

        for word in words:
            word_tokens = self.tokenize_word(word)
            for t in word_tokens:
                if t in self.token_to_id:
                    token_ids.append(self.token_to_id[t])
                    continue
                # Fall back to single characters; refuse characters the tokenizer has never seen.
                # (Byte-level BPE avoids this case by starting from all 256 byte values.)
                for char in t:
                    if char not in self.token_to_id:
                        raise ValueError(f"character {char!r} is not in the vocabulary")
                    token_ids.append(self.token_to_id[char])
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Reconstruct human-readable text from a list of integer token IDs."""
        tokens = [self.id_to_token[tid] for tid in token_ids if tid in self.id_to_token]
        # Rejoin tokens and replace end-of-word marker with space
        joined = "".join(tokens)
        text = joined.replace(self.end_of_word, " ").strip()
        return text


def run_verification_suite() -> None:
    """Validate BPE learning, merge order, token mapping, and round-trip decoding."""
    # Classical linguistic dataset: frequent subword stems and suffixes
    sample_corpus = (
        "low low low low low "
        "lower lower "
        "newest newest newest newest newest newest "
        "widest widest widest"
    )

    tokenizer = BytePairEncoder()
    tokenizer.train(corpus=sample_corpus, num_merges=6)

    # 1. Verify merges were learned in descending frequency
    assert len(tokenizer.merges) == 6, f"Expected 6 merges, got {len(tokenizer.merges)}"

    # 'es', 'st' and 't</w>' each occur 9 times (newest x 6 + widest x 3); ties go to the pair counted first.
    # Then 'lo' and 'low' (7 times: low x 5 + lower x 2). This is the worked example in the README.
    assert tokenizer.merges[:5] == [("e", "s"), ("es", "t"), ("est", "</w>"), ("l", "o"), ("lo", "w")], (
        f"Unexpected merge order: {tokenizer.merges}"
    )

    # 2. Tokenize known word
    tokens_newest = tokenizer.tokenize_word("newest")
    # 'newest' should be compressed into fewer tokens than raw characters (len('newest') + 1 = 7)
    assert len(tokens_newest) < 7, f"Expected subword compression, got {tokens_newest}"

    # 3. Round-trip encoding and decoding
    test_phrase = "low newest widest"
    encoded_ids = tokenizer.encode(test_phrase)
    decoded_phrase = tokenizer.decode(encoded_ids)

    assert decoded_phrase == test_phrase, (
        f"Round-trip failed: original '{test_phrase}' != decoded '{decoded_phrase}'"
    )

    # 4. Token IDs are non-negative integers mapped 1-to-1
    for tid in encoded_ids:
        assert isinstance(tid, int) and tid >= 0
        assert tid in tokenizer.id_to_token

    # 5. A character the tokenizer has never seen is refused, not silently dropped
    try:
        tokenizer.encode("lowz")
    except ValueError:
        pass
    else:
        raise AssertionError("an unseen character must raise ValueError")

    print("[OK] All BPE tokenization invariants and round-trip assertions verified successfully.")


if __name__ == "__main__":
    print("Running bare-metal BPE tokenizer verification...")
    run_verification_suite()

    # Demonstrative printout
    demo_corpus = "the cat in the hat sat on the mat with another cat"
    bpe = BytePairEncoder()
    bpe.train(corpus=demo_corpus, num_merges=5)

    print("\nLearned Merge Rules (Ordered):")
    for step, (a, b) in enumerate(bpe.merges, 1):
        print(f"  Merge {step}: '{a}' + '{b}' -> '{a + b}'")

    sample = "the cat sat"
    encoded = bpe.encode(sample)
    print(f"\nOriginal text: '{sample}'")
    print(f"Encoded token IDs: {encoded}")
    print(f"Tokens: {[bpe.id_to_token[tid] for tid in encoded]}")
    print(f"Decoded text: '{bpe.decode(encoded)}'")
