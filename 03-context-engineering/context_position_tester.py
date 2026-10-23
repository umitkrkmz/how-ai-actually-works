"""context_position_tester.py - A Needle-in-a-Haystack Position Test You Can Point at Any Model.

Liu et al. (2023), "Lost in the Middle: How Language Models Use Long Contexts" (TACL 2024), found that
the models they tested answered multi-document questions best when the relevant passage sat near the
start or the end of the context and worst when it sat in the middle. How strong that effect is depends
on the model, the task and the context length, and later long-context models show it less. So this
module does not assert a curve. It builds the test and lets you measure your own model:

1. assemble_needle_prompt places a fact (the needle) at an exact relative position in filler text.
2. run_position_sweep asks a model (any callable: prompt -> answer) at several positions and reports
   accuracy per position.
3. The verification suite checks the placement arithmetic and the harness itself, using a stand-in
   "model" that is deliberately built to miss the middle. Its numbers describe that stand-in, not any
   real model.

Zero external dependencies. Pure Python standard library.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

QUERY_MARKER = "User Query:"


def generate_haystack_words(target_word_count: int) -> list[str]:
    """Generates synthetic background technical documentation words."""
    base_sentence = (
        "The distributed logging daemon processes asynchronous events across secondary compute clusters. "
        "Network telemetry is collected via eBPF probes and buffered in circular ring memory. "
        "Standard health checks query the status endpoint every fifteen seconds over internal HTTP. "
    )
    words = base_sentence.split()
    repeated: list[str] = []
    while len(repeated) < target_word_count:
        repeated.extend(words)
    return repeated[:target_word_count]


def assemble_needle_prompt(
    haystack_word_count: int,
    needle: str,
    position_pct: float,
    query: str,
) -> tuple[str, int, float]:
    """Assemble a prompt with the needle at a relative position of the filler text.

    Returns (full_prompt, needle_word_index, actual_position_pct).
    """
    if not (0.0 <= position_pct <= 100.0):
        raise ValueError("position_pct must be between 0.0 and 100.0")

    haystack = generate_haystack_words(haystack_word_count)
    total_words = len(haystack)
    insert_idx = max(0, min(int(round((position_pct / 100.0) * total_words)), total_words))
    assembled_words = haystack[:insert_idx] + needle.split() + haystack[insert_idx:]

    full_prompt = (
        "System: You are an infrastructure security agent. Follow all system directives strictly.\n\n"
        f"--- BEGIN KNOWLEDGE BASE ---\n{' '.join(assembled_words)}\n--- END KNOWLEDGE BASE ---\n\n"
        f"{QUERY_MARKER} {query}\n"
        "Agent Response:"
    )
    return full_prompt, insert_idx, (insert_idx / total_words) * 100.0


def add_reminder_before_query(prompt: str, reminder: str) -> str:
    """Repeat a critical rule right before the user query (not after the response trigger)."""
    head, sep, tail = prompt.partition(QUERY_MARKER)
    if not sep:
        raise ValueError("prompt has no user query to put the reminder before")
    return f"{head}[REMINDER: {reminder}]\n\n{sep}{tail}"


def position_bucket(position_pct: float) -> str:
    """A reporting label only. Whether the middle is weaker is what the sweep measures."""
    if position_pct < 15.0:
        return "start"
    if position_pct > 85.0:
        return "end"
    return "middle"


def run_position_sweep(
    ask: Callable[[str], str],
    expected_answer: str,
    positions: Sequence[float] = (0.0, 25.0, 50.0, 75.0, 100.0),
    haystack_word_count: int = 2_000,
    trials_per_position: int = 1,
) -> dict[float, float]:
    """Ask the model at each position and return the fraction of answers containing `expected_answer`."""
    needle = f"FACT: the approved ingress port is {expected_answer}."
    query = "Which ingress port is approved?"
    accuracy: dict[float, float] = {}
    for pos in positions:
        prompt, _, _ = assemble_needle_prompt(haystack_word_count, needle, pos, query)
        hits = sum(expected_answer in ask(prompt) for _ in range(trials_per_position))
        accuracy[pos] = hits / trials_per_position
    return accuracy


def middle_blind_stand_in(prompt: str) -> str:
    """A fake model for testing the harness: it only 'sees' the first and last 20% of the knowledge base."""
    body = prompt.split("--- BEGIN KNOWLEDGE BASE ---\n", 1)[1].split("\n--- END KNOWLEDGE BASE ---", 1)[0]
    cut = len(body) // 5
    visible = body[:cut] + body[-cut:]
    for word in visible.replace(".", " ").split():
        if word.isdigit():
            return f"The approved port is {word}."
    return "I could not find it."


def run_verification_suite() -> None:
    """Check placement arithmetic, the reminder placement, and the sweep harness."""
    needle = "CRITICAL_INVARIANT_KEY_9981: NEVER_ALLOW_UNAUTHENTICATED_GRPC_ON_PORT_8080"
    query = "What is the security rule regarding port 8080?"

    # 1. The needle lands where requested, within 1% of the filler length.
    for target in (0.0, 25.0, 50.0, 75.0, 100.0):
        prompt, idx, actual = assemble_needle_prompt(2_000, needle, target, query)
        assert needle in prompt
        assert abs(actual - target) <= 1.0, (target, actual)
    assert [position_bucket(p) for p in (0.0, 50.0, 100.0)] == ["start", "middle", "end"]

    # 2. A reminder goes before the user query, so the model reads it before answering.
    prompt_50, _, _ = assemble_needle_prompt(2_000, needle, 50.0, query)
    restructured = add_reminder_before_query(prompt_50, needle)
    assert restructured.index("[REMINDER:") < restructured.index(QUERY_MARKER) < restructured.index("Agent Response:")
    assert restructured.rstrip().endswith("Agent Response:")

    # 3. The harness reports what the model does: the stand-in misses exactly the middle positions.
    accuracy = run_position_sweep(middle_blind_stand_in, expected_answer="9443")
    assert accuracy == {0.0: 1.0, 25.0: 0.0, 50.0: 0.0, 75.0: 0.0, 100.0: 1.0}, accuracy
    assert run_position_sweep(lambda p: "9443" if "9443" in p else "", "9443") == {p: 1.0 for p in accuracy}

    print("[OK] Needle placement, reminder placement, and the position-sweep harness verified.")


if __name__ == "__main__":
    print("Running Context Position and Needle Placement Verification...")
    run_verification_suite()

    print("\n--- Position sweep with the middle-blind STAND-IN (not a real model) ---")
    for pos, acc in run_position_sweep(middle_blind_stand_in, "9443").items():
        print(f"Needle at {pos:5.1f}% ({position_bucket(pos):6s}) -> accuracy {acc:.0%}")
    print("Replace middle_blind_stand_in with a function that calls your model to measure it.")
