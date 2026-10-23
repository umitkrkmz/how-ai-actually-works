"""json_grammar_fsm.py - Grammar-Constrained JSON Finite State Machine (FSM).

This module implements the mechanics of grammar-guided logit masking:
1. Why Prompted JSON Fails:
   Language models are unconstrained token samplers. Without grammar guidance,
   the model can sample markdown backticks (```json), unescaped quotes, trailing commas,
   or string values where integers are required, breaking parsers.
2. The Grammar-Constrained Solution:
   - A Finite State Machine (FSM) models the exact structural grammar of the target schema.
   - At each decoding step, the FSM determines the legal next tokens.
   - Illegal tokens have their logits masked to -infinity (-inf) before Softmax.
   - exp(-inf) = 0.0, so an illegal next token can never be sampled.
3. What that guarantees, and what it does not (the suite asserts all three):
   - Syntax: every prefix the model emits is a valid prefix of the grammar, with no retries.
     A run cut off by a token budget still ends as an incomplete (unparseable) document.
   - Schema: only what the grammar encodes. Here the grammar is the schema, so types hold.
   - Meaning: never. A schema-valid object can still contain the wrong values.

This is a toy. Its grammar enumerates a few allowed values instead of building sub-automata for
"any string" or "any integer", and its vocabulary is aligned with grammar symbols. Real tokenizers emit
multi-character tokens that do not line up with JSON syntax (e.g. '":' or 'Al' + 'ice"'), so real
engines precompute, for each automaton state, which vocabulary tokens keep the automaton valid. That
index is the hard part of production constrained decoding.

Zero external dependencies. Pure Python standard library.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Set, Tuple


# Target Schema: {"name": string, "age": integer, "active": boolean}
# States in the deterministic generation graph:
STATE_START = 0           # Expects '{'
STATE_KEY_NAME = 1        # Expects '"name"'
STATE_COLON_1 = 2         # Expects ':'
STATE_VAL_NAME = 3        # Expects string literal (e.g. '"Alice"', '"Bob"')
STATE_COMMA_1 = 4         # Expects ','
STATE_KEY_AGE = 5         # Expects '"age"'
STATE_COLON_2 = 6         # Expects ':'
STATE_VAL_AGE = 7         # Expects integer digits (e.g. '25', '30')
STATE_COMMA_2 = 8         # Expects ','
STATE_KEY_ACTIVE = 9      # Expects '"active"'
STATE_COLON_3 = 10        # Expects ':'
STATE_VAL_ACTIVE = 11     # Expects 'true' or 'false'
STATE_END = 12            # Expects '}'
STATE_DONE = 13           # Expects '<EOS>'


class JSONSchemaFSM:
    """Finite State Machine enforcing the schema: {"name": str, "age": int, "active": bool}."""

    def __init__(self) -> None:
        self.transitions: dict[int, dict[str, int]] = {
            STATE_START: {"{": STATE_KEY_NAME},
            STATE_KEY_NAME: {'"name"': STATE_COLON_1},
            STATE_COLON_1: {":": STATE_VAL_NAME},
            STATE_VAL_NAME: {
                '"Alice"': STATE_COMMA_1,
                '"Bob"': STATE_COMMA_1,
                '"Charlie"': STATE_COMMA_1,
            },
            STATE_COMMA_1: {",": STATE_KEY_AGE},
            STATE_KEY_AGE: {'"age"': STATE_COLON_2},
            STATE_COLON_2: {":": STATE_VAL_AGE},
            STATE_VAL_AGE: {
                "25": STATE_COMMA_2,
                "30": STATE_COMMA_2,
                "42": STATE_COMMA_2,
            },
            STATE_COMMA_2: {",": STATE_KEY_ACTIVE},
            STATE_KEY_ACTIVE: {'"active"': STATE_COLON_3},
            STATE_COLON_3: {":": STATE_VAL_ACTIVE},
            STATE_VAL_ACTIVE: {
                "true": STATE_END,
                "false": STATE_END,
            },
            STATE_END: {"}": STATE_DONE},
            STATE_DONE: {},
        }

    def get_allowed_tokens(self, state: int) -> set[str]:
        """Returns the set of structurally legal tokens for the current FSM state."""
        return set(self.transitions.get(state, {}).keys())

    def next_state(self, current_state: int, token: str) -> int:
        """Transitions to the next state upon emitting a token."""
        allowed = self.transitions.get(current_state, {})
        if token not in allowed:
            raise ValueError(f"Illegal transition: Token '{token}' not allowed in state {current_state}")
        return allowed[token]


def mask_logits(
    logits: dict[str, float],
    allowed_tokens: set[str],
) -> dict[str, float]:
    """Applies logit masking: sets illegal token logits to -inf."""
    masked: dict[str, float] = {}
    for token, logit in logits.items():
        if token in allowed_tokens:
            masked[token] = logit
        else:
            masked[token] = -float("inf")
    return masked


def sample_argmax(logits: dict[str, float]) -> str:
    """Picks the token with the highest logit (Greedy decoding)."""
    return max(logits, key=logits.get)


def run_verification_suite() -> None:
    """Verifies what masking guarantees (syntax, schema) and what it does not (completion, meaning)."""

    fsm = JSONSchemaFSM()

    # Representative Vocabulary containing both valid JSON syntax and noisy LLM filler tokens
    vocab = [
        # JSON structural tokens
        "{", "}", ":", ",", '"name"', '"age"', '"active"',
        '"Alice"', '"Bob"', "25", "30", "true", "false",
        # Noisy filler tokens common in raw LLM output
        "```json", "\n", "Sure!", "thirty", "active:", "yes", "<EOS>"
    ]

    # Simulated step-by-step raw model logits (favoring noisy text without constraints)
    step_logits = [
        # Step 0: Model strongly wants to emit conversational intro 'Sure!' or markdown '```json'
        {"Sure!": 12.0, "```json": 10.5, "{": 4.0, "}": 1.0},
        # Step 1: Model wants to emit random text
        {'"name"': 6.0, "here_is_data": 8.0, ",": 2.0},
        # Step 2: Colon
        {":": 7.0, "=": 5.0},
        # Step 3: Value for name
        {'"Alice"': 5.0, '"Bob"': 6.5, "unquoted_alice": 9.0},
        # Step 4: Comma
        {",": 6.0, ";": 4.0},
        # Step 5: Key age
        {'"age"': 7.0, "years_old": 5.0},
        # Step 6: Colon
        {":": 8.0, "-": 3.0},
        # Step 7: Value for age (Model wants to emit English word 'thirty' instead of integer digits)
        {"thirty": 9.5, "25": 4.0, "30": 5.2},
        # Step 8: Comma
        {",": 7.0, "\n": 6.0},
        # Step 9: Key active
        {'"active"': 6.5, "is_enabled": 4.0},
        # Step 10: Colon
        {":": 7.0, "->": 3.0},
        # Step 11: Value active (Model wants to emit 'yes' instead of boolean 'true')
        {"yes": 8.5, "true": 5.0, "false": 4.0},
        # Step 12: End brace
        {"}": 8.0, ",": 4.0},
    ]

    # --- 1. UNCONSTRAINED DECODING (Standard LLM Failure) ---
    unconstrained_tokens = []
    for step_log in step_logits:
        chosen = sample_argmax(step_log)
        unconstrained_tokens.append(chosen)

    unconstrained_text = "".join(unconstrained_tokens)
    # The unconstrained output starts with 'Sure!' and contains 'thirty' and 'yes'
    assert "Sure!" in unconstrained_text
    assert "thirty" in unconstrained_text
    assert "yes" in unconstrained_text

    # Unconstrained text must fail JSON parsing
    unconstrained_failed = False
    try:
        json.loads(unconstrained_text)
    except json.JSONDecodeError:
        unconstrained_failed = True

    assert unconstrained_failed is True, "Unconstrained output should have failed JSON parsing!"

    # --- 2. FSM-CONSTRAINED DECODING ---
    current_state = STATE_START
    constrained_tokens = []

    for step_log in step_logits:
        allowed = fsm.get_allowed_tokens(current_state)
        # Apply logit masking: all non-allowed tokens become -inf
        masked = mask_logits(step_log, allowed)

        chosen = sample_argmax(masked)
        constrained_tokens.append(chosen)

        # Transition FSM to next state
        current_state = fsm.next_state(current_state, chosen)

    constrained_text = "".join(constrained_tokens)
    assert current_state == STATE_DONE, f"FSM should reach STATE_DONE, reached {current_state}"

    # FSM-constrained text MUST be 100% valid JSON matching the schema
    parsed_json = json.loads(constrained_text)
    assert parsed_json["name"] in ["Alice", "Bob"]
    assert isinstance(parsed_json["age"], int)
    assert parsed_json["age"] in [25, 30]
    assert isinstance(parsed_json["active"], bool)

    # --- 3. A token budget can cut the run short: valid prefix, invalid document ---
    state = STATE_START
    for token in constrained_tokens[:8]:
        state = fsm.next_state(state, token)  # every emitted token was legal...
    try:
        json.loads("".join(constrained_tokens[:8]))
    except json.JSONDecodeError:
        pass  # ...but the document is unfinished, so it does not parse
    else:
        raise AssertionError("a truncated document should not parse")

    # --- 4. Schema-valid is not correct ---
    source_record = {"name": "Alice", "age": 42, "active": False}  # what the input text actually said
    assert parsed_json == {"name": "Bob", "age": 30, "active": True}
    assert parsed_json != source_record  # right shape, right types, wrong facts

    print("[OK] JSON Grammar FSM: unconstrained output fails to parse; masked output parses and matches the schema; "
          "truncation and wrong values remain possible.")


if __name__ == "__main__":
    print("Executing JSON Grammar FSM Verification...")
    run_verification_suite()

    # Demonstrative printout
    print("\n--- Output Comparison ---")
    fsm = JSONSchemaFSM()
    step_logits_demo = [
        {"```json": 10.0, "{": 3.0},
        {'"name"': 5.0, "user": 8.0},
        {":": 7.0},
        {'"Bob"': 6.0, "bob": 9.0},
        {",": 6.0},
        {'"age"': 5.0},
        {":": 7.0},
        {"forty_two": 11.0, "42": 4.0},
        {",": 6.0},
        {'"active"': 5.0},
        {":": 7.0},
        {"yep": 10.0, "true": 4.0},
        {"}": 8.0},
    ]

    unconstrained = "".join(sample_argmax(s) for s in step_logits_demo)
    print(f"Unconstrained Raw LLM : {unconstrained}")

    state = STATE_START
    constrained = []
    for s in step_logits_demo:
        allowed = fsm.get_allowed_tokens(state)
        masked = mask_logits(s, allowed)
        t = sample_argmax(masked)
        constrained.append(t)
        state = fsm.next_state(state, t)

    valid_str = "".join(constrained)
    print(f"FSM-Constrained       : {valid_str}")
    print(f"Parsed Python Object  : {json.loads(valid_str)}")
