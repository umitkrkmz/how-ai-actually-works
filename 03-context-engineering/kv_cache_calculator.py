"""kv_cache_calculator.py - Multi-Turn Token and Cost Simulator.

This module models the economics of context engineering:
1. Quadratic token accumulation: without caching, re-sending the conversation history makes the
   cumulative input tokens grow as O(T^2) in the number of turns T.
2. Three caching policies:
   - none:        every call pays the full input price for the whole prompt.
   - static:      only a fixed prefix (system prompt) is cached; the history is paid in full every call.
   - incremental: everything already sent is read from cache and only new tokens are written. This is
                  how agent tools commonly use provider prompt caching.
3. Tool reads are context too: every file an agent reads arrives as a tool result, joins the history,
   and costs an extra model call that re-sends the context. "Two-tier" context (a small static
   manifesto plus on-demand reads) is cheap only while the reads stay few.

Prices are an example (a 1.25x cache write and a 0.10x cache read, as some providers publish).
Check your provider's current pricing and caching rules before relying on any number here.

Zero external dependencies. Pure Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PricingModel:
    """Pricing rates per 1,000,000 tokens (USD)."""
    name: str
    base_input_per_m: float      # e.g., $3.00 / 1M tokens
    cache_write_per_m: float     # e.g., $3.75 / 1M tokens (1.25x base)
    cache_read_per_m: float      # e.g., $0.30 / 1M tokens (0.10x base)
    output_per_m: float          # e.g., $15.00 / 1M tokens


# An example price sheet with the multipliers above; not a quote for any specific model.
STANDARD_PRICING = PricingModel(
    name="Example-LLM-Tier",
    base_input_per_m=3.00,
    cache_write_per_m=3.75,
    cache_read_per_m=0.30,
    output_per_m=15.00,
)

POLICIES = ("none", "static", "incremental")


@dataclass
class SimulationResult:
    calls: int
    total_input_tokens: int
    total_output_tokens: int
    cached_read_tokens: int
    cache_write_tokens: int
    uncached_input_tokens: int
    total_cost_usd: float


def simulate_session(
    prefix_tokens: int,
    turns: int,
    policy: str = "none",
    user_tokens_per_turn: int = 150,
    assistant_tokens_per_turn: int = 350,
    reads_per_turn: int = 0,
    tokens_per_read: int = 2_000,
    tool_call_tokens: int = 60,
    pricing: PricingModel = STANDARD_PRICING,
) -> SimulationResult:
    """Simulate a chat or agent session and bill it under one caching policy.

    Each user turn makes `reads_per_turn` tool calls (each a short model output of `tool_call_tokens`
    followed by a tool result of `tokens_per_read`), then one final answer of `assistant_tokens_per_turn`.
    Every model call sends the full prompt: prefix + history + whatever is new since the last call.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    calls = total_input = total_output = cached_reads = cache_writes = uncached = 0
    history = 0        # tokens of conversation after the prefix
    cached = 0         # tokens currently in cache (incremental policy)

    for turn in range(turns):
        pending = user_tokens_per_turn  # new input since the previous call
        for call in range(reads_per_turn + 1):
            prompt = prefix_tokens + history + pending
            total_input += prompt
            calls += 1
            if policy == "none":
                uncached += prompt
            elif policy == "static":
                if turn == 0 and call == 0:
                    cache_writes += prefix_tokens
                else:
                    cached_reads += prefix_tokens
                uncached += history + pending
            else:  # incremental
                cached_reads += cached
                cache_writes += prompt - cached
                cached = prompt
            is_tool_call = call < reads_per_turn
            output = tool_call_tokens if is_tool_call else assistant_tokens_per_turn
            total_output += output
            history += pending + output
            pending = tokens_per_read if is_tool_call else 0

    cost = (
        cache_writes * pricing.cache_write_per_m
        + cached_reads * pricing.cache_read_per_m
        + uncached * pricing.base_input_per_m
        + total_output * pricing.output_per_m
    ) / 1_000_000
    return SimulationResult(calls, total_input, total_output, cached_reads, cache_writes, uncached, cost)


def calculate_cache_breakeven_turns(prefix_tokens: int, pricing: PricingModel = STANDARD_PRICING) -> int:
    """The turn at which caching a static prefix becomes cheaper than not caching it.

    Turn 1 pays the write surcharge prefix * (write - base); every later turn saves prefix * (base - read).
    """
    extra_write_cost = prefix_tokens * (pricing.cache_write_per_m - pricing.base_input_per_m)
    per_turn_savings = prefix_tokens * (pricing.base_input_per_m - pricing.cache_read_per_m)
    if per_turn_savings <= 0:
        return 999_999
    turns_discounted_needed = extra_write_cost / per_turn_savings
    return 1 + int(turns_discounted_needed + 0.999)


def run_verification_suite() -> None:
    """Verify the closed form, the policies' ordering, the breakeven turn, and the two-tier crossover."""
    S, U, A, T = 25_000, 150, 350, 10

    # 1. The uncached total matches the closed form T*S + T*U + T(T-1)/2 * (U + A).
    plain = simulate_session(S, T, "none", U, A)
    assert plain.total_input_tokens == T * S + T * U + T * (T - 1) // 2 * (U + A)

    # 2. Quadratic growth of the history alone: twice the turns, more than three times the tokens.
    five = simulate_session(0, 5, "none", 100, 100)
    ten = simulate_session(0, 10, "none", 100, 100)
    assert ten.total_input_tokens / five.total_input_tokens > 3.0

    # 3. Caching a static prefix pays for its write surcharge on turn 2 with these prices.
    assert calculate_cache_breakeven_turns(S) == 2

    # 4. Policies: static caching beats no caching; incremental caching also caches the history.
    static = simulate_session(S, T, "static", U, A)
    incremental = simulate_session(S, T, "incremental", U, A)
    assert static.total_cost_usd < 0.5 * plain.total_cost_usd
    assert incremental.total_cost_usd < static.total_cost_usd
    assert plain.total_input_tokens == static.total_input_tokens == incremental.total_input_tokens  # same tokens, different bill

    # 5. Two-tier context is cheaper than a cached monolith only while the reads stay few.
    monolith = simulate_session(30_000, 12, "incremental")
    lean_no_reads = simulate_session(1_500, 12, "incremental")
    lean_three_reads = simulate_session(1_500, 12, "incremental", reads_per_turn=3)
    assert lean_no_reads.total_cost_usd < monolith.total_cost_usd
    assert lean_three_reads.total_cost_usd > monolith.total_cost_usd
    assert lean_three_reads.calls == 4 * 12  # every read is one more model call

    print("[OK] KV cache cost calculator: closed form, quadratic growth, breakeven, policies, and two-tier crossover verified.")


if __name__ == "__main__":
    print("Running KV Cache Cost & Token Simulation...")
    run_verification_suite()

    turns = 12
    scenarios = [
        ("Monolith 30k, no cache", dict(prefix_tokens=30_000, policy="none")),
        ("Monolith 30k, static-prefix cache", dict(prefix_tokens=30_000, policy="static")),
        ("Monolith 30k, incremental cache", dict(prefix_tokens=30_000, policy="incremental")),
        ("Two-tier 1.5k, no reads", dict(prefix_tokens=1_500, policy="incremental")),
        ("Two-tier 1.5k, 1 read of 2k per turn", dict(prefix_tokens=1_500, policy="incremental", reads_per_turn=1)),
        ("Two-tier 1.5k, 3 reads of 2k per turn", dict(prefix_tokens=1_500, policy="incremental", reads_per_turn=3)),
    ]
    print(f"\n--- Simulation: {turns} user turns, example prices ---")
    for label, kwargs in scenarios:
        r = simulate_session(turns=turns, **kwargs)
        print(f"{label:40s} calls={r.calls:3d}  input={r.total_input_tokens:9,d}  cost=${r.total_cost_usd:.4f}")
    print(f"\nBreak-even turn for caching a 30k static prefix: Turn {calculate_cache_breakeven_turns(30_000)}")
