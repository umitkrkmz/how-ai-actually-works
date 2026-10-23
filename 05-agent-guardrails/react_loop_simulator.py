"""react_loop_simulator.py - Bare-Metal ReAct Loop & Deterministic Circuit Breakers.

This module simulates the mechanics and failure modes of the ReAct (Reason -> Act -> Observe) pattern:
1. The Standard ReAct Loop: Iterative execution of (Thought, Action, Observation) cycles.
2. Derailment Failure Modes:
   - Infinite Ping-Pong / Repetition Loops: Model repeatedly invokes the same tool with identical args.
   - Step Budget Exhaustion: Model wanders aimlessly without converging on an exit condition.
3. Deterministic Circuit Breakers:
   - Maximum Step Budget: Absolute upper bound on turns (preventing token exhaustion).
   - Repetition Detector: halts before the (N+1)-th consecutive identical action (same tool, same
     arguments), so at most N identical calls in a row are executed.
   (A further breaker you can add: halt when N consecutive observations are identical, which catches an
   agent that varies its arguments trivially while making no progress.)

Zero external dependencies. Pure Python standard library.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class StepRecord:
    step_number: int
    thought: str
    action_name: str
    action_args: dict[str, Any]
    observation: str


class CircuitBreakerTripped(Exception):
    """Raised when deterministic safeguards halt an errant agent loop."""
    pass


class ReActAgentLoop:
    """A deterministic execution harness for ReAct agent cycles."""

    def __init__(
        self,
        max_steps: int = 10,
        max_identical_actions: int = 2,
    ) -> None:
        self.max_steps = max_steps
        self.max_identical_actions = max_identical_actions
        self.history: list[StepRecord] = []
        self._action_fingerprints: list[str] = []

    def _hash_action(self, action_name: str, args: dict[str, Any]) -> str:
        """Computes a deterministic hash of an action and its serialized arguments."""
        canonical_json = json.dumps({"action": action_name, "args": args}, sort_keys=True, default=str)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def step(
        self,
        thought: str,
        action_name: str,
        action_args: dict[str, Any],
        tool_runner: Callable[[str, dict[str, Any]], str],
    ) -> StepRecord:
        """Executes a single ReAct step through deterministic circuit breakers."""
        current_step = len(self.history) + 1

        # Circuit Breaker 1: Hard Step Budget
        if current_step > self.max_steps:
            raise CircuitBreakerTripped(
                f"HALT: Maximum step budget of {self.max_steps} exceeded. "
                "Agent is wandering or stuck in an infinite planning loop."
            )

        # Circuit Breaker 2: Consecutive Repetitive Action Detection
        # Halt before executing the (N+1)-th identical action in a row, N = max_identical_actions.
        fp = self._hash_action(action_name, action_args)
        previous = self._action_fingerprints[-self.max_identical_actions :]
        if len(previous) == self.max_identical_actions and all(h == fp for h in previous):
            raise CircuitBreakerTripped(
                f"HALT: Infinite loop detected! Tool '{action_name}' already ran {self.max_identical_actions} "
                f"consecutive times with identical arguments: {action_args}"
            )
        self._action_fingerprints.append(fp)

        # Execute the actual deterministic tool
        observation = tool_runner(action_name, action_args)

        record = StepRecord(
            step_number=current_step,
            thought=thought,
            action_name=action_name,
            action_args=action_args,
            observation=observation,
        )
        self.history.append(record)
        return record


def run_verification_suite() -> None:
    """Verifies standard loop completion, repetition arrest, and step budget enforcement."""

    # Mock tool runner simulating environment
    def mock_tools(action: str, args: dict[str, Any]) -> str:
        if action == "query_db":
            return json.dumps({"rows": [1, 2, 3], "status": "ok"})
        elif action == "write_report":
            return json.dumps({"status": "written", "bytes": 104})
        return json.dumps({"error": "unknown_tool"})

    # 1. Test Scenario A: Normal Converging Agent
    agent_a = ReActAgentLoop(max_steps=5, max_identical_actions=2)

    s1 = agent_a.step(
        thought="I need to fetch active records first.",
        action_name="query_db",
        action_args={"table": "users"},
        tool_runner=mock_tools,
    )
    assert s1.step_number == 1
    assert "rows" in s1.observation

    s2 = agent_a.step(
        thought="Data received, now writing summary report.",
        action_name="write_report",
        action_args={"filename": "summary.txt"},
        tool_runner=mock_tools,
    )
    assert s2.step_number == 2
    assert len(agent_a.history) == 2

    # 2. Test Scenario B: Repetitive Tool-Call Loop Trap
    agent_b = ReActAgentLoop(max_steps=10, max_identical_actions=3)
    loop_halted = False

    # Simulate an agent hallucinating and repeatedly querying the same missing table
    try:
        for i in range(5):
            agent_b.step(
                thought="Maybe if I query again it will appear.",
                action_name="query_db",
                action_args={"table": "missing_table"},
                tool_runner=mock_tools,
            )
    except CircuitBreakerTripped as err:
        loop_halted = True
        assert "Infinite loop detected!" in str(err)
        assert agent_b.history[-1].step_number == 3  # 3 identical calls ran; the 4th was refused

    assert loop_halted is True, "Circuit breaker failed to halt repetitive execution loop"
    assert len(agent_b.history) == 3

    # 2b. A different action in between resets the count (A, A, B, A, A is not a loop of 3)
    agent_d = ReActAgentLoop(max_steps=10, max_identical_actions=2)
    for name in ("query_db", "query_db", "write_report", "query_db", "query_db"):
        agent_d.step("alternating", name, {"table": "t"}, mock_tools)
    assert len(agent_d.history) == 5

    # 3. Test Scenario C: Step Budget Enforcement (Wandering Agent)
    agent_c = ReActAgentLoop(max_steps=4, max_identical_actions=5)
    budget_halted = False

    try:
        for step_idx in range(10):
            # Agent calls different args every turn to evade repetition checker, but wanders
            agent_c.step(
                thought=f"Exploring random pathway {step_idx}...",
                action_name="query_db",
                action_args={"table": f"table_{step_idx}"},
                tool_runner=mock_tools,
            )
    except CircuitBreakerTripped as err:
        budget_halted = True
        assert "Maximum step budget of 4 exceeded" in str(err)

    assert budget_halted is True, "Circuit breaker failed to enforce step budget"
    assert len(agent_c.history) == 4

    print("[OK] ReAct loop simulator: convergence, repetition arrest, and budget enforcement verified.")


if __name__ == "__main__":
    print("Executing ReAct Loop Simulator Verification...")
    run_verification_suite()

    # Manual walkthrough demonstration
    print("\n--- Demonstrating Circuit Breaker Halting Errant Agent ---")
    breaker = ReActAgentLoop(max_steps=5, max_identical_actions=2)
    dummy_runner = lambda action, args: "No rows found."

    try:
        print("Step 1: Agent calls read_file('secrets.env')...")
        breaker.step("Reading config", "read_file", {"path": "secrets.env"}, dummy_runner)
        print("Step 2: Agent calls read_file('secrets.env') again (retrying)...")
        breaker.step("Trying again", "read_file", {"path": "secrets.env"}, dummy_runner)
        print("Step 3: Agent calls read_file('secrets.env') a third time...")
        breaker.step("Once more", "read_file", {"path": "secrets.env"}, dummy_runner)
    except CircuitBreakerTripped as cb:
        print(f"-> CIRCUIT BREAKER TRIPPED: {cb}")
