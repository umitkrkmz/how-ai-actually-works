"""mutation_tester.py - AST-Based Mutation Testing from Scratch.

This module demonstrates why 100% code coverage is often an illusion:
1. The Problem with 'Vibe Tests' and Mocked Suites:
   Tests can achieve 100% line coverage while asserting almost nothing, allowing
   regressions and broken business logic to pass undetected.
2. Mutation Testing Mechanics:
   - Deliberately injects small bugs (mutants) into the source code via AST manipulation:
     * Body Neutralization: Replaces function body with 'return None'.
     * Comparison Inversion: flips equality ('==' -> '!=', '!=' -> '==').
     * Boundary Shift: Off-by-one shifts ('>=' -> '>', '<=' -> '<').
   - Executes candidate test suites against each mutant.
   - If tests FAIL (red) -> Mutant is KILLED (High test quality).
   - If tests PASS (green) -> Mutant SURVIVED (Weak test quality).
3. Mutation Score Formula:
   Mutation Score = (Killed Mutants / Total Mutants) * 100%

Simplifications in this demo: each operator class rewrites every matching site at once (the target has one
of each), and mutants run in memory via exec. Real tools mutate one site at a time, run the project's own
test command against a temporary copy, and restore the file. Some mutants are *equivalent* (they do not
change behavior), so a surviving mutant needs a human look before you call a test weak. For a tool that
does this on real files, see scripts/mutation_guard.py in github.com/umitkrkmz/ai-context-mapping.

Zero external dependencies. Pure Python standard library.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from typing import Callable, List, Tuple


TARGET_FUNCTION_SOURCE = """
def is_authorized(role: str, requests_count: int, max_limit: int = 100) -> bool:
    if role == "admin":
        return True
    if requests_count >= max_limit:
        return False
    return True
"""


class BodyNeutralizer(ast.NodeTransformer):
    """Mutant 1: Replaces the entire function body with 'return None'."""
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        new_node = copy.deepcopy(node)
        new_node.body = [ast.Return(value=ast.Constant(value=None))]
        return new_node


class ComparisonInverter(ast.NodeTransformer):
    """Mutant 2: Flips '==' and '!='. In the target this turns role == "admin" into role != "admin"."""
    def visit_Compare(self, node: ast.Compare) -> ast.Compare:
        new_node = copy.deepcopy(node)
        new_ops = []
        for op in new_node.ops:
            if isinstance(op, ast.Eq):
                new_ops.append(ast.NotEq())
            elif isinstance(op, ast.NotEq):
                new_ops.append(ast.Eq())
            else:
                new_ops.append(op)
        new_node.ops = new_ops
        return new_node


class BoundaryShifter(ast.NodeTransformer):
    """Mutant 3: Shifts boundary check from '>=' to '>' (Off-by-one error)."""
    def visit_Compare(self, node: ast.Compare) -> ast.Compare:
        new_node = copy.deepcopy(node)
        new_ops = []
        for op in new_node.ops:
            if isinstance(op, ast.GtE):
                new_ops.append(ast.Gt())
            elif isinstance(op, ast.LtE):
                new_ops.append(ast.Lt())
            else:
                new_ops.append(op)
        new_node.ops = new_ops
        return new_node


@dataclass
class MutationResult:
    total_mutants: int
    killed_mutants: int
    survived_mutants: int
    mutation_score_pct: float
    details: list[tuple[str, str]]  # (mutant_name, "KILLED" | "SURVIVED")


def generate_mutants(source_code: str) -> list[tuple[str, str]]:
    """Generates mutated source code variants via AST transformations."""
    tree = ast.parse(source_code)
    mutants: list[tuple[str, str]] = []

    # 1. Mutant: Body Neutralizer
    m1_tree = BodyNeutralizer().visit(copy.deepcopy(tree))
    ast.fix_missing_locations(m1_tree)
    mutants.append(("BodyNeutralization", ast.unparse(m1_tree)))

    # 2. Mutant: Comparison Inverter
    m2_tree = ComparisonInverter().visit(copy.deepcopy(tree))
    ast.fix_missing_locations(m2_tree)
    mutants.append(("ComparisonInversion_EqToNotEq", ast.unparse(m2_tree)))

    # 3. Mutant: Boundary Shifter
    m3_tree = BoundaryShifter().visit(copy.deepcopy(tree))
    ast.fix_missing_locations(m3_tree)
    mutants.append(("BoundaryShift_GtEToGt", ast.unparse(m3_tree)))

    return mutants


def evaluate_test_suite_against_mutants(
    mutants: list[tuple[str, str]],
    fn_name: str,
    test_suite_fn: Callable[[Callable], None],
) -> MutationResult:
    """Executes a test suite against all mutants and calculates the Mutation Score."""
    killed = 0
    survived = 0
    details = []

    for mutant_name, mutant_code in mutants:
        scope: dict = {}
        exec(mutant_code, scope)
        mutated_func = scope[fn_name]

        try:
            test_suite_fn(mutated_func)
            # If the test suite passed without exception, the bug survived!
            survived += 1
            details.append((mutant_name, "SURVIVED"))
        except (AssertionError, Exception):
            # Test caught the regression! Mutant killed.
            killed += 1
            details.append((mutant_name, "KILLED"))

    total = len(mutants)
    score = (killed / total) * 100.0 if total > 0 else 0.0
    return MutationResult(
        total_mutants=total,
        killed_mutants=killed,
        survived_mutants=survived,
        mutation_score_pct=score,
        details=details,
    )


# --- SAMPLE TEST SUITES ---

def weak_mock_test_suite(fn: Callable) -> None:
    """A weak test suite: Asserts almost nothing and mocks out boundary conditions."""
    # This weak test does not check return values strictly or test the exact boundary (100)
    res = fn("admin", 10)
    # Allows None, True, False — survives body neutralization and boolean bugs!
    assert res in [True, False, None]


def strict_boundary_test_suite(fn: Callable) -> None:
    """A comprehensive test suite: Tests boundary conditions and negative constraints."""
    # Check 1: Admin override must work regardless of request count
    assert fn("admin", 9999) is True, "Admin should always be authorized"

    # Check 2: Non-admin under limit must pass
    assert fn("user", 99, 100) is True, "User under limit should be authorized"

    # Check 3: Exact boundary check (requests_count == max_limit MUST be blocked)
    # This strictly kills BoundaryShift (>= vs >)
    assert fn("user", 100, 100) is False, "User at limit MUST be blocked (>= 100)"

    # Check 4: Non-admin over limit must be blocked
    assert fn("user", 101, 100) is False, "User over limit should be blocked"


def boundary_blind_test_suite(fn: Callable) -> None:
    """The strict suite without its exact-boundary check: looks thorough, misses the off-by-one."""
    assert fn("admin", 9999) is True
    assert fn("user", 99, 100) is True
    assert fn("user", 101, 100) is False


def run_verification_suite() -> None:
    """Verifies that mutation scoring correctly identifies weak vs. rigorous test suites."""
    mutants = generate_mutants(TARGET_FUNCTION_SOURCE)
    assert len(mutants) == 3, f"Expected 3 distinct AST mutants, got {len(mutants)}"
    original = ast.unparse(ast.parse(TARGET_FUNCTION_SOURCE))
    assert all(code != original for _, code in mutants), "every mutant must change the code"
    assert 'role != "admin"' in dict(mutants)["ComparisonInversion_EqToNotEq"].replace("'", '"')

    # 1. Weak Test Suite Evaluation
    weak_res = evaluate_test_suite_against_mutants(mutants, "is_authorized", weak_mock_test_suite)
    # The weak test suite allows all mutants to survive!
    assert weak_res.killed_mutants == 0, f"Weak test should kill 0 mutants, killed {weak_res.killed_mutants}"
    assert weak_res.mutation_score_pct == 0.0, f"Weak test score should be 0%, got {weak_res.mutation_score_pct}%"

    # 2. Strict Boundary Test Suite Evaluation
    strict_res = evaluate_test_suite_against_mutants(mutants, "is_authorized", strict_boundary_test_suite)
    # The strict test suite kills every single mutant (Body neutralization, == -> !=, and >= -> >)
    assert strict_res.killed_mutants == 3, (
        f"Strict test should kill all 3 mutants, killed {strict_res.killed_mutants}"
    )
    assert strict_res.mutation_score_pct == 100.0, (
        f"Strict test score should be 100%, got {strict_res.mutation_score_pct}%"
    )

    # 3. Remove one boundary check and exactly the off-by-one mutant survives.
    blind_res = evaluate_test_suite_against_mutants(mutants, "is_authorized", boundary_blind_test_suite)
    assert dict(blind_res.details) == {"BodyNeutralization": "KILLED", "ComparisonInversion_EqToNotEq": "KILLED",
                                       "BoundaryShift_GtEToGt": "SURVIVED"}, blind_res.details

    print("[OK] AST mutation testing: weak suite 0%, boundary-blind suite 67% (off-by-one survives), strict suite 100%.")


if __name__ == "__main__":
    print("Executing AST Mutation Testing Verification...")
    run_verification_suite()

    # Demonstrative printout
    print("\n--- Mutation Testing Results Comparison ---")
    muts = generate_mutants(TARGET_FUNCTION_SOURCE)
    weak_r = evaluate_test_suite_against_mutants(muts, "is_authorized", weak_mock_test_suite)
    strict_r = evaluate_test_suite_against_mutants(muts, "is_authorized", strict_boundary_test_suite)

    print(f"Weak Mock Test Suite Mutation Score  : {weak_r.mutation_score_pct:.1f}% ({weak_r.killed_mutants}/{weak_r.total_mutants} killed)")
    print(f"Strict Boundary Test Mutation Score  : {strict_r.mutation_score_pct:.1f}% ({strict_r.killed_mutants}/{strict_r.total_mutants} killed)")
    print("\nMutant Breakdown (Strict Suite):")
    for name, status in strict_r.details:
        print(f"  [{status}] {name}")
