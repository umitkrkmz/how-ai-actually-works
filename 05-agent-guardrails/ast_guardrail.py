"""ast_guardrail.py - Deterministic AST-Level Code Boundary Guardrail.

A pre-commit / pre-merge gate that inspects Python source without running it:

1. Import allowlist: every absolute import (ast.Import, ast.ImportFrom) must name an allowed top-level
   package. Relative imports (from . import x) are local code and always allowed. Dynamic imports
   (__import__, importlib.import_module) and eval/exec are flagged, because they would bypass the check.
2. Negative invariants on protected functions: each must still exist (def or async def), still carry
   its mandatory decorators (plain or dotted, e.g. @require_mfa or @auth.require_mfa), and still contain
   its mandatory calls (e.g. record_audit_event).
3. Fails with an exception (exit code 1 from a CLI wrapper) on the first violation.

What a static check cannot see, and what the suite below demonstrates on purpose:
- Names are not semantics. A decorator called require_mfa that does nothing passes.
- Presence is not execution. A required call inside `if False:` passes.
So this gate catches the deletions and additions it was told about, deterministically; it does not prove
the code is safe. Pair it with tests (and with mutation testing, module 06) for behavior.

Zero external dependencies. Pure Python standard library.
"""

from __future__ import annotations

import ast
from typing import Dict, Iterable, List, Set, Union

FunctionNode = Union[ast.FunctionDef, ast.AsyncFunctionDef]
DYNAMIC_CODE_CALLS = {"__import__", "import_module", "eval", "exec"}


class SecurityInvariantViolation(Exception):
    """Raised when candidate code violates deterministic AST security policies."""


def _dotted_name(node: ast.AST) -> str:
    """'require_mfa' for a Name, 'auth.require_mfa' for an Attribute, '' otherwise."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _decorator_names(fn: FunctionNode) -> Set[str]:
    """Both the full dotted name and its last part, for every decorator (called or not)."""
    names: Set[str] = set()
    for dec in fn.decorator_list:
        full = _dotted_name(dec.func if isinstance(dec, ast.Call) else dec)
        if full:
            names.update({full, full.rsplit(".", 1)[-1]})
    return names


def _called_names(fn: FunctionNode) -> Set[str]:
    names: Set[str] = set()
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Call):
            full = _dotted_name(sub.func)
            if full:
                names.update({full, full.rsplit(".", 1)[-1]})
    return names


class CodeGuardrailAST:
    """Deterministic AST inspector enforcing a dependency allowlist and negative invariants."""

    def __init__(
        self,
        allowed_modules: Set[str],
        required_decorators: Dict[str, List[str]],
        required_calls: Dict[str, List[str]],
    ) -> None:
        self.allowed_modules = allowed_modules
        self.required_decorators = required_decorators
        self.required_calls = required_calls

    def validate_code(self, source_code: str) -> None:
        """Parse and validate source code; raise SecurityInvariantViolation on the first violation."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            raise SecurityInvariantViolation(f"Code failed basic syntax compilation: {e}")
        self._check_imports(tree)
        self._check_negative_invariants(tree)

    def _check_imports(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._require_allowed(alias.name, f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                self._require_allowed(node.module, f"from {node.module} import ...")
            elif isinstance(node, ast.Call):
                name = _dotted_name(node.func).rsplit(".", 1)[-1]
                if name in DYNAMIC_CODE_CALLS:
                    raise SecurityInvariantViolation(
                        f"DYNAMIC CODE: '{_dotted_name(node.func)}(...)' can bypass the import allowlist."
                    )

    def _require_allowed(self, module: str, statement: str) -> None:
        if module.split(".")[0] not in self.allowed_modules:
            raise SecurityInvariantViolation(
                f"FORBIDDEN IMPORT: '{statement}' is not in the allowlist {sorted(self.allowed_modules)}."
            )

    def _functions(self, tree: ast.AST) -> Dict[str, List[FunctionNode]]:
        found: Dict[str, List[FunctionNode]] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found.setdefault(node.name, []).append(node)
        return found

    def _check_negative_invariants(self, tree: ast.AST) -> None:
        """Agents must not 'refactor away' protected functions, their decorators, or their audit calls."""
        functions = self._functions(tree)
        protected: Iterable[str] = set(self.required_decorators) | set(self.required_calls)
        for name in sorted(protected):
            if name not in functions:
                raise SecurityInvariantViolation(
                    f"NEGATIVE INVARIANT VIOLATED: protected function '{name}' is missing (deleted or renamed)."
                )
            for fn in functions[name]:  # a redefinition must not slip through either
                present = _decorator_names(fn)
                for required in self.required_decorators.get(name, []):
                    if required not in present:
                        raise SecurityInvariantViolation(
                            f"NEGATIVE INVARIANT VIOLATED in '{name}': "
                            f"Mandatory security decorator '@{required}' was deleted or bypassed!"
                        )
                calls = _called_names(fn)
                for required in self.required_calls.get(name, []):
                    if required not in calls:
                        raise SecurityInvariantViolation(
                            f"NEGATIVE INVARIANT VIOLATED in '{name}': "
                            f"Mandatory audit/validation call '{required}(...)' was removed from function body!"
                        )


def _expect_violation(guardrail: CodeGuardrailAST, code: str, fragment: str) -> None:
    try:
        guardrail.validate_code(code)
    except SecurityInvariantViolation as err:
        assert fragment in str(err), f"expected {fragment!r} in {err}"
        return
    raise AssertionError(f"guardrail failed to flag: {fragment}")


VALID = '''
import json
from .helpers import money  # relative import: local code

@require_mfa
@audit.boundary
def execute_fund_transfer(account_id: str, amount_cents: int) -> bool:
    record_audit_event(account_id, amount_cents)
    return True
'''


def run_verification_suite() -> None:
    """Compliant code passes; each bypass is caught; the documented blind spots stay visible."""
    guardrail = CodeGuardrailAST(
        allowed_modules={"math", "json", "os", "sys", "typing", "collections", "hashlib"},
        required_decorators={"execute_fund_transfer": ["require_mfa", "boundary"]},
        required_calls={"execute_fund_transfer": ["record_audit_event"]},
    )

    # 1. Compliant code passes (dotted decorator and relative import included).
    guardrail.validate_code(VALID)

    # 2. Forbidden dependencies, static and dynamic.
    _expect_violation(guardrail, VALID + "\nimport requests\n", "FORBIDDEN IMPORT: 'import requests'")
    _expect_violation(guardrail, VALID + "\nfrom langchain.llms import X\n", "from langchain.llms")
    _expect_violation(guardrail, VALID + "\nrequests = __import__('requests')\n", "DYNAMIC CODE: '__import__")
    _expect_violation(guardrail, VALID + "\nimport importlib\n", "FORBIDDEN IMPORT")

    # 3. Stripped decorator, stripped audit call.
    _expect_violation(guardrail, VALID.replace("@require_mfa\n", ""), "'@require_mfa' was deleted")
    _expect_violation(guardrail, VALID.replace("    record_audit_event(account_id, amount_cents)\n", ""),
                      "'record_audit_event(...)' was removed")

    # 4. Bypasses an earlier version missed: delete the function, rename it, make it async, redefine it.
    _expect_violation(guardrail, "import json\n", "'execute_fund_transfer' is missing")
    _expect_violation(guardrail, VALID.replace("def execute_fund_transfer", "def execute_transfer"), "is missing")
    _expect_violation(guardrail, VALID.replace("def execute_fund_transfer", "async def execute_fund_transfer").replace("@require_mfa\n", ""),
                      "'@require_mfa' was deleted")
    _expect_violation(guardrail, VALID + "\ndef execute_fund_transfer(a, b):\n    return True\n", "was deleted")

    # 5. Blind spots, asserted so nobody mistakes this gate for a proof of safety.
    hollow = "def require_mfa(f):\n    return f  # does nothing\n" + VALID.replace("from .helpers import money  # relative import: local code\n", "")
    guardrail.validate_code(hollow)  # names are not semantics: a do-nothing require_mfa passes
    dead_call = VALID.replace("    record_audit_event(", "    if False:\n        record_audit_event(")
    guardrail.validate_code(dead_call)  # presence is not execution: a call that never runs passes

    print("[OK] AST guardrails: compliant code passes; imports, dynamic code, stripped and bypassed invariants caught; blind spots documented.")


if __name__ == "__main__":
    print("Executing AST Guardrail Invariant Checks...")
    run_verification_suite()
