"""verify_all.py - Master Verification Harness for how-ai-actually-works.

Executes every first-principles verification suite across Modules 01 through 08:
- The 15 module scripts, each in a fresh subprocess.
- Every ```python block in the module READMEs, each in a fresh subprocess run from its module
  directory. A block must contain at least one `assert`. Blocks that only illustrate a pattern
  (an anti-pattern, a library call, a live system) are marked with SKIP_MARKER on the last
  non-blank line before the fence; they are listed as skipped, never silently ignored.
- Formats results into an ASCII status table with millisecond-precision timing.
- Exits with return code 0 if every suite passes; exits with code 1 on any failure.

Zero external dependencies. Pure Python standard library.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import List, Tuple


# Ordered list of all module verification scripts
TEST_SCRIPTS: list[tuple[str, str]] = [
    # Module 01: Primitives & Embeddings
    ("01", os.path.join("01-primitives-and-embeddings", "cosine_math.py")),
    ("01", os.path.join("01-primitives-and-embeddings", "bpe_tokenizer.py")),
    # Module 02: Retrieval & RAG
    ("02", os.path.join("02-retrieval-and-rag", "naive_rag.py")),
    ("02", os.path.join("02-retrieval-and-rag", "chunking_comparison.py")),
    # Module 03: Context Engineering
    ("03", os.path.join("03-context-engineering", "kv_cache_calculator.py")),
    ("03", os.path.join("03-context-engineering", "context_position_tester.py")),
    # Module 04: Tools & MCP
    ("04", os.path.join("04-tools-and-mcp", "mcp_client_tester.py")),
    # Module 05: Agent Guardrails
    ("05", os.path.join("05-agent-guardrails", "react_loop_simulator.py")),
    ("05", os.path.join("05-agent-guardrails", "ast_guardrail.py")),
    # Module 06: Evals & Benchmarking
    ("06", os.path.join("06-evals-and-benchmarking", "rag_evaluator.py")),
    ("06", os.path.join("06-evals-and-benchmarking", "mutation_tester.py")),
    # Module 07: Decoding & Structured Outputs
    ("07", os.path.join("07-decoding-and-structured-outputs", "sampling_math.py")),
    ("07", os.path.join("07-decoding-and-structured-outputs", "json_grammar_fsm.py")),
    # Module 08: Hardware, VRAM & Quantization
    ("08", os.path.join("08-hardware-vram-quantization", "vram_calculator.py")),
    ("08", os.path.join("08-hardware-vram-quantization", "int8_quantizer.py")),
]

# Put this HTML comment (invisible on GitHub) on the last non-blank line before a ```python fence
# to mark the block as an illustrative fragment that is not executed.
SKIP_MARKER = "<!-- verify_all: illustrative fragment, not executed -->"

TIMEOUT_SECONDS = 120


@dataclass
class ReadmeBlock:
    line: int        # 1-based line number of the opening fence
    code: str
    skipped: bool


def extract_python_blocks(markdown: str) -> list[ReadmeBlock]:
    """Returns every ```python block, scanning line by line.

    A regex over the whole file breaks on a block that contains a line such as '```json' inside a
    string. A fence closes only on a line that is exactly ``` after stripping, as on GitHub.
    """
    lines = markdown.splitlines()
    blocks: list[ReadmeBlock] = []
    last_text = ""
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == "```python":
            indent = len(lines[i]) - len(lines[i].lstrip())
            j = i + 1
            while j < len(lines) and lines[j].strip() != "```":
                j += 1
            if j == len(lines):
                raise ValueError(f"unclosed ```python fence at line {i + 1}")
            code = textwrap.dedent("\n".join(l[indent:] if l[:indent].isspace() else l for l in lines[i + 1:j]))
            blocks.append(ReadmeBlock(line=i + 1, code=code + "\n", skipped=last_text == SKIP_MARKER))
            last_text = "```"
            i = j + 1
            continue
        if stripped:
            last_text = stripped
        i += 1
    return blocks


def _check_extractor() -> None:
    """The harness checks its own parser before trusting it."""
    sample = "\n".join([
        "Intro text.",
        "```python",
        "s = '```json'",          # a fence-like string must not close the block
        "assert s.endswith('json')",
        "```",
        "",
        SKIP_MARKER,
        "",
        "```python",
        "fragment()",
        "```",
        "Text between blocks.",
        "```python",
        "after_text = True",     # the marker does not leak past the next block
        "```",
        "1. A list item:",
        "   ```python",
        "   assert 1 + 1 == 2",
        "   ```",
    ])
    blocks = extract_python_blocks(sample)
    assert [(b.line, b.skipped) for b in blocks] == [(2, False), (9, True), (13, False), (17, False)], blocks
    assert blocks[0].code == "s = '```json'\nassert s.endswith('json')\n"
    assert blocks[3].code == "assert 1 + 1 == 2\n"
    try:
        extract_python_blocks("```python\nx = 1\n")
    except ValueError:
        pass
    else:
        raise AssertionError("an unclosed fence must be reported")


def run_command(cmd: list[str], cwd: str) -> tuple[bool, float, str, str]:
    """Runs a command in a subprocess; returns pass/fail, timing, a one-line summary and stderr."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    start_time = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return False, elapsed_ms, f"timed out after {TIMEOUT_SECONDS}s", ""
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    stdout = proc.stdout.decode("utf-8", errors="replace").strip()
    stderr = proc.stderr.decode("utf-8", errors="replace").strip()
    passed = proc.returncode == 0
    source = stdout if passed else stderr
    summary = source.splitlines()[-1] if source else ""
    return passed, elapsed_ms, summary, stderr


def collect_readme_blocks(repo_root: str) -> list[tuple[str, str, ReadmeBlock]]:
    """Returns (module, README path, block) for every python block in the module READMEs."""
    found = []
    for readme in sorted(glob.glob(os.path.join(repo_root, "[0-9][0-9]-*", "README.md"))):
        rel_path = os.path.relpath(readme, repo_root).replace(os.sep, "/")
        with open(readme, encoding="utf-8") as f:
            for block in extract_python_blocks(f.read()):
                found.append((rel_path[:2], rel_path, block))
    return found


def main() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)
    _check_extractor()

    print("=" * 102)
    print("                      HOW-AI-ACTUALLY-WORKS: MASTER VERIFICATION HARNESS")
    print("                      Zero Abstractions. Pure Standard Library Python.")
    print("=" * 102)
    print(f"{'Mod':<5} | {'Verification Suite':<48} | {'Status':<8} | {'Time (ms)':<10} | {'Summary'}")
    print("-" * 102)

    results: list[tuple[str, bool, str]] = []   # (name, passed, stderr)
    skipped = 0
    total_start = time.perf_counter()

    def report(mod: str, name: str, passed: bool, duration_ms: float, summary: str, stderr: str) -> None:
        status_str = "PASSED" if passed else "FAILED"
        clean_summary = summary.replace("[OK] ", "")[:40] if passed else f"ERR: {summary[:35]}"
        print(f"{mod:<5} | {name:<48} | {status_str:<8} | {duration_ms:8.2f}ms | {clean_summary}")
        results.append((name, passed, stderr))

    for mod, rel_path in TEST_SCRIPTS:
        full_path = os.path.join(repo_root, rel_path)
        if not os.path.exists(full_path):
            print(f"{mod:<5} | {rel_path:<48} | {'MISSING':<8} | {'0.0ms':<10} | File not found on disk")
            results.append((rel_path, False, ""))
            continue
        report(mod, rel_path.replace(os.sep, "/"), *run_command([sys.executable, full_path], repo_root))

    print("-" * 102)
    for mod, rel_path, block in collect_readme_blocks(repo_root):
        name = f"{rel_path}:{block.line}"
        if block.skipped:
            print(f"{mod:<5} | {name:<48} | {'SKIPPED':<8} | {'':<10} | illustrative fragment")
            skipped += 1
        elif "assert" not in block.code:
            print(f"{mod:<5} | {name:<48} | {'FAILED':<8} | {'':<10} | no assert; add one or mark it")
            results.append((name, False, "A runnable block must assert its behavior (rule 3)."))
        else:
            report(mod, name, *run_command([sys.executable, "-c", block.code], os.path.dirname(rel_path)))

    total_duration = time.perf_counter() - total_start
    passed_count = sum(1 for _, passed, _ in results if passed)
    failures = [(name, stderr) for name, passed, stderr in results if not passed]
    print("=" * 102)

    if not failures:
        print(f"SUCCESS: All {passed_count}/{len(results)} verification suites PASSED in {total_duration:.2f}s "
              f"({len(TEST_SCRIPTS)} scripts, {len(results) - len(TEST_SCRIPTS)} README blocks; "
              f"{skipped} illustrative fragments skipped).")
        print("Every assertion in every script and runnable README block held.")
        print("=" * 102)
        sys.exit(0)
    else:
        print(f"FAILURE: {len(failures)} suite(s) failed out of {len(results)}.")
        for name, stderr in failures:
            print(f"\n--- {name} ---\n{stderr}")
        print("=" * 102)
        sys.exit(1)


if __name__ == "__main__":
    main()
