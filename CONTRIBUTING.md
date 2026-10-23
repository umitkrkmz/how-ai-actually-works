# Contributing to how-ai-actually-works

Thank you for helping. This repository explains AI mechanisms with small, runnable implementations, so
a contribution is judged on two things: does the code prove what the text claims, and is every claim
true? The [5 Invariant Rules](README.md#the-5-invariant-rules) are the short version of this file.

## Setup

You need Python 3.10 or newer and nothing else. There are no third-party dependencies, and a contribution
must not add one.

```bash
python verify_all.py
```

Run this before you open a pull request. It must exit `0`.

## What a contribution must meet

1. **The mechanism comes first.** Build it in plain standard-library Python before mentioning a framework,
   a vector database or an SDK. A pull request that introduces an abstraction library without first
   building the primitive will be closed.
2. **Code proves its claims.** Scripts and runnable README blocks end in `assert` statements. A pull
   request without runnable code and explicit assertions will be closed.
3. **Claims carry evidence.** Every claim about a model, a paper, a protocol or performance needs one of:
   - a source: a paper (authors and year), a specification with its version, or official documentation;
   - a measurement: a script in this repository, or the benchmarks in the companion repository;
   - a label saying it is an assumption, with the assumed values written out.

   Numbers that depend on the model, the hardware or the provider (latency, prices, memory overhead) must
   state the setup they come from. If you do not know something, say so in the text instead of guessing.
4. **No hype.** Plain engineering English, with no marketing adjectives. Buzzwords will be edited out.

## Adding or changing a script

- Keep it self-contained: standard library only, no network access, no files outside the repository.
- Print what it computes, then assert the behavior. The assertions should fail if the mechanism were
  wrong. Before you submit, break the function on purpose and check that the script fails;
  [module 06](06-evals-and-benchmarking/README.md) explains this as mutation testing.
- Where the implementation has a known blind spot, assert the blind spot as well, so nobody mistakes the
  small version for a complete one. `ast_guardrail.py`, `rag_evaluator.py` and `json_grammar_fsm.py` do
  this.
- Add a new script to `TEST_SCRIPTS` in [`verify_all.py`](verify_all.py) and to the directory structure in
  the [README](README.md#directory-structure).

## README code blocks

`verify_all.py` runs every block fenced as `python` in the module READMEs. Each block runs in a fresh
process, from its module directory, with a timeout. A block that contains no `assert` fails.

Some blocks only illustrate a pattern: an anti-pattern, a library call, or code that needs a live system.
Mark such a block by putting this comment on the last non-blank line before its fence. GitHub does not
render the comment.

````markdown
<!-- verify_all: illustrative fragment, not executed -->
```python
response = llm.generate(prompt)   # needs a model; shown for its shape only
```
````

The harness lists every skipped block in its output, so a skipped block is always visible. Use the marker
only when the block cannot run without something outside the standard library. Use `text`, `bash` or
`json` fences for content that is not Python.

## Module structure

Each module README follows the same outline:

1. Visual mental models (ASCII or Mermaid) before formulas and code.
2. The mechanism, with the math it needs and no more.
3. A `When NOT to Use This (Architecture Waste & Anti-Patterns)` section that compares the approach with
   boring alternatives on cost, latency and failure modes.
4. A decision matrix.
5. A runnable section, and links to the module's scripts.
6. A link to the next module.

## The companion repository

[ai-context-mapping](https://github.com/umitkrkmz/ai-context-mapping) holds the applied tools and the
measured evidence. This repository keeps the minimal versions. A change that turns a small implementation
here into a full production tool belongs there; link to it from here instead.

## Pull requests

- Keep each pull request to one topic.
- Say what you checked: the commands you ran, and the source or measurement behind any new claim.
- Line endings are LF on every OS (see `.gitattributes`).
- Contributions are accepted under the [MIT License](LICENSE).
