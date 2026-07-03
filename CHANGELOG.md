# Changelog

## Post-camera-ready bugfixes (2026-07-03)

This release fixes latent defects found during a post-publication code review.

**Released numbers are unaffected.** All numbers reported in the paper live in
`results/` and are static artifacts; nothing under `results/` was regenerated
or modified. The two most significant bugs below (W1, W2) never fired in the
released runs: `additional_io` is empty in every released CodeContests and
APPS record, so the accumulation/corruption path they affect left no footprint
on any published score. The tag `acl2026-camera-ready` marks the repository as
submitted, before these fixes.

Because these are behavioral fixes, a **re-run** of current `main` may produce
different numbers than the released artifacts (see per-fix notes below). To
study the exact as-submitted behavior, check out the tag.

### W1 — Live Execution could not verify stdin-driven programs (fixed)

`_concrete_verify_script` disabled `input()` outright, so on competitive
datasets (APPS/CodeContests) — where the harness instructs generated code to
read from standard input — even a correct program crashed immediately and the
verdict was always `FAIL_CRASH`.

Fix: for competitive datasets the red-team prompt now asks for a breaking
*standard-input payload*, and the new `src/evaluations/concrete_verify.py`
injects it via an in-memory stdin (`input()` and `sys.stdin.readline()` both
work, stdout is captured, and the real streams are restored via `try/finally`
even on timeout). The HumanEval function-style path is byte-for-byte
unchanged: without a payload, `input()` stays disabled.

Disclosure: in the released runs the LLM judge rejected the function-call
style attack scripts on stdin problems before verification, which is why this
bug left no footprint. In judge-accepted rounds the forced `FAIL_CRASH` *may
have contributed* to Live Execution's ablation gain by triggering extra fix
rounds; the ablation delta alone cannot isolate that. Re-runs with genuine
stdin verification may therefore follow different debug trajectories.

### W2 — Defensive Accumulation could feed raw scripts to the contest evaluator (fixed)

On competitive datasets, assumption-breaking test scripts (plain Python
strings) were appended to `additional_io`, which the dataset evaluator
forwards to `contest_evaluate_public_tests` — an evaluator that requires
`{input, output}` dicts. A string there raises `TypeError` inside the
feedback formatter, and the bare `except` re-indexed the same string, letting
the exception escape and zero out the problem. This path never executed in
the released runs (`additional_io` empty in all records).

Fix: on competitive datasets, breaking stdin payloads are recorded in
`accumulated_inputs` only and are never pushed into `additional_io`; they are
verified through the concrete stdin path instead. The HumanEval branch is
unchanged (assert scripts still accumulate into `additional_io`).
`contest_evaluate_public_tests` additionally gained a *logging-only* guard
that warns loudly if non-dict test entries ever reach it; its control flow is
otherwise identical.

### W3 — Dynamic code executed in shared module globals (fixed)

`evaluate_io`, `evaluate_io_et`, and `evaluate_functional_correctness` ran
generated code with `exec(code, globals())`, i.e. inside the harness module's
own namespace, allowing state to leak across test cases and problems.

Fix: each execution now gets a fresh, empty namespace.

Numeric-safety audit (offline, deterministic, no APIs): every released
`results/HumanEval/**/Results.jsonl` record (2,461 records across 16 runs)
was re-scored with the fresh-namespace evaluator and compared against the
recorded `is_solved`. **2,460 of 2,461 match.** The single stable divergence:

- `SelfPlanning / grok-4.1-fast / HumanEval/150` — recorded `True`, fresh
  `False` (stable across 5 re-runs). The recorded final code calls
  `is_prime(n)` **without defining it**; it passed originally only because an
  earlier problem's code had left `is_prime` in the shared namespace. The
  recorded pass is itself a contamination artifact of this bug, inflating
  that baseline cell by one problem (159/164 → a re-run scores 158/164).

The released artifact is preserved as-is; only future re-runs are affected.

### W4 — Timed-out worker threads leaked and masked TimeoutError (fixed)

`function_with_timeout` used non-daemon threads, so code stuck in an infinite
loop kept the interpreter alive after long batch runs. In addition,
`PropagatingThread.ret` was unset on timeout, so callers received an
`AttributeError` instead of the intended `TimeoutError` (both were treated as
failures, so scoring was unaffected).

Fix: worker threads are daemonized and `ret` is initialized, restoring the
intended `TimeoutError`. Residual limitation: daemonizing does not *kill* an
abandoned worker; a timed-out thread may keep running (and, in the live-verify
stdin path, briefly race the restored `sys.stdin`/`sys.stdout`) until the
process exits.

### W5 — Hygiene

- Removed unused `signal`/`contextlib` imports from
  `src/evaluations/func_evaluate.py`.
- `requirements.txt` now documents that the exact paper-environment versions
  were not recorded; packages are intentionally left unpinned rather than
  pinned to guessed versions.

### Packaging — missing dataset modules restored

The published `src/` tree omitted `src/datasets/MBPPDataset.py` and
`src/datasets/XCodeDataset.py`, which `CodeSIM.py` and `DatasetFactory.py`
import unconditionally — so `src/main.py` could not even be imported from a
fresh clone. Both modules are restored from the original experiment tree
(`CodeAgent/CodeGenerator-master/src/datasets/`).

### Tests

`tests/test_bugfixes.py` (stdlib `unittest`, no API keys or ExecEval needed)
covers W1–W4: stdin injection semantics and stream restoration (including the
timeout path), fresh-namespace isolation, contest-evaluator guard behavior
identity, competitive rounds keeping `additional_io` empty, HumanEval rounds
preserving legacy accumulation, and daemonized timeout threads.
