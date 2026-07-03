"""Concrete (live) verification helper for SolidCoder's assumption breaking.

Executes generated code plus an optional test script in an isolated namespace.
For stdin-driven programs (competitive-programming style), a standard-input
payload can be injected so the program is exercised for real instead of
crashing on a disabled ``input()``.
"""
import builtins
import io
import sys

from .executor_utils import function_with_timeout


def _make_stdin_input(stdin_buf):
    """Return an input() replacement that reads successive lines from stdin_buf."""
    def _input(prompt=None):
        line = stdin_buf.readline()
        if not line:
            raise EOFError("EOF when reading a line")
        return line.rstrip("\n")
    return _input


def concrete_verify_script(code, test_script, language="python", timeout=5, stdin_payload=None):
    """Run ``code`` followed by ``test_script`` and classify the outcome.

    Returns one of: "PASS", "FAIL_ASSERT", "FAIL_CRASH", "SKIP_NON_PY".

    Without ``stdin_payload`` (function-style problems, e.g. HumanEval) the
    behavior is unchanged from the original implementation: ``input()`` is
    disabled so accidental reads surface as crashes.

    With ``stdin_payload`` the program's ``input()`` and ``sys.stdin`` read
    from the payload and stdout is captured, so stdin-driven programs can be
    genuinely verified. ``sys.stdin``/``sys.stdout`` are always restored, even
    on timeout. Residual limitation: a timed-out (daemonized, abandoned) worker
    thread cannot be killed and may briefly race the restored streams.
    """
    if not language.lower().startswith("python"):
        return "SKIP_NON_PY"

    full_code = f"""
import sys
import math
from typing import List, Dict, Any, Optional, Union, Tuple

{code}

{test_script}
"""
    safe_builtins = builtins.__dict__.copy()
    old_stdin = sys.stdin
    old_stdout = sys.stdout
    try:
        if stdin_payload is None:
            safe_builtins["input"] = lambda *_, **__: (_ for _ in ()).throw(
                RuntimeError("input() disabled during live verify"))
        else:
            stdin_buf = io.StringIO(stdin_payload)
            safe_builtins["input"] = _make_stdin_input(stdin_buf)
            sys.stdin = stdin_buf
            sys.stdout = io.StringIO()
        exec_globals = {"__builtins__": safe_builtins}
        function_with_timeout(exec, (full_code, exec_globals), timeout)
        return "PASS"
    except AssertionError:
        return "FAIL_ASSERT"
    except Exception:
        return "FAIL_CRASH"
    finally:
        sys.stdin = old_stdin
        sys.stdout = old_stdout
