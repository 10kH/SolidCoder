"""Regression tests for the post-camera-ready bugfixes (W1-W4).

Run from the repository root:
    python -m unittest tests.test_bugfixes -v

No API keys or ExecEval server required.
"""
import contextlib
import io
import os
import sys
import threading
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
os.chdir(REPO_ROOT)  # evaluations.evalute resolves ./src/... relative to cwd

from evaluations.executor_utils import function_with_timeout
from evaluations import func_evaluate
from evaluations.concrete_verify import concrete_verify_script


class TestW4TimeoutThread(unittest.TestCase):
    def test_timeout_raises_and_worker_is_daemon(self):
        before = set(threading.enumerate())
        with self.assertRaises(TimeoutError):
            function_with_timeout(exec, ("while True: pass", {}), 1)
        leaked = [t for t in threading.enumerate() if t not in before and t.is_alive()]
        self.assertTrue(leaked, "expected the timed-out worker to still be alive")
        for t in leaked:
            self.assertTrue(t.daemon, "timed-out worker must be a daemon thread")

    def test_return_value_unchanged(self):
        self.assertEqual(function_with_timeout(int, ("5",), 2), 5)

    def test_exception_propagates(self):
        with self.assertRaises(ZeroDivisionError):
            function_with_timeout(exec, ("1/0", {}), 2)


class TestW3FreshNamespace(unittest.TestCase):
    def test_passing_and_failing_results(self):
        completion = "def add(a, b):\n    return a + b\n"
        test = "def check(candidate):\n    assert candidate(1, 2) == 3\n"
        self.assertEqual(
            func_evaluate.evaluate_functional_correctness(test, "add", completion), "passed")
        bad = "def add(a, b):\n    return a - b\n"
        self.assertTrue(
            func_evaluate.evaluate_functional_correctness(test, "add", bad).startswith("failed"))

    def test_no_module_globals_pollution(self):
        completion = "MARKER_XYZ = 42\ndef noop():\n    return MARKER_XYZ\n"
        test = "def check(candidate):\n    assert candidate() == 42\n"
        self.assertEqual(
            func_evaluate.evaluate_functional_correctness(test, "noop", completion), "passed")
        self.assertNotIn("MARKER_XYZ", vars(func_evaluate))

    def test_no_cross_call_contamination(self):
        defines = "def helper_xyz():\n    return 1\n"
        passed, _ = func_evaluate.evaluate_io(["assert helper_xyz() == 1"], defines)
        self.assertTrue(passed)
        # A later completion must NOT see helper_xyz from the earlier call.
        passed, _ = func_evaluate.evaluate_io(["assert helper_xyz() == 1"], "x = 1")
        self.assertFalse(passed)


class TestW1ConcreteVerify(unittest.TestCase):
    STDIN_CODE = (
        "def solve(a, b):\n"
        "    return a + b\n"
        "\n"
        "a, b = map(int, input().split())\n"
        "print(solve(a, b))\n"
    )

    def test_stdin_program_passes_with_payload(self):
        self.assertEqual(
            concrete_verify_script(self.STDIN_CODE, "", stdin_payload="1 2\n"), "PASS")

    def test_stdin_program_crash_detected(self):
        code = "x = int(input())\nprint(1 // x)\n"
        self.assertEqual(
            concrete_verify_script(code, "", stdin_payload="0\n"), "FAIL_CRASH")

    def test_stdin_assert_failure_detected(self):
        code = "x = int(input())\nassert x > 0\n"
        self.assertEqual(
            concrete_verify_script(code, "", stdin_payload="-5\n"), "FAIL_ASSERT")

    def test_sys_stdin_readline_supported(self):
        code = "import sys\nline = sys.stdin.readline()\nprint(len(line))\n"
        self.assertEqual(
            concrete_verify_script(code, "", stdin_payload="hello\n"), "PASS")

    def test_input_eof_raises(self):
        code = "input()\ninput()\n"
        self.assertEqual(
            concrete_verify_script(code, "", stdin_payload="only-one-line\n"), "FAIL_CRASH")

    def test_streams_restored_after_success_and_crash(self):
        old_stdin, old_stdout = sys.stdin, sys.stdout
        concrete_verify_script(self.STDIN_CODE, "", stdin_payload="1 2\n")
        self.assertIs(sys.stdin, old_stdin)
        self.assertIs(sys.stdout, old_stdout)
        concrete_verify_script("1/0", "", stdin_payload="x\n")
        self.assertIs(sys.stdin, old_stdin)
        self.assertIs(sys.stdout, old_stdout)

    def test_streams_restored_after_timeout(self):
        old_stdin, old_stdout = sys.stdin, sys.stdout
        status = concrete_verify_script("while True: pass", "", timeout=1, stdin_payload="1\n")
        self.assertEqual(status, "FAIL_CRASH")
        self.assertIs(sys.stdin, old_stdin)
        self.assertIs(sys.stdout, old_stdout)

    def test_legacy_function_style_unchanged(self):
        code = "def add(a, b):\n    return a + b\n"
        self.assertEqual(concrete_verify_script(code, "assert add(1, 2) == 3"), "PASS")
        self.assertEqual(concrete_verify_script(code, "assert add(1, 2) == 4"), "FAIL_ASSERT")
        # Without a payload, input() must stay disabled (legacy behavior).
        self.assertEqual(concrete_verify_script("input()", ""), "FAIL_CRASH")

    def test_non_python_skipped(self):
        self.assertEqual(
            concrete_verify_script("int main(){}", "", language="C++"), "SKIP_NON_PY")


class TestW2ContestEvaluatorGuard(unittest.TestCase):
    def setUp(self):
        from evaluations import evalute
        self.evalute = evalute
        self._orig_execute = evalute.api_comm.execute_code

    def tearDown(self):
        self.evalute.api_comm.execute_code = self._orig_execute

    def test_dict_tests_behavior_identical(self):
        tests = [{"input": "1 2", "output": ["3"]}]
        self.evalute.api_comm.execute_code = lambda **kw: (
            [{"exec_outcome": "PASSED", "result": "3"}], None, kw.get("task_id"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            passed, feedback = self.evalute.contest_evaluate_public_tests(
                "print(3)", "Python3", 1, tests)
        self.assertTrue(passed)
        self.assertIn("## Tested passed:", feedback)
        self.assertIn("Input:\n1 2", feedback)
        self.assertNotIn("WARNING", buf.getvalue())

    def test_malformed_tests_logged_without_behavior_change(self):
        tests = ["assert solve(1) == 2"]  # legacy bug shape: raw script string
        self.evalute.api_comm.execute_code = lambda **kw: (None, "bad unittests", kw.get("task_id"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            passed, feedback = self.evalute.contest_evaluate_public_tests(
                "print(3)", "Python3", 1, tests)
        self.assertIn("WARNING", buf.getvalue())
        # Legacy control flow preserved: error path returns graceful False.
        self.assertFalse(passed)
        self.assertIn("Syntax Error Message", feedback)


class TestW2CompetitiveAdditionalIO(unittest.TestCase):
    """A competitive live-verify round must never push strings into additional_io."""

    def _make_strategy(self, is_competitive, responses):
        from promptings.SolidCoder import SolidCoder
        s = SolidCoder.__new__(SolidCoder)
        s.language = "Python3"
        s.verbose = 0
        s.is_competitive = is_competitive
        s.max_assumption_rounds = 1
        s.enable_live_verify = True
        s.enable_oracle_assert = False
        s.enable_defensive_test = True
        s.accumulated_inputs = []
        it = iter(responses)
        s.gpt_chat = lambda *a, **kw: next(it)
        return s

    def test_competitive_round_keeps_additional_io_empty(self):
        attack = "Assumption: division by zero\nInput:\n```\n0\n```\n"
        judge = "**VALID**"
        fix = "```Python3\nx = int(input())\nprint(1 // x if x else 0)\n```"
        s = self._make_strategy(True, [attack, judge, fix])
        additional_io = []
        code = "x = int(input())\nprint(1 // x)\n"
        s._run_assumption_breaking("Divide 1 by x", code, "", additional_io)
        self.assertEqual(additional_io, [])
        self.assertEqual(s.accumulated_inputs, ["0\n"])

    def test_humaneval_round_still_appends_to_additional_io(self):
        attack = (
            "Assumption: b may be zero\n"
            "Test Script:\n```python\nresult = div(1, 0)\n```\n"
        )
        judge = "**VALID**"
        fix = "```Python3\ndef div(a, b):\n    return a / b if b else 0\n```"
        s = self._make_strategy(False, [attack, judge, fix])
        additional_io = []
        code = "def div(a, b):\n    return a / b\n"
        s._run_assumption_breaking("Divide a by b", code, "", additional_io)
        self.assertEqual(additional_io, ["result = div(1, 0)"])
        self.assertEqual(s.accumulated_inputs, ["result = div(1, 0)"])


if __name__ == "__main__":
    unittest.main()
