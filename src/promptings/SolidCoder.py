"""
SolidCoder: CodeSIM + S.O.L.I.D. Architecture

S.O.L.I.D.:
- Shift-left Planning (Edge Case Awareness)
- Oracle-based Assertions (Assertion Verification)
- Live Execution (Concrete Verification)
- Intermediate Simulation (Code Simulation)
- Defensive Accumulation (Accumulative Testing)
"""

import re
import builtins
from typing import Optional

from evaluations.executor_utils import function_with_timeout

from .CodeSIM import (
    CodeSIM,
    prompt_for_planning,
    prompt_for_simulation,
    prompt_for_plan_refinement,
    prompt_for_code_generation,
    prompt_for_debugging,
)
from utils.parse import parse_response


# Assumption Breaking Prompts (originally from AssumeCoder)
# ============================================================

prompt_for_assumption_breaking = """You are an Assumption Breaker - a security-minded code reviewer whose job is to find hidden assumptions in code that could cause failures.

## Problem

{problem}

## Code

```{language}
{code}
```

## Your Mission: Find and Break Assumptions

**Core Principle**: Every bug is a violated assumption.

### Step 1: Hunt for Assumptions

Read the code carefully. What does it assume about its inputs?

**Types**:
- Does it assume integer when input could be float?
- Does it assume string when input could be None?
- Does it assume list when input could be empty?

**Values**:
- Does it assume positive numbers?
- Does it assume non-zero values?
- Does it assume values within certain range?

**Structure**:
- Does it assume non-empty collections?
- Does it assume specific length?
- Does it assume sorted order?

**Relationships**:
- Does it assume unique elements?
- Does it assume valid indices?
- Does it assume certain properties hold?

### Step 2: Challenge the Weakest Assumption

Pick the assumption most likely to be wrong in practice.
Construct a *specific, concrete input* that violates it.
Explain what would happen if this input is provided.

### Step 3: Verdict

**If you found a way to break the code**, respond with:

```
FAIL

Assumption Violated: <what the code wrongly assumed>
Breaking Input: <the specific input that breaks it>
Expected Behavior: <what correct code should do>
Actual Behavior: <what this code would do>
```

**If the code handles all reasonable edge cases** based on the problem constraints, respond with:

```
PASS

All assumptions are properly handled for the given problem constraints.
```

--------
**Important**: Focus on assumptions that matter for THIS specific problem. Don't invent edge cases that are outside the problem's constraints.
"""


prompt_for_assumption_fix = """You are a programmer fixing a vulnerability in code. An Assumption Breaker found that the code makes an incorrect assumption.

## Problem

{problem}

## Current Code (with vulnerability)

```{language}
{code}
```

## Vulnerability Report

{attack_result}

## Your Task

Fix the code to handle this case correctly. The fix should:
1. Address the specific assumption violation identified
2. Not break any existing functionality
3. Be minimal and focused

--------
**Important Instructions:**
- The fixed **{language}** code must be inside a triple backtick (```) code block.
- Add a comment explaining the fix.
{std_input_prompt}"""

from constants.verboseType import *


class SolidCoder(CodeSIM):
    def __init__(
        self,
        max_assumption_rounds=3,
        enable_shift_left=False,
        enable_oracle_assert=False,
        enable_live_verify=False,
        enable_inter_sim=False,
        enable_defensive_test=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.max_assumption_rounds = max_assumption_rounds
        self.enable_shift_left = enable_shift_left
        self.enable_oracle_assert = enable_oracle_assert
        self.enable_live_verify = enable_live_verify
        self.enable_inter_sim = enable_inter_sim
        self.enable_defensive_test = enable_defensive_test
        
        self.accumulated_inputs = []

    def run_single_pass(self, data_row: dict):
        print("", flush=True)

        problem = self.data.get_prompt(data_row)

        std_input_prompt = ""

        if self.is_competitive:
            std_input_prompt = \
"""- Strictly follow the sample input and output format. 
    - The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take the input using `input()` function then call the function with specified parameters and finally print the output of the function. 
    - For array input parse the array then pass it to the function. Parsing technique is given in the sample input output format section.
    - Do not add extra print statement otherwise it will failed the test cases."""
            marker = "-------\nImportant Note:"
            marker_idx = problem.find(marker)
            if marker_idx != -1:
                problem = problem[:marker_idx]

        additional_io = []
        self.accumulated_inputs = []
        self.run_details["additional_io"] = additional_io

        # Planning, Coding, Assumption Breaking, Debugging
        for plan_no in range(1, self.max_plan_try + 1):

            # ============================================================
            # PHASE 1: PLANNING (Shift-left optional)
            # ============================================================
            if self.enable_shift_left:
                plan_prompt_content = f"""
You are an expert software engineer.
You are given a problem and you need to generate a detailed plan to solve it.

## Problem
{problem}

## Killer Edge Cases
Identify 3 potential edge cases that could break a naive solution:
1. Empty/minimal input
2. Maximum constraint input
3. Special pattern (all same, alternating, etc.) or Boundary values

## Plan
Write a detailed plan that handles these edge cases.
The plan should be step-by-step and easy to implement.
"""
            else:
                plan_prompt_content = prompt_for_planning.format(
                    problem=problem,
                    language=self.language,
                )

            input_for_planning = [
                {
                    "role": "user",
                    "content": plan_prompt_content,
                },
            ]

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(f"[SolidCoder] Planning: {plan_no}\n\n")
                print(input_for_planning[0]['content'], flush=True)

            response = self.gpt_chat(processed_input=input_for_planning)

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(f"[SolidCoder] Planning Response: {plan_no}\n\n")
                print(response, flush=True)

            if "### Plan" not in response:
                plan = f"### Plan\n\n{response}"
            else:
                plan = response[response.rfind("### Plan"):]

            problem_with_planning = f"## Problem:\n{problem}\n\n{plan}"

            # ============================================================
            # PHASE 1b: PLAN SIMULATION (from CodeSIM)
            # ============================================================
            input_for_simulation = [
                {
                    "role": "user",
                    "content": prompt_for_simulation.format(
                        problem_with_planning=problem_with_planning,
                        language=self.language,
                    )
                },
            ]

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(f"[SolidCoder] Plan Simulation: {plan_no}\n\n")
                print(input_for_simulation[0]['content'], flush=True)

            response = self.gpt_chat(processed_input=input_for_simulation)

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(f"[SolidCoder] Simulation Response: {plan_no}\n\n")
                print(response, flush=True)

            # ============================================================
            # PHASE 1c: PLAN REFINEMENT (from CodeSIM)
            # ============================================================
            if "Plan Modification Needed" in response and \
                "No Plan Modification Needed" not in response:
                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(f"[SolidCoder] **Plan Modification Needed.**\n")

                input_for_plan_refinement = [
                    {
                        "role": "user",
                        "content": prompt_for_plan_refinement.format(
                            problem_with_planning=problem_with_planning,
                            language=self.language,
                            critique=response
                        )
                    },
                ]

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(f"[SolidCoder] Plan Refinement: {plan_no}\n\n")
                    print(input_for_plan_refinement[0]['content'], flush=True)

                plan = self.gpt_chat(processed_input=input_for_plan_refinement)

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(f"[SolidCoder] Refined Plan: {plan_no}\n\n")
                    print(plan, flush=True)

                problem_with_planning = f"## Problem:\n{problem}\n\n{plan}"

            # ============================================================
            # PHASE 2: CODE GENERATION (from CodeSIM)
            # ============================================================
            input_for_final_code_generation = [
                {
                    "role": "user",
                    "content": prompt_for_code_generation.format(
                        problem_with_planning=problem_with_planning,
                        language=self.language,
                        std_input_prompt=std_input_prompt,
                    )
                }
            ]

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(f"[SolidCoder] Code Generation:\n\n")
                print(input_for_final_code_generation[0]['content'], flush=True)

            response = self.gpt_chat(input_for_final_code_generation)

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(f"[SolidCoder] Generated Code:\n\n")
                print(response, flush=True)

            code = parse_response(response)

            # ============================================================
            # PHASE 3: INTERMEDIATE SIMULATION (optional)
            # ============================================================
            if self.enable_inter_sim:
                if self.verbose >= VERBOSE_FULL:
                    print("\n[SolidCoder] Intermediate Code Simulation\n")

                prompt_code_sim = f"""
## Problem
{problem}

## Code
```{self.language}
{code}
```

## Your Task: Mental Execution
1. Select a sample input.
2. Trace the code execution step-by-step.
3. Track variable values.
4. Predict the final output.

If you find a logic error or mismatch with the plan, output: **CODE_SIMULATION_FAILED**
If the code seems correct, output: **CODE_SIMULATION_PASSED**
"""
                code_sim_result = self.gpt_chat(processed_input=[{"role": "user", "content": prompt_code_sim}])

                if self.verbose >= VERBOSE_FULL:
                    print(f"[SolidCoder] Code Simulation Result:\n{code_sim_result}\n")

                if "CODE_SIMULATION_FAILED" in code_sim_result and \
                    "CODE_SIMULATION_PASSED" not in code_sim_result:
                    if self.verbose >= VERBOSE_FULL:
                        print(f"\n[SolidCoder] Fixing Code based on Simulation\n")

                    prompt_fix_sim = f"""
## Problem
{problem}

## Code
```{self.language}
{code}
```

## Simulation Result
{code_sim_result}

## Task
Fix the code based on the simulation failure.
Wrap the code in ```{self.language} ... ``` block.
"""
                    code_response = self.gpt_chat(processed_input=[{"role": "user", "content": prompt_fix_sim}])
                    code = parse_response(code_response)

            # ============================================================
            # PHASE 4: ASSUMPTION BREAKING (Live or Mental)
            # ============================================================
            code = self._run_assumption_breaking(problem, code, std_input_prompt, additional_io)

            # ============================================================
            # PHASE 5: TESTING (from CodeSIM)
            # ============================================================
            passed, test_log = self.check(data_row, additional_io, code)

            if passed:
                if self.verbose >= VERBOSE_FULL:
                    print("\n[SolidCoder] PASSED after self-check\n")
                break

            # ============================================================
            # PHASE 6: DEBUGGING (from CodeSIM)
            # ============================================================
            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print("[SolidCoder] Entering Debug Phase\n", flush=True)

            for debug_no in range(1, self.max_debug_try + 1):
                input_for_debugging = [
                    {
                        "role": "user",
                        "content": prompt_for_debugging.format(
                            problem_with_planning=problem_with_planning,
                            code=code,
                            language=self.language,
                            test_log=test_log,
                            std_input_prompt=std_input_prompt,
                        )
                    }
                ]

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(f"[SolidCoder] Debug: Plan {plan_no}, Debug {debug_no}\n\n")
                    print(input_for_debugging[0]['content'], flush=True)

                response = self.gpt_chat(input_for_debugging)

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(f"[SolidCoder] Debug Response:\n\n")
                    print(response, flush=True)

                code = parse_response(response)

                passed, test_log = self.check(data_row, additional_io, code)

                if passed:
                    break

            if passed:
                break

        if self.verbose >= VERBOSE_FULL:
            print("\n\n" + "_" * 70)

        return code

    def _run_assumption_breaking(self, problem: str, code: str, std_input_prompt: str, additional_io: list[str]) -> str:
        """Run mental or live assumption breaking rounds and return possibly fixed code."""
        if self.max_assumption_rounds <= 0:
            return code

        is_python = self.language.lower().startswith("python")

        for ab_round in range(1, self.max_assumption_rounds + 1):
            if self.verbose >= VERBOSE_FULL:
                print("\n[SolidCoder] Assumption Breaking Round "
                      f"{ab_round}/{self.max_assumption_rounds}\n")

            # Live verification only makes sense for Python code
            if self.enable_live_verify and is_python:
                attack_prompt = self._build_live_assumption_prompt(problem, code, oracle=self.enable_oracle_assert)
                attack_result = self.gpt_chat(processed_input=[{"role": "user", "content": attack_prompt}])

                if self.verbose >= VERBOSE_FULL:
                    print(f"[SolidCoder] Live Attack Result:\n{attack_result}\n")

                test_script = self._extract_test_script(attack_result)
                if not test_script:
                    if self.verbose >= VERBOSE_FULL:
                        print("[SolidCoder] No runnable test script parsed; skipping fix.\n")
                    break

                # JUDGE STEP: Verify if the test case is valid
                is_valid_test = self._judge_test_case(problem, test_script)
                if not is_valid_test:
                    if self.verbose >= VERBOSE_FULL:
                        print("[SolidCoder] Judge rejected the test case as INVALID.\n")
                    continue

                verify_status = self._concrete_verify_script(code, test_script)
                if verify_status in {"FAIL_CRASH", "FAIL_ASSERT"}:
                    if self.enable_defensive_test:
                        self.accumulated_inputs.append(test_script)
                        additional_io.append(test_script)
                    code = self._fix_with_test_script(problem, code, test_script)
                    continue

                # Robust or non-actionable -> stop
                break

            else:
                # Mental assumption breaking (AssumeCoder prompt)
                attack_prompt = prompt_for_assumption_breaking.format(
                    problem=problem,
                    code=code,
                    language=self.language,
                )
                attack_result = self.gpt_chat(processed_input=[{"role": "user", "content": attack_prompt}])

                if self.verbose >= VERBOSE_FULL:
                    print(f"[SolidCoder] Mental Attack Result:\n{attack_result}\n")

                is_pass = re.search(r'\bPASS\b', attack_result, re.IGNORECASE)
                is_fail = re.search(r'\bFAIL\b', attack_result, re.IGNORECASE)

                if is_pass and not is_fail:
                    break
                if not is_fail and not is_pass:
                    # Ambiguous response: avoid pointless fix
                    break
                
                # For mental mode, we can also apply a "Mental Judge" if needed, 
                # but let's focus on Live Verify first as it's more critical.

                fix_prompt = prompt_for_assumption_fix.format(
                    problem=problem,
                    code=code,
                    attack_result=attack_result,
                    language=self.language,
                    std_input_prompt=std_input_prompt,
                )
                code = parse_response(self.gpt_chat(processed_input=[{"role": "user", "content": fix_prompt}]))

        return code

    def _judge_test_case(self, problem: str, test_script: str) -> bool:
        """Ask the LLM (Judge) if the proposed test case is valid according to the problem."""
        prompt_judge = f"""
You are an impartial Judge.
## Problem
{problem}

## Proposed Test Case
```python
{test_script}
```

## Task
Determine if this test case is VALID and CORRECT for the given problem.
1. Does the input satisfy all constraints (e.g. range, type, format)?
2. Is the asserted output (if any) logically correct?
3. Is it a fair test?

If the test case is valid, output: **VALID**
If it violates constraints or expects wrong output, output: **INVALID**
"""
        response = self.gpt_chat(processed_input=[{"role": "user", "content": prompt_judge}])
        
        if self.verbose >= VERBOSE_FULL:
            print(f"[SolidCoder] Judge Verdict: {response}\n")

        return "VALID" in response and "INVALID" not in response

    def _build_live_assumption_prompt(self, problem: str, code: str, oracle: bool) -> str:
        """Create prompt asking for a runnable Python test script to break the code."""
        if oracle:
            return f"""
You are an expert software tester (Red Team).
Your goal is to break the following code by finding hidden assumptions.

## Problem
{problem}

## Code
```{self.language}
{code}
```

## Your Task
1. Identify a weak assumption (Type, Value, Structure, or Relationship).
2. Produce a **Python test script** that calls the target function with a breaking input.
3. Include an assert that would fail if the assumption is violated.

Format:
Assumption: <short text>
Test Script:
```python
# call the function defined above
result = <call>
assert <oracle about result>
```
"""
        return f"""
You are an expert software tester (Red Team).
Your goal is to break the following code by finding hidden assumptions.

## Problem
{problem}

## Code
```{self.language}
{code}
```

## Your Task
1. Identify a weak assumption (Type, Value, Structure, or Relationship).
2. Produce a **Python test script** that calls the target function with a breaking input.
3. If you don't know the exact oracle, just call it; any crash is a failure.

Format:
Assumption: <short text>
Test Script:
```python
<call the function with the breaking input>
```
"""

    def _extract_test_script(self, attack_result: str) -> Optional[str]:
        """Extract python test script from the LLM attack response."""
        script_match = re.search(
            r"Test Script:\s*```(?:python)?\s*(.*?)```",
            attack_result,
            re.DOTALL | re.IGNORECASE,
        )
        if script_match:
            return script_match.group(1).strip()

        # Fallback: grab the first python fenced block
        generic_block = re.search(r"```python\s*(.*?)```", attack_result, re.DOTALL | re.IGNORECASE)
        if generic_block:
            return generic_block.group(1).strip()
        return None

    def _fix_with_test_script(self, problem: str, code: str, test_script_str: str) -> str:
        """Ask the LLM to patch code based on a failing test script."""
        prompt_fix = f"""
## Problem
{problem}

## Code
```{self.language}
{code}
```

## Vulnerability Found
The following test script failed (Crash or Assertion Error):
```python
{test_script_str}
```

## Task
Fix the code to handle this edge case.
Wrap the code in ```{self.language} ... ``` block.
"""
        code_response = self.gpt_chat(processed_input=[{"role": "user", "content": prompt_fix}])
        return parse_response(code_response)

    def _concrete_verify_script(self, code, test_script, timeout: int = 5):
        if not self.language.lower().startswith("python"):
            return "SKIP_NON_PY"

        full_code = f"""
import sys
import math
from typing import List, Dict, Any, Optional, Union, Tuple

{code}

{test_script}
"""
        try:
            # Create a new global namespace for execution with input disabled
            safe_builtins = builtins.__dict__.copy()
            safe_builtins["input"] = lambda *_, **__: (_ for _ in ()).throw(RuntimeError("input() disabled during live verify"))
            exec_globals = {"__builtins__": safe_builtins}
            function_with_timeout(exec, (full_code, exec_globals), timeout)
            return "PASS"
        except AssertionError:
            return "FAIL_ASSERT"
        except Exception:
            return "FAIL_CRASH"
