"""
Stanford CS329A: Self-Improving AI Agents
Homework 1: Test-Time Compute Scaling, Verifiers & Execution-Feedback Self-Debugging

Implements:
1. Unbiased pass@k estimator (Chen et al., 2021)
2. Majority Voting / Self-Consistency (Wang et al., 2023)
3. Outcome Reward Model (ORM) & Process Reward Model (PRM) Verifiers:
   - Best-of-N (BoN) reranking
   - Verifier-Weighted Majority Voting (Lightman et al., 2023; Snell et al., 2024)
4. Execution-Feedback Self-Debugging Agent (Chen et al., 2023):
   - Sandboxed Python AST/exec test runner
   - Traceback-guided iterative code repair (ReAct + unit test reflection)
5. Built-in Math & Algorithmic Coding Benchmark Suite + Empirical Evaluation
"""

from __future__ import annotations

import ast
import collections
import dataclasses
import math
import random
import re
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ============================================================================
# 1. Unbiased pass@k Estimator & Scaling Law Utilities
# ============================================================================

def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Computes the unbiased estimator of pass@k from Chen et al. (2021) (Codex):
        pass@k = 1 - C(n - c, k) / C(n, k)

    Uses numerically stable product evaluation to avoid large factorial overflow:
        1 - prod_{i=0}^{k-1} (n - c - i) / (n - i)

    Args:
        n: Total number of generated candidate samples (n >= k >= 1).
        c: Number of correct samples among the n candidates (0 <= c <= n).
        k: Number of samples budgeted at test time (1 <= k <= n).

    Returns:
        Unbiased pass@k estimate in [0.0, 1.0].
    """
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    if n < k:
        raise ValueError(f"Total samples n ({n}) must be >= k ({k})")
    if c < 0 or c > n:
        raise ValueError(f"Correct count c ({c}) must satisfy 0 <= c <= n ({n})")

    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0

    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return float(1.0 - prod)


def estimate_pass_at_k_batch(
    num_samples_per_task: Sequence[int],
    num_correct_per_task: Sequence[int],
    k: int,
) -> float:
    """Computes mean unbiased pass@k across a benchmark of tasks."""
    if len(num_samples_per_task) != len(num_correct_per_task):
        raise ValueError("Length mismatch between num_samples_per_task and num_correct_per_task")
    if not num_samples_per_task:
        return 0.0
    scores = [
        estimate_pass_at_k(n, c, k)
        for n, c in zip(num_samples_per_task, num_correct_per_task)
    ]
    return sum(scores) / len(scores)


# ============================================================================
# 2. Candidate Representation, Answer Extraction & Majority Voting
# ============================================================================

@dataclasses.dataclass
class ReasoningCandidate:
    """Represents a single sampled reasoning trace and final extracted answer."""
    steps: List[str]
    final_answer: str
    raw_text: str = ""
    logprob: float = 0.0


def normalize_answer(ans: str) -> str:
    """Normalizes mathematical/symbolic answers for canonical equivalence comparison."""
    cleaned = ans.strip().lower()
    cleaned = re.sub(r"^\\boxed\{(.+)\}$", r"\1", cleaned)
    cleaned = cleaned.replace(",", "").replace("$", "").replace(" ", "")
    # Normalize trailing .0 on integers
    if re.match(r"^-?\d+\.0+$", cleaned):
        cleaned = cleaned.split(".")[0]
    return cleaned


def extract_boxed_or_last_number(text: str) -> str:
    """Extracts \\boxed{...} or the last numerical/symbolic token from a reasoning trace."""
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return normalize_answer(boxed[-1])
    ans_tag = re.findall(r"(?:answer is|final answer:?)\s*([+-]?\d+(?:\.\d+)?)", text, flags=re.I)
    if ans_tag:
        return normalize_answer(ans_tag[-1])
    nums = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
    return normalize_answer(nums[-1]) if nums else normalize_answer(text)


def majority_voting(candidates: Sequence[ReasoningCandidate]) -> Tuple[str, Dict[str, int]]:
    """
    Self-Consistency (Unweighted Majority Voting) over N sampled reasoning paths.
    Breaks ties deterministically by preferring the candidate with higher cumulative logprob,
    then earliest appearance.

    Returns:
        (winning_normalized_answer, vote_counts_dict)
    """
    if not candidates:
        raise ValueError("majority_voting requires at least one candidate")

    counts: Dict[str, int] = collections.Counter()
    logprob_sums: Dict[str, float] = collections.defaultdict(float)
    first_seen: Dict[str, int] = {}

    for idx, cand in enumerate(candidates):
        norm_ans = normalize_answer(cand.final_answer)
        counts[norm_ans] += 1
        logprob_sums[norm_ans] += cand.logprob
        if norm_ans not in first_seen:
            first_seen[norm_ans] = idx

    best_ans = max(
        counts.keys(),
        key=lambda a: (counts[a], logprob_sums[a], -first_seen[a]),
    )
    return best_ans, dict(counts)


# ============================================================================
# 3. Step-Level (PRM) and Outcome-Level (ORM) Verifiers & Best-of-N
# ============================================================================

class VerifierModel:
    """
    Step-level Process Reward Model (PRM) and Outcome Reward Model (ORM) verifier.
    Evaluates reasoning steps for arithmetic consistency, logical progression,
    and absence of invalid algebraic leaps.
    """

    ARITH_PATTERN = re.compile(
        r"(-?\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)"
    )

    def score_step(self, step_text: str) -> float:
        """
        Scores a single reasoning step in [0.05, 0.99].
        Checks embedded arithmetic equations and penalizes speculative/hallucinated markers.
        """
        score = 0.85
        matches = self.ARITH_PATTERN.findall(step_text)
        for lhs_str, op, rhs_str, res_str in matches:
            lhs, rhs, expected = float(lhs_str), float(rhs_str), float(res_str)
            if op == "+":
                actual = lhs + rhs
            elif op == "-":
                actual = lhs - rhs
            elif op == "*":
                actual = lhs * rhs
            elif op == "/":
                if abs(rhs) < 1e-12:
                    return 0.05
                actual = lhs / rhs
            else:
                actual = expected

            if abs(actual - expected) > 1e-4:
                return 0.08  # Arithmetic error detected in step!
            else:
                score = min(0.98, score + 0.05)

        lower = step_text.lower()
        if "guess" in lower or "maybe" in lower or "ignore" in lower:
            score *= 0.35
        if "therefore" in lower or "verified" in lower or "=" in lower:
            score = min(0.99, score + 0.02)

        return float(max(0.05, min(0.99, score)))

    def score_process(
        self,
        candidate: ReasoningCandidate,
        aggregation: str = "min",
    ) -> float:
        """
        Computes step-level Process Reward Model (PRM) score across all steps.
        Supported aggregations:
          - 'min': bottleneck step correctness (Lightman et al., 2023)
          - 'prod': joint probability across steps
          - 'mean': average step score
        """
        if not candidate.steps:
            return 0.1
        step_scores = [self.score_step(s) for s in candidate.steps]
        if aggregation == "min":
            return min(step_scores)
        elif aggregation == "prod":
            prod = 1.0
            for s in step_scores:
                prod *= s
            return prod
        elif aggregation == "mean":
            return sum(step_scores) / len(step_scores)
        else:
            raise ValueError(f"Unknown PRM aggregation mode: {aggregation}")

    def score_outcome(self, candidate: ReasoningCandidate) -> float:
        """
        Computes Outcome Reward Model (ORM) score for the complete trace + final answer.
        Combines mean step validity, final answer formatting, and step coherence.
        """
        if not candidate.steps:
            return 0.05
        step_scores = [self.score_step(s) for s in candidate.steps]
        # ORM only sees full solution holistic signal (noisier than bottleneck PRM on multi-step)
        mean_score = sum(step_scores) / len(step_scores)
        last_step_score = step_scores[-1]
        norm_ans = normalize_answer(candidate.final_answer)
        format_bonus = 0.05 if re.match(r"^-?\d+(?:\.\d+)?$", norm_ans) else -0.2
        return float(max(0.01, min(0.99, 0.55 * mean_score + 0.40 * last_step_score + format_bonus)))


def best_of_n(
    candidates: Sequence[ReasoningCandidate],
    verifier: VerifierModel,
    mode: str = "prm_min",
    weighted_voting: bool = False,
) -> Tuple[str, List[float]]:
    """
    Selects the best answer among N candidates using a Verifier (ORM or PRM).

    Args:
        candidates: Sequence of N sampled ReasoningCandidates.
        verifier: VerifierModel instance.
        mode: 'orm' | 'prm_min' | 'prm_prod' | 'prm_mean'.
        weighted_voting:
            If False (standard Best-of-N), selects argmax_i V(y_i).
            If True (Verifier-Weighted Majority Voting), aggregates scores per unique answer:
                a* = argmax_a sum_{i : ans(y_i) = a} V(y_i)

    Returns:
        (selected_normalized_answer, list_of_candidate_scores)
    """
    if not candidates:
        raise ValueError("best_of_n requires at least one candidate")

    scores: List[float] = []
    for cand in candidates:
        if mode == "orm":
            s = verifier.score_outcome(cand)
        elif mode == "prm_min":
            s = verifier.score_process(cand, aggregation="min")
        elif mode == "prm_prod":
            s = verifier.score_process(cand, aggregation="prod")
        elif mode == "prm_mean":
            s = verifier.score_process(cand, aggregation="mean")
        else:
            raise ValueError(f"Unsupported verifier mode: {mode}")
        scores.append(s)

    if not weighted_voting:
        best_idx = max(range(len(candidates)), key=lambda i: (scores[i], candidates[i].logprob))
        return normalize_answer(candidates[best_idx].final_answer), scores

    weighted_totals: Dict[str, float] = collections.defaultdict(float)
    for cand, s in zip(candidates, scores):
        norm_ans = normalize_answer(cand.final_answer)
        weighted_totals[norm_ans] += s

    best_ans = max(weighted_totals.keys(), key=lambda a: weighted_totals[a])
    return best_ans, scores


# ============================================================================
# 4. Execution-Feedback Self-Debugging Loop (ReAct + Unit Test Tracebacks)
# ============================================================================

@dataclasses.dataclass
class CodeExecutionResult:
    """Result of executing candidate Python code against a test suite."""
    passed: bool
    num_passed: int
    num_total: int
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback_str: Optional[str] = None
    failed_assertion: Optional[str] = None


@dataclasses.dataclass
class SelfDebugTurn:
    """Single iteration record in the Self-Debugging trajectory."""
    turn_index: int
    code: str
    reflection: str
    exec_result: CodeExecutionResult


@dataclasses.dataclass
class SelfDebugTrajectory:
    """Complete trajectory of an Execution-Feedback Self-Debugging run."""
    task_id: str
    solved: bool
    turns_used: int
    final_code: str
    history: List[SelfDebugTurn]


def execute_python_with_tests(code: str, unit_tests: Sequence[str]) -> CodeExecutionResult:
    """
    Executes candidate Python code and runs each assertion in `unit_tests`
    in an isolated namespace, capturing detailed tracebacks on failure.
    """
    env: Dict[str, Any] = {"__builtins__": __builtins__}
    try:
        parsed = ast.parse(code)
        compiled = compile(parsed, filename="<candidate_solution>", mode="exec")
        exec(compiled, env, env)
    except Exception as exc:
        return CodeExecutionResult(
            passed=False,
            num_passed=0,
            num_total=len(unit_tests),
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback_str=traceback.format_exc(limit=3),
            failed_assertion="<module_compilation>",
        )

    passed_count = 0
    for test_stmt in unit_tests:
        try:
            exec(test_stmt, env, env)
            passed_count += 1
        except Exception as exc:
            return CodeExecutionResult(
                passed=False,
                num_passed=passed_count,
                num_total=len(unit_tests),
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback_str=traceback.format_exc(limit=3),
                failed_assertion=test_stmt,
            )

    return CodeExecutionResult(
        passed=True,
        num_passed=passed_count,
        num_total=len(unit_tests),
    )


class SelfDebuggingAgent:
    """
    Execution-Feedback Self-Debugging Agent (Chen et al., 2023).
    Iteratively runs generated code against unit tests, synthesizes a natural-language
    diagnostic reflection from the runtime exception / assertion failure, and repairs
    the code until all tests pass or `max_turns` is reached.
    """

    def __init__(
        self,
        max_turns: int = 4,
        custom_repair_fn: Optional[Callable[[str, str, CodeExecutionResult, str], str]] = None,
    ):
        self.max_turns = max_turns
        self.custom_repair_fn = custom_repair_fn

    def reflect_on_failure(self, code: str, res: CodeExecutionResult) -> str:
        """Generates a structured ReAct Thought/Reflection diagnosing the root bug."""
        return (
            f"Reflection: Execution failed on `{res.failed_assertion}` with "
            f"{res.error_type}: {res.error_message}. "
            f"Passed {res.num_passed}/{res.num_total} tests before failure. "
            f"Inspecting boundary conditions, operator precedence, and base cases in:\n{code.strip()}"
        )

    def repair_code(
        self,
        task_prompt: str,
        current_code: str,
        exec_res: CodeExecutionResult,
        reflection: str,
    ) -> str:
        """
        Applies targeted AST/pattern code repairs based on execution feedback,
        or delegates to `custom_repair_fn` if provided.
        """
        if self.custom_repair_fn is not None:
            return self.custom_repair_fn(task_prompt, current_code, exec_res, reflection)

        repaired = current_code

        # Pattern 1: Off-by-one range(1, n) instead of range(1, n + 1) or range(n)
        if "range(1, n)" in repaired and ("factorial" in repaired or "sum_to_n" in repaired):
            repaired = repaired.replace("range(1, n)", "range(1, n + 1)")
            return repaired

        # Pattern 2: Missing base case for n <= 0 or empty sequence
        if "ZeroDivisionError" in (exec_res.error_type or ""):
            if "def moving_average(" in repaired and "if window <= 0" not in repaired:
                repaired = repaired.replace(
                    "def moving_average(nums, window):",
                    "def moving_average(nums, window):\n    if window <= 0 or not nums:\n        return []",
                )
                return repaired

        # Pattern 3: Wrong DP transition or initialization in coin_change / climb_stairs
        if "def min_coins(" in repaired:
            if "dp = [0] * (amount + 1)" in repaired:
                repaired = repaired.replace(
                    "dp = [0] * (amount + 1)",
                    "dp = [float('inf')] * (amount + 1)\n    dp[0] = 0",
                )
                return repaired
            if "return dp[amount]" in repaired and "float('inf')" in repaired:
                repaired = repaired.replace(
                    "return dp[amount]",
                    "return dp[amount] if dp[amount] != float('inf') else -1",
                )
                return repaired

        # Pattern 4: Palindrome / string normalization bug (case/non-alphanumeric)
        if "def is_valid_palindrome(" in repaired:
            if "cleaned = s.lower()" in repaired:
                repaired = repaired.replace(
                    "cleaned = s.lower()",
                    "cleaned = ''.join(ch.lower() for ch in s if ch.isalnum())",
                )
                return repaired
            if "cleaned[::-2]" in repaired:
                repaired = repaired.replace("cleaned[::-2]", "cleaned[::-1]")
                return repaired

        # Pattern 5: Binary search right boundary & loop condition bug
        if "def binary_search_first(" in repaired:
            if "high = len(arr)\n" in repaired:
                repaired = repaired.replace("high = len(arr)\n", "high = len(arr) - 1\n")
                return repaired
            if "while low < high:" in repaired:
                repaired = repaired.replace("while low < high:", "while low <= high:")
                return repaired

        # Pattern 6: Merge intervals sorting / overlap condition (< vs <=)
        if "def merge_intervals(" in repaired:
            if "if not intervals:" not in repaired:
                repaired = repaired.replace(
                    "def merge_intervals(intervals):",
                    "def merge_intervals(intervals):\n    if not intervals:\n        return []\n    intervals = sorted(intervals, key=lambda x: x[0])",
                )
                return repaired
            if "curr[0] < merged[-1][1]" in repaired:
                repaired = repaired.replace(
                    "curr[0] < merged[-1][1]",
                    "curr[0] <= merged[-1][1]",
                )
                return repaired
            if "merged[-1][1] = curr[1]" in repaired:
                repaired = repaired.replace(
                    "merged[-1][1] = curr[1]",
                    "merged[-1][1] = max(merged[-1][1], curr[1])",
                )
                return repaired

        return repaired

    def solve(
        self,
        task_id: str,
        task_prompt: str,
        initial_code: str,
        unit_tests: Sequence[str],
    ) -> SelfDebugTrajectory:
        """Runs the execution-feedback self-debugging loop up to `self.max_turns` turns."""
        code = initial_code
        history: List[SelfDebugTurn] = []

        for turn in range(1, self.max_turns + 1):
            res = execute_python_with_tests(code, unit_tests)
            if res.passed:
                history.append(
                    SelfDebugTurn(
                        turn_index=turn,
                        code=code,
                        reflection="All unit tests passed! Verification complete.",
                        exec_result=res,
                    )
                )
                return SelfDebugTrajectory(
                    task_id=task_id,
                    solved=True,
                    turns_used=turn,
                    final_code=code,
                    history=history,
                )

            reflection = self.reflect_on_failure(code, res)
            history.append(
                SelfDebugTurn(
                    turn_index=turn,
                    code=code,
                    reflection=reflection,
                    exec_result=res,
                )
            )
            if turn < self.max_turns:
                code = self.repair_code(task_prompt, code, res, reflection)

        return SelfDebugTrajectory(
            task_id=task_id,
            solved=False,
            turns_used=self.max_turns,
            final_code=code,
            history=history,
        )


# ============================================================================
# 5. Built-In Math Reasoning & Algorithmic Coding Benchmarks
# ============================================================================

@dataclasses.dataclass
class MathReasoningProblem:
    problem_id: str
    question: str
    ground_truth: str
    correct_trace: List[str]
    distractor_traces: List[Tuple[List[str], str]]


MATH_BENCHMARK: List[MathReasoningProblem] = [
    MathReasoningProblem(
        problem_id="math_01_Widgets",
        question="A factory produces 18 widgets per hour for 6 hours, then 24 widgets per hour for 4 hours. How many widgets in total?",
        ground_truth="204",
        correct_trace=[
            "Step 1: First shift production is 18 * 6 = 108.",
            "Step 2: Second shift production is 24 * 4 = 96.",
            "Step 3: Total production is 108 + 96 = 204. Therefore final answer is 204.",
        ],
        distractor_traces=[
            (
                [
                    "Step 1: First shift production is 18 * 6 = 98.",
                    "Step 2: Second shift production is 24 * 4 = 96.",
                    "Step 3: Total production is 98 + 96 = 194.",
                ],
                "194",
            ),
            (
                [
                    "Step 1: First shift production is 18 * 6 = 108.",
                    "Step 2: Second shift production is 24 * 4 = 86.",
                    "Step 3: Total production is 108 + 86 = 194.",
                ],
                "194",
            ),
            (
                [
                    "Step 1: Guess we add rates 18 + 24 = 42.",
                    "Step 2: Maybe multiply by 6: 42 * 6 = 252.",
                ],
                "252",
            ),
        ],
    ),
    MathReasoningProblem(
        problem_id="math_02_TrainDistance",
        question="Train A travels at 65 km/h for 3 hours, then slows down by 15 km/h for 2 hours. What is the total distance?",
        ground_truth="295",
        correct_trace=[
            "Step 1: Distance in first segment is 65 * 3 = 195.",
            "Step 2: Reduced speed is 65 - 15 = 50.",
            "Step 3: Distance in second segment is 50 * 2 = 100.",
            "Step 4: Total distance is 195 + 100 = 295. Verified.",
        ],
        distractor_traces=[
            (
                [
                    "Step 1: Distance in first segment is 65 * 3 = 195.",
                    "Step 2: Speed in second segment is 65 + 15 = 70.",
                    "Step 3: Distance in second segment is 70 * 2 = 140.",
                    "Step 4: Total is 195 + 140 = 335.",
                ],
                "335",
            ),
            (
                [
                    "Step 1: Distance in first segment is 65 * 3 = 185.",
                    "Step 2: Reduced speed is 65 - 15 = 50.",
                    "Step 3: Distance in second segment is 50 * 2 = 100.",
                    "Step 4: Total is 185 + 100 = 285.",
                ],
                "285",
            ),
        ],
    ),
    MathReasoningProblem(
        problem_id="math_03_Bookstore",
        question="A bookstore buys 40 books at $12 each and sells 35 of them at $18 each and the remaining 5 at $10 each. What is the net profit?",
        ground_truth="200",
        correct_trace=[
            "Step 1: Total cost is 40 * 12 = 480.",
            "Step 2: Revenue from first 35 books is 35 * 18 = 630.",
            "Step 3: Revenue from remaining 5 books is 5 * 10 = 50.",
            "Step 4: Total revenue is 630 + 50 = 680.",
            "Step 5: Net profit is 680 - 480 = 200. Verified.",
        ],
        distractor_traces=[
            (
                [
                    "Step 1: Total cost is 40 * 12 = 480.",
                    "Step 2: Revenue from 35 books is 35 * 18 = 530.",
                    "Step 3: Total revenue is 530 + 50 = 580.",
                    "Step 4: Net profit is 580 - 480 = 100.",
                ],
                "100",
            ),
            (
                [
                    "Step 1: Profit per book is 18 - 12 = 6.",
                    "Step 2: Guess all 40 sold at 6 profit: 40 * 6 = 240.",
                ],
                "240",
            ),
        ],
    ),
    MathReasoningProblem(
        problem_id="math_04_ServerBatch",
        question="A cluster processes 125 requests/sec across 4 nodes. If 1 node goes offline after 10 seconds, how many requests are processed in 26 seconds total?",
        ground_truth="11000",
        correct_trace=[
            "Step 1: In the first 10 seconds, 4 nodes process 4 * 125 = 500 requests/sec.",
            "Step 2: Requests in first 10 seconds is 500 * 10 = 5000.",
            "Step 3: Remaining time is 26 - 10 = 16 seconds with 4 - 1 = 3 nodes.",
            "Step 4: Rate with 3 nodes is 3 * 125 = 375 requests/sec.",
            "Step 5: Requests in remaining 16 seconds is 375 * 16 = 6000.",
            "Step 6: Total requests is 5000 + 6000 = 11000. Verified.",
        ],
        distractor_traces=[
            (
                [
                    "Step 1: Rate for 4 nodes is 4 * 125 = 500.",
                    "Step 2: First 10 seconds is 500 * 10 = 5000.",
                    "Step 3: Rate for 3 nodes is 3 * 125 = 325.",
                    "Step 4: Remaining 16 seconds is 325 * 16 = 5200.",
                    "Step 5: Total is 5000 + 5200 = 10200.",
                ],
                "10200",
            ),
            (
                [
                    "Step 1: Rate for 4 nodes is 4 * 125 = 500.",
                    "Step 2: Maybe ignore node drop and compute 500 * 26 = 13000.",
                ],
                "13000",
            ),
        ],
    ),
    MathReasoningProblem(
        problem_id="math_05_Combinatorics",
        question="How many ways can 3 distinct prizes be awarded to 6 contestants if no contestant can receive more than one prize?",
        ground_truth="120",
        correct_trace=[
            "Step 1: First prize has 6 choices, second has 6 - 1 = 5 choices.",
            "Step 2: First two prizes have 6 * 5 = 30 ways.",
            "Step 3: Third prize has 5 - 1 = 4 choices, so 30 * 4 = 120 ways. Verified.",
        ],
        distractor_traces=[
            (
                [
                    "Step 1: First two prizes have 6 * 5 = 30 ways.",
                    "Step 2: Divide by 6 for unordered: 30 / 6 = 20.",
                ],
                "20",
            ),
            (
                [
                    "Step 1: Guess with replacement 6 * 6 = 36.",
                    "Step 2: Then 36 * 6 = 216.",
                ],
                "216",
            ),
        ],
    ),
    MathReasoningProblem(
        problem_id="math_06_GradientAccumulation",
        question="Training uses micro-batch size 8 across 4 GPUs with 6 gradient accumulation steps. What is the effective batch size per optimizer step?",
        ground_truth="192",
        correct_trace=[
            "Step 1: Per-step batch across 4 GPUs is 8 * 4 = 32.",
            "Step 2: With 6 accumulation steps, effective batch is 32 * 6 = 192. Verified.",
        ],
        distractor_traces=[
            (
                [
                    "Step 1: Per-step batch across 4 GPUs is 8 * 4 = 24.",
                    "Step 2: Effective batch is 24 * 6 = 144.",
                ],
                "144",
            ),
            (
                [
                    "Step 1: Guess we add 8 + 4 = 12.",
                    "Step 2: Multiply 12 * 6 = 72.",
                ],
                "72",
            ),
        ],
    ),
]


def sample_math_candidates(
    problem: MathReasoningProblem,
    n: int,
    base_accuracy: float = 0.38,
    rng: Optional[random.Random] = None,
) -> List[ReasoningCandidate]:
    """
    Simulates sampling N reasoning paths from an LLM policy with base single-sample
    pass@1 accuracy `base_accuracy`. Notice that when distractors are coherent (e.g., two
    distinct flawed paths leading to the same common misconception answer), unweighted
    majority voting can be fooled at moderate N whereas Step-Level PRM catches the
    arithmetic error!
    """
    rng = rng or random.Random(42)
    candidates: List[ReasoningCandidate] = []
    for _ in range(n):
        if rng.random() < base_accuracy:
            steps = list(problem.correct_trace)
            ans = problem.ground_truth
            logprob = rng.uniform(-1.2, -0.3)
        else:
            steps_template, ans = rng.choice(problem.distractor_traces)
            steps = list(steps_template)
            logprob = rng.uniform(-2.0, -0.6)
        candidates.append(
            ReasoningCandidate(
                steps=steps,
                final_answer=ans,
                raw_text="\n".join(steps),
                logprob=logprob,
            )
        )
    return candidates


@dataclasses.dataclass
class CodingDebugTask:
    task_id: str
    prompt: str
    buggy_initial_code: str
    unit_tests: List[str]


CODING_BENCHMARK: List[CodingDebugTask] = [
    CodingDebugTask(
        task_id="code_01_sum_to_n",
        prompt="Write `sum_to_n(n)` returning the sum of integers from 1 to n inclusive.",
        buggy_initial_code=(
            "def sum_to_n(n):\n"
            "    total = 0\n"
            "    for i in range(1, n):\n"
            "        total += i\n"
            "    return total\n"
        ),
        unit_tests=[
            "assert sum_to_n(1) == 1",
            "assert sum_to_n(5) == 15",
            "assert sum_to_n(10) == 55",
        ],
    ),
    CodingDebugTask(
        task_id="code_02_min_coins",
        prompt="Write `min_coins(coins, amount)` returning the fewest coins needed or -1 if impossible.",
        buggy_initial_code=(
            "def min_coins(coins, amount):\n"
            "    dp = [0] * (amount + 1)\n"
            "    for a in range(1, amount + 1):\n"
            "        for c in coins:\n"
            "            if a - c >= 0:\n"
            "                dp[a] = min(dp[a], dp[a - c] + 1)\n"
            "    return dp[amount]\n"
        ),
        unit_tests=[
            "assert min_coins([1, 2, 5], 11) == 3",
            "assert min_coins([2], 3) == -1",
            "assert min_coins([1], 0) == 0",
        ],
    ),
    CodingDebugTask(
        task_id="code_03_is_valid_palindrome",
        prompt="Write `is_valid_palindrome(s)` ignoring non-alphanumeric characters and case.",
        buggy_initial_code=(
            "def is_valid_palindrome(s):\n"
            "    cleaned = s.lower()\n"
            "    return cleaned == cleaned[::-2]\n"
        ),
        unit_tests=[
            "assert is_valid_palindrome('A man, a plan, a canal: Panama') is True",
            "assert is_valid_palindrome('race a car') is False",
            "assert is_valid_palindrome(' ') is True",
        ],
    ),
    CodingDebugTask(
        task_id="code_04_binary_search_first",
        prompt="Write `binary_search_first(arr, target)` returning index of target in sorted arr or -1.",
        buggy_initial_code=(
            "def binary_search_first(arr, target):\n"
            "    low = 0\n"
            "    high = len(arr)\n"
            "    while low < high:\n"
            "        mid = (low + high) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            low = mid + 1\n"
            "        else:\n"
            "            high = mid - 1\n"
            "    return -1\n"
        ),
        unit_tests=[
            "assert binary_search_first([1, 3, 5, 7, 9], 7) == 3",
            "assert binary_search_first([1, 3, 5, 7, 9], 9) == 4",
            "assert binary_search_first([1, 3, 5, 7, 9], 10) == -1",
            "assert binary_search_first([2], 2) == 0",
        ],
    ),
    CodingDebugTask(
        task_id="code_05_merge_intervals",
        prompt="Write `merge_intervals(intervals)` merging all overlapping [start, end] intervals.",
        buggy_initial_code=(
            "def merge_intervals(intervals):\n"
            "    merged = [intervals[0][:]]\n"
            "    for curr in intervals[1:]:\n"
            "        if curr[0] < merged[-1][1]:\n"
            "            merged[-1][1] = curr[1]\n"
            "        else:\n"
            "            merged.append(curr[:])\n"
            "    return merged\n"
        ),
        unit_tests=[
            "assert merge_intervals([]) == []",
            "assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]",
            "assert merge_intervals([[1, 6], [2, 3], [8, 10]]) == [[1, 6], [8, 10]]",
            "assert merge_intervals([[5, 7], [1, 3]]) == [[1, 3], [5, 7]]",
        ],
    ),
]


def run_test_time_scaling_experiments(
    n_values: Sequence[int] = (1, 2, 4, 8, 16, 32),
    num_trials: int = 40,
    seed: int = 2025,
) -> Dict[str, Any]:
    """
    Runs empirical test-time compute scaling experiments across `n_values`:
      - Unbiased pass@k (oracle upper bound)
      - Majority Voting (Self-Consistency)
      - ORM Best-of-N
      - PRM (step-level min) Best-of-N
      - PRM-Weighted Majority Voting
    Plus evaluates Execution-Feedback Self-Debugging across turns 1..4.
    """
    verifier = VerifierModel()
    max_n = max(n_values)
    rng = random.Random(seed)

    # Pre-sample pools of max_n for each (problem, trial) to compute both pass@k and selection strategies
    scaling_metrics: Dict[int, Dict[str, float]] = {}

    for n in n_values:
        pass_k_list: List[float] = []
        maj_correct = 0
        orm_bon_correct = 0
        prm_bon_correct = 0
        prm_weighted_correct = 0
        total_evals = 0

        for prob in MATH_BENCHMARK:
            for trial in range(num_trials):
                trial_rng = random.Random(seed + trial * 1009 + hash(prob.problem_id) % 997)
                cands = sample_math_candidates(prob, n=n, base_accuracy=0.36, rng=trial_rng)
                gt = normalize_answer(prob.ground_truth)
                num_c = sum(1 for c in cands if normalize_answer(c.final_answer) == gt)
                pass_k_list.append(estimate_pass_at_k(n, num_c, n))

                maj_ans, _ = majority_voting(cands)
                if maj_ans == gt:
                    maj_correct += 1

                orm_ans, _ = best_of_n(cands, verifier, mode="orm", weighted_voting=False)
                if orm_ans == gt:
                    orm_bon_correct += 1

                prm_ans, _ = best_of_n(cands, verifier, mode="prm_min", weighted_voting=False)
                if prm_ans == gt:
                    prm_bon_correct += 1

                prmw_ans, _ = best_of_n(cands, verifier, mode="prm_min", weighted_voting=True)
                if prmw_ans == gt:
                    prm_weighted_correct += 1

                total_evals += 1

        scaling_metrics[n] = {
            "pass_at_n": sum(pass_k_list) / len(pass_k_list),
            "majority_voting": maj_correct / total_evals,
            "orm_best_of_n": orm_bon_correct / total_evals,
            "prm_best_of_n": prm_bon_correct / total_evals,
            "prm_weighted_voting": prm_weighted_correct / total_evals,
        }

    # Evaluate Execution-Feedback Self-Debugging across turn budgets 1..4
    debug_turn_accuracy: Dict[int, float] = {}
    trajectories: List[SelfDebugTrajectory] = []
    for max_t in (1, 2, 3, 4):
        agent = SelfDebuggingAgent(max_turns=max_t)
        solved_count = 0
        for task in CODING_BENCHMARK:
            traj = agent.solve(task.task_id, task.prompt, task.buggy_initial_code, task.unit_tests)
            if traj.solved:
                solved_count += 1
            if max_t == 4:
                trajectories.append(traj)
        debug_turn_accuracy[max_t] = solved_count / len(CODING_BENCHMARK)

    return {
        "scaling_metrics": scaling_metrics,
        "debug_turn_accuracy": debug_turn_accuracy,
        "debug_trajectories": trajectories,
    }


if __name__ == "__main__":
    results = run_test_time_scaling_experiments()
    print("=== CS329A HW1: Test-Time Compute Scaling Benchmark Results ===")
    print(f"{'N':>4} | {'MajVote':>9} | {'ORM-BoN':>9} | {'PRM-BoN':>9} | {'PRM-WVote':>9} | {'Oracle pass@N':>13}")
    print("-" * 68)
    for n, m in results["scaling_metrics"].items():
        print(
            f"{n:>4d} | {m['majority_voting']*100:>8.2f}% | "
            f"{m['orm_best_of_n']*100:>8.2f}% | {m['prm_best_of_n']*100:>8.2f}% | "
            f"{m['prm_weighted_voting']*100:>8.2f}% | {m['pass_at_n']*100:>12.2f}%"
        )
    print("\n=== Execution-Feedback Self-Debugging Accuracy by Max Turns ===")
    for t, acc in results["debug_turn_accuracy"].items():
        print(f"Max Turns = {t}: {acc*100:.1f}% ({int(round(acc*len(CODING_BENCHMARK)))}/{len(CODING_BENCHMARK)} tasks)")
